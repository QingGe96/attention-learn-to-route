import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    """
    如果是并行模型，获取内部模型
    :param model:
    :return:
    """
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    """
    验证模型，返回平均损失
    :param model:
    :param dataset:
    :param opts:
    :return: avg_cost
    """
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    """
    评价模型在数据集上的性能，返回模型在每个batch上的路径长度(用于获得baseline)
    :param model: 需要评估的模型 net
    :param dataset: 训练数据集
    :param opts: 训练相关的一些设置，如使用GPU，相关参数等
    :return: 每个batch的平均损失(batch,)
    """
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)   # 将每个batch的cost cat成一个向量并返回


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param param_groups: 参加梯度剪裁的参数列表，列表中的每个元素是一个字典，其中键'params'对应的是模型需要更新的参数
    :param max_norm: 允许的范数最大值，超过这个值将被剪裁
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    # 记录剪裁前的梯度范数值
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    # 进行梯度剪裁
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped   # 返回未剪裁前的梯度范数值和剪裁后的梯度范数


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    """
    训练一个epoch，训练集每个epoch是重新随机生成的，所以不用输入
    val_dataset仅用于评估策略网络，判断是否要更新rollout baseline(如果使用)的数据集也是每个epoch随机生成(为了防止过拟合)
    :param model: 策略网络
    :param optimizer: 优化器
    :param baseline: baseline
    :param lr_scheduler:
    :param epoch: 当前epoch数(如果是第一次训练就是0)
    :param val_dataset: 测试数据集，在run.py中生成
    :param problem: 问题类型，TSP包含两个函数get_costs()和make_dataset(返回TSP数据集，如果有load则load，否则创建一个新的)
    :param opts: 参数设置
    :return:
    """
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    # 每个epoch使用一个新的训练数据集
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        # 保存每个epoch训练后的模型
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)    # 在测试集上的平均奖励

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)
    # 训练一个epoch后，尝试更新baseline
    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    """
    给定batch进行训练
    :param model:
    :param optimizer:
    :param baseline:
    :param epoch:
    :param batch_id:
    :param step:
    :param batch:
    :param opts:
    :return:
    """
    x, bl_val = baseline.unwrap_batch(batch)         # x表示批量样本， bl_val表示对应的baseline评估值
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    # 按照模型计算结果
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    # 除rollout baseline外bl_val在这里计算
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    # (模型输出实际长度-bl长度) * 轨迹的对数概率
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()    # batch上的平均损失(用样本近似策略梯度) scalar
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
