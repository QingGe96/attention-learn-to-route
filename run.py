#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem


def run(opts):

    # Pretty print the run args
    # 打印参数信息
    pp.pprint(vars(opts))

    # Set the random seed
    # 设置随机种子
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    # 配置tensorboard，configure接受两个参数，第一个是存放目录，第二个是TensorBoard任务的名称
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    # 创建一个保存参数的文件夹(如果已经存在就不进行操作)
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    # 保存当前参数设置到save_dir文件夹下的args.json文件中
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    # 返回问题类
    problem = load_problem(opts.problem)

    # Load data from load_path
    # 如果需要，导入model和optimizer
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    # 初始化模型(策略网络), 同样返回对应的模型实例
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)   # 并行训练模型

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    # 根据opts中baseline的类型初始化baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    # 初始化优化器，如果使用的是critic网络作为baseline网络，则优化参数需要包含critic网络的参数
    # lr_model和lr_critic分别是model和critic网络(如果使用)的初始学习率
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    # 如果在load_data中有需要加载的优化器，则使用加载的优化器
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    # 每个epoch后学习率会衰减，衰减的倍数是opts.lr_decay ** epoch
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    # 构建测试数据集
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:        # 仅评估模型
        validate(model, val_dataset, opts)
    else:          # 进行训练，每个epoch会评估模型
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    run(get_options())
