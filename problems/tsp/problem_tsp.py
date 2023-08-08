from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        """
            :param dataset: (batch_size, graph_size, node_dim)
            :param pi: tensor (batch_size, lenth)
            :return cost: (batch,)
                    mask: None
        """
        # Check that tours are valid, i.e. contain 0 to n -1
        # out=pi.data.new()指定新创建的张量使用与pi相同的数据类型和设备。
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        # 按照路径顺序排列dataset
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """
        :param args:
        :param kwargs:
                        filename=None, 指定.pkl文件，可以读取数据
                        size=50, 问题规模
                        num_samples=1000000 数据集大小
        :return: TSPDataset
        """
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """
        接受两个参数loc和visited_dtype。返回一个初始化后的StateTSP类
        :param args:
        :param kwargs:
        :return:
        """
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            # 如果指定.pkl文件，可以加载数据集
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            # 通过循环生成数据集，生成的格式是一个list，其中每个元素是二维tensor
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
