import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateTSP(NamedTuple):
    """
    这段代码定义了一个名为 StateTSP 的类，用于表示解决旅行商问题（TSP）时的状态。

    包括以下成员变量：
    loc：旅行地点的位置向量，大小为 (batch_size, n_loc, feature_dim)。
    dist：旅行地点之间的距离矩阵，大小为 (batch_size, n_loc, n_loc)。
    ids：用于索引原始数据行的张量，大小为 (batch_size, 1)。
    first_a：第一步动作的张量，大小为 (batch_size, 1)。
    prev_a：上一步动作的张量，大小为 (batch_size, 1)。
    visited_：表示已访问节点的张量。
    lengths：路径长度的张量，大小为 (batch_size, 1)。
    cur_coord：当前坐标的张量，大小为 (batch_size, feature_dim)。
    i：用于记录步数的张量，大小为 (1,)。

    同时还包括了一些方法：
    visited：获取已访问节点的张量，返回类型是 torch.uint8 或者 bool。
    __getitem__：用于获取和更新部分数据的方法。
    initialize：用于初始化状态的静态方法。
    get_final_cost：用于获取最终路径长度的方法。
    update：用于更新状态的方法。
    all_finished：判断是否已经访问过所有节点的方法。
    get_current_node：获取当前节点的方法。
    get_mask：获取访问节点的掩码的方法。
    get_nn：获取最近的未访问节点的方法。
    get_nn_current：获取最近的当前节点的方法。
    construct_solutions：构建解决方案的方法
    """
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)   # (batch,1)的全0向量
        return StateTSP(
            loc=loc,   # 基础输入，(batch, lenth, feature_dim)
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),   # 距离矩阵
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # (batch, 1) Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        """
        更新prev_a(上一步动作)，cur_coord(当前坐标)，lengths(路径长度)，
        first_a(第一步动作，如果是第一次解码要更新)，visited_(已访问节点)
        :param selected:
        :return:
        """
        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.loc[self.ids, prev_a]
        lengths = self.lengths
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)
        # 由于状态是一个命名元组(NamedTuple)，不可被修改，调用_replace方法返回一个修改后的新元组
        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
            self.dist[
                self.ids,
                self.prev_a
            ] +
            self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
