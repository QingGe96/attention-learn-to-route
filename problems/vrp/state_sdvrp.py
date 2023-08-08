import torch
from typing import NamedTuple


class StateSDVRP(NamedTuple):
    """
    StateSDVRP包含以下成员：
    coords: 表示地点的坐标的张量。
    demand: 表示各个地点的需求的张量。
    ids: 用于索引coords和demand张量中正确行的原始数据索引的张量。
    prev_a: 表示上一步选择的动作的张量。
    used_capacity: 表示已使用的车辆容量的张量。
    demands_with_depot: 表示剩余需求的张量。
    lengths: 表示已经完成路径长度的张量。
    cur_coord: 表示当前坐标的张量。
    i: 表示步数的张量。
    此外，StateSDVRP还包含了一些静态方法和实例方法，用于对状态进行初始化、更新状态、获取最终成本、判断是否已完成等操作。
    StateSDVRP的初始化方法initialize接受一个输入input作为参数，并根据输入的数据对状态进行初始化。
    get_final_cost方法用于计算最终的成本，其中包括已经完成的路径长度和当前坐标到起始坐标的欧氏距离。
    update方法用于更新状态，根据选择的动作对状态的各个成员进行更新。
    all_finished方法判断是否已经完成了所有的动作。
    get_current_node方法用于获取当前节点。
    get_mask方法用于获取可行动作的掩码，表示哪些动作是可行的，哪些是不可行的。
    construct_solutions方法用于构造解决方案。
    """
    # Fixed input
    coords: torch.Tensor
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    demands_with_depot: torch.Tensor  # Keeps track of remaining demands
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            demands_with_depot=self.demands_with_depot[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    @staticmethod
    def initialize(input):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']

        batch_size, n_loc, _ = loc.size()
        return StateSDVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            demands_with_depot=torch.cat((
                demand.new_zeros(batch_size, 1),
                demand[:, :]
            ), 1)[:, None, :],
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = self.demands_with_depot.gather(-1, prev_a[:, :, None])[:, :, 0]
        delivered_demand = torch.min(selected_demand, self.VEHICLE_CAPACITY - self.used_capacity)

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + delivered_demand)
        used_capacity = (self.used_capacity + delivered_demand) * (prev_a != 0).float()

        # demands_with_depot = demands_with_depot.clone()[:, 0, :]
        # Add one dimension since we write a single value
        demands_with_depot = self.demands_with_depot.scatter(
            -1,
            prev_a[:, :, None],
            self.demands_with_depot.gather(-1, prev_a[:, :, None]) - delivered_demand[:, :, None]
        )
        
        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, demands_with_depot=demands_with_depot,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.demands_with_depot.size(-1) and not (self.demands_with_depot > 0).any()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = (self.demands_with_depot[:, :, 1:] == 0) | (self.used_capacity[:, :, None] >= self.VEHICLE_CAPACITY)

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def construct_solutions(self, actions):
        return actions
