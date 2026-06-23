# CVRPEnv.py

from dataclasses import dataclass
import torch
from CVRProblemDef import generate_batched_pdtsp_instances, augment_xy_data_by_8_fold, calculate_cluster_means, training_augment

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    node_xy: torch.Tensor = None
    depot_aug: torch.Tensor = None
    node_aug: torch.Tensor = None
    node_demand: torch.Tensor = None
    node_demand_encode: torch.Tensor = None
    cluster_list: torch.Tensor = None
    cluster_means: torch.Tensor = None
    node_clus: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    selected_count: int = None
    load: torch.Tensor = None
    current_node: torch.Tensor = None
    current_cluster: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None
    in_cluster: torch.Tensor = None
    # Added field: the first visited node in the current cluster.
    first_node_in_cluster: torch.Tensor = None


class CVRPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params.get('problem_size', 0)
        self.pomo_size = env_params['pomo_size']
        self.distribution = env_params.get('distribution', 'uniform')

        self.pairing = None
        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_cluster_list = None
        self.saved_pairing = None

        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, depot_xy, node_xy, node_demand, cluster_list, pairing):
        self.FLAG__use_saved_problems = True
        self.saved_depot_xy = depot_xy
        self.saved_node_xy = node_xy
        self.saved_node_demand = node_demand
        self.saved_cluster_list = cluster_list
        self.saved_pairing = pairing
        self.problem_size = node_xy.size(1)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size
        device = self.env_params.get('device', 'cpu')

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand, cluster_list, _, self.pairing = \
                generate_batched_pdtsp_instances(batch_size, self.problem_size, device=device, distribution=self.distribution)
        else:
            depot_xy = self.saved_depot_xy.to(device)
            node_xy = self.saved_node_xy.to(device)
            node_demand = self.saved_node_demand.to(device)
            cluster_list = self.saved_cluster_list.to(device)
            self.pairing = self.saved_pairing.to(device)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                cluster_list = cluster_list.repeat(8, 1)
                self.pairing = self.pairing.repeat(8, 1)
            else:
                raise NotImplementedError

        node_demand_encode = node_demand / 1.0
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(size=(self.batch_size, 1), device=node_demand.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        depot_cluster = torch.zeros(size=(self.batch_size, 1), dtype=torch.long, device=cluster_list.device)
        self.cluster_list = torch.cat((depot_cluster, cluster_list), dim=1)

        all_coords = torch.cat((depot_xy, node_xy), dim=1)
        cluster_means = calculate_cluster_means(all_coords, self.cluster_list, 3)
        expanded_cluster_indices = self.cluster_list[:, 1:].unsqueeze(-1).expand(-1, -1, 2)
        node_clus = cluster_means.gather(1, expanded_cluster_indices)

        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.depot_aug = training_augment(depot_xy)
        self.reset_state.node_aug = training_augment(node_xy)
        self.reset_state.cluster_list = self.cluster_list[:, 1:]
        self.reset_state.node_demand = node_demand
        self.reset_state.node_clus = node_clus
        self.reset_state.node_demand_encode = node_demand_encode

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        device = self.depot_node_xy.device
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long, device=device)

        # Initialize the cluster-entry state at reset.
        # All routes start at the depot, so the first node in the cluster is depot index 0.
        self.first_node_in_cluster = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.long, device=device)

        self.delivery_available_mask = torch.full(size=(self.batch_size, self.pomo_size, self.problem_size + 1),
                                                  fill_value=float('-inf'), device=device)
        num_pickups = self.problem_size // 2
        self.delivery_available_mask[:, :, :num_pickups + 1] = 0

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1), device=device)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1), device=device)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=device)

        self.load = torch.ones(size=(self.batch_size, self.pomo_size), device=device)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        if self.selected_count == 0:
            self.in_cluster = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=self.ninf_mask.device)
        else:
            # Keep the in_cluster decision logic unchanged.
            is_pickup_node_mask = (self.cluster_list == 1)
            unvisited_pickups_exist = ((self.visited_ninf_flag == 0) & is_pickup_node_mask.unsqueeze(1)).any(dim=2)

            current_cluster = self.cluster_list[self.BATCH_IDX, self.current_node]

            is_switching_moment = (current_cluster == 1) & (~unvisited_pickups_exist)

            self.in_cluster = ~is_switching_moment

        self.step_state.in_cluster = self.in_cluster

        # Pass the new state field to the model during pre_step.
        self.step_state.first_node_in_cluster = self.first_node_in_cluster

        reward = None
        done = False
        return self.step_state, reward, done

    # CVRPEnv.py

    def step(self, selected):
        # Record the previous node so cluster transitions can be detected after the update.
        prev_node = self.current_node

        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        # Core logic for updating the first node of the current cluster.

        # ======================= Key fix =======================
        # Clone self.first_node_in_cluster before any in-place update.
        # This preserves computation-graph dependencies from earlier steps.
        self.first_node_in_cluster = self.first_node_in_cluster.clone()
        # ==========================================================

        if self.selected_count > 1:  # Only start checking after at least two moves.
            # Get the cluster IDs of the previous and current nodes.
            prev_cluster = self.cluster_list[self.BATCH_IDX, prev_node]
            current_cluster = self.cluster_list[self.BATCH_IDX, self.current_node]

            # Detect routes that just switched clusters.
            # Conditions: the cluster changed and the new cluster is not the depot (0).
            is_switched_mask = (prev_cluster != current_cluster) & (current_cluster != 0)

            # For switched routes, the current node becomes the entry node of the new cluster.
            self.first_node_in_cluster[is_switched_mask] = self.current_node[is_switched_mask]

        elif self.selected_count == 1:  # First move after leaving the depot.
            # The first visited node is naturally the entry node of the first cluster.
            # This assignment is not in-place, but the clone above keeps the logic consistent and safe.
            self.first_node_in_cluster = self.current_node


        # --- The remaining logic stays unchanged. ---
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')

        selected_clusters = self.cluster_list[self.BATCH_IDX, selected]
        is_pickup_visit = (selected_clusters == 1)

        if is_pickup_visit.any():
            delivery_partners = self.pairing[self.BATCH_IDX, selected]

            update_mask = torch.zeros_like(self.delivery_available_mask, dtype=torch.bool)
            update_mask[self.BATCH_IDX[is_pickup_visit], self.POMO_IDX[is_pickup_visit], delivery_partners[is_pickup_visit]] = True
            self.delivery_available_mask[update_mask] = 0

        self.ninf_mask = self.visited_ninf_flag + self.delivery_available_mask

        newly_finished = (self.visited_ninf_flag[..., 1:] == float('-inf')).all(dim=2)
        self.finished = self.finished | newly_finished

        self.ninf_mask[:, :, 0][self.finished] = 0

        self.current_cluster = self.cluster_list[self.BATCH_IDX, self.current_node]

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_cluster = self.current_cluster
        self.step_state.load = self.load

        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq)**2).sum(3).sqrt()
        travel_distances = segment_lengths.sum(2)
        return travel_distances
