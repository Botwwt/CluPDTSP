import torch
import numpy as np
import os
import re
import pickle

os.environ["OMP_NUM_THREADS"] = "1"

def load_pdtsp_instance(filepath, device="cpu"):
    coords = {}
    pickups = {}
    deliveries = {}
    depot_id = -1

    with open(filepath, 'r') as f:
        section = None
        for line in f:
            line = line.strip()
            if not line or line == "EOF" or line == "-1":
                continue

            if "NODE_COORD_SECTION" in line:
                section = "coords"
                continue
            elif "PICKUP_AND_DELIVERY_SECTION" in line:
                section = "pd"
                continue
            elif "DEPOT_SECTION" in line:
                section = "depot"
                continue

            if section == "coords":
                parts = re.split(r'\s+', line)
                node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                coords[node_id] = (x, y)
            
            elif section == "pd":
                parts = re.split(r'\s+', line)
                node_id, delivery_partner, pickup_partner = int(parts[0]), int(parts[5]), int(parts[6])
                if delivery_partner != 0:
                    pickups[node_id] = delivery_partner
                if pickup_partner != 0:
                    deliveries[node_id] = pickup_partner

            elif section == "depot":
                depot_id = int(line)

    if depot_id == -1:
        raise ValueError("Depot not found in file")

    depot_coord_raw = coords.pop(depot_id)
    customer_node_ids = sorted(coords.keys())
    id_to_idx = {node_id: i for i, node_id in enumerate(customer_node_ids)}
    
    customer_coords_raw = torch.tensor([coords[nid] for nid in customer_node_ids], dtype=torch.float32)
    depot_coord_raw = torch.tensor([depot_coord_raw], dtype=torch.float32)

    all_coords_raw = torch.cat([depot_coord_raw, customer_coords_raw], dim=0)
    min_vals, _ = torch.min(all_coords_raw, dim=0)
    max_vals, _ = torch.max(all_coords_raw, dim=0)
    range_vals = max_vals - min_vals
    scale = range_vals.max()
    if scale == 0: scale = 1.0

    depot_xy_normalized = (depot_coord_raw - min_vals) / scale
    node_xy_normalized = (customer_coords_raw - min_vals) / scale

    depot_xy = depot_xy_normalized.unsqueeze(0)
    node_xy = node_xy_normalized.unsqueeze(0)
    
    problem_size = len(customer_node_ids)
    
    cluster_list_np = np.zeros(problem_size, dtype=np.int64)
    for node_id in customer_node_ids:
        idx = id_to_idx[node_id]
        if node_id in pickups:
            cluster_list_np[idx] = 1
        elif node_id in deliveries:
            cluster_list_np[idx] = 2
    
    cluster_list = torch.from_numpy(cluster_list_np).long().unsqueeze(0)

    pairing = torch.zeros(1, problem_size + 1, dtype=torch.long)
    for pickup_id, delivery_id in pickups.items():
        pickup_env_idx = id_to_idx[pickup_id] + 1
        delivery_env_idx = id_to_idx[delivery_id] + 1
        pairing[0, pickup_env_idx] = delivery_env_idx
        pairing[0, delivery_env_idx] = pickup_env_idx

    node_demand = torch.zeros(1, problem_size)

    depot_xy = depot_xy.to(device)
    node_xy = node_xy.to(device)
    node_demand = node_demand.to(device)
    cluster_list = cluster_list.to(device)
    pairing = pairing.to(device)

    return depot_xy, node_xy, node_demand, cluster_list, pairing, scale.item(), id_to_idx, depot_id


def generate_batched_pdtsp_instances(batch_size, problem_size, device="cpu", distribution='uniform', cluster_std=0.1):
    """
    Generate a batch of PDTSP instances with either uniform or clustered sampling.
    """
    if problem_size % 2 != 0:
        raise ValueError(f"Problem size must be even for PDTSP, but got {problem_size}")
    
    num_pairs = problem_size // 2

    if distribution == 'uniform':
        # Original method: sample points uniformly inside the [0, 1] square.
        depot_xy = torch.rand(size=(batch_size, 1, 2), device=device, dtype=torch.float32)
        node_xy = torch.rand(size=(batch_size, problem_size, 2), device=device, dtype=torch.float32)

    elif distribution == 'clustered':
        # Clustered method: generate data with two Gaussian clusters.
        
        # Keep the depot near the center with a small random perturbation.
        depot_xy = torch.full((batch_size, 1, 2), 0.5, device=device, dtype=torch.float32)
        depot_xy += torch.randn(batch_size, 1, 2, device=device) * 0.01

        # Define the centers of the pickup and delivery clusters.
        pickup_center = torch.tensor([0.25, 0.25], device=device, dtype=torch.float32)
        delivery_center = torch.tensor([0.75, 0.75], device=device, dtype=torch.float32)

        # Sample pickup nodes from the first Gaussian cluster.
        pickup_coords = pickup_center.unsqueeze(0).expand(batch_size, num_pairs, -1) + \
                        torch.randn(batch_size, num_pairs, 2, device=device) * cluster_std
        
        # Sample delivery nodes from the second Gaussian cluster.
        delivery_coords = delivery_center.unsqueeze(0).expand(batch_size, num_pairs, -1) + \
                          torch.randn(batch_size, num_pairs, 2, device=device) * cluster_std
        
        # Concatenate coordinates with all pickup nodes first, then all delivery nodes.
        node_xy = torch.cat((pickup_coords, delivery_coords), dim=1)

        # Clip all coordinates into [0, 1] to avoid boundary overflow.
        depot_xy.clamp_(0, 1)
        node_xy.clamp_(0, 1)

    else:
        raise ValueError(f"Unknown data distribution: {distribution}")

    # --- The downstream logic for cluster_list, pairing, and node_demand is unchanged. ---
    
    pickup_clusters = torch.full((batch_size, num_pairs), 1, dtype=torch.long, device=device)
    delivery_clusters = torch.full((batch_size, num_pairs), 2, dtype=torch.long, device=device)
    cluster_list = torch.cat([pickup_clusters, delivery_clusters], dim=1)
    
    pairing = torch.zeros(batch_size, problem_size + 1, dtype=torch.long, device=device)
    
    pickup_indices = torch.arange(1, num_pairs + 1, device=device)
    delivery_indices = torch.arange(num_pairs + 1, problem_size + 1, device=device)
    
    pairing[:, pickup_indices] = delivery_indices
    pairing[:, delivery_indices] = pickup_indices
    
    node_demand = torch.zeros(size=(batch_size, problem_size), device=device, dtype=torch.float32)
    
    all_coords = torch.cat((depot_xy, node_xy), dim=1)
    all_clusters = torch.cat((torch.zeros(batch_size, 1, dtype=torch.long, device=device), cluster_list), dim=1)
    cluster_means = calculate_cluster_means(all_coords, all_clusters, 3)
    
    return depot_xy, node_xy, node_demand, cluster_list, cluster_means, pairing


def calculate_cluster_means(node_xy, cluster_list, cluster_size):
    batch_size = node_xy.size(0)
    cluster_sums = torch.zeros(batch_size, cluster_size, 2, device=node_xy.device)
    count_per_cluster = torch.zeros(batch_size, cluster_size, 1, device=node_xy.device)
    expanded_indices = cluster_list.unsqueeze(-1).expand(-1, -1, 2)
    cluster_sums.scatter_add_(1, expanded_indices, node_xy)
    count_per_cluster.scatter_add_(1, cluster_list.unsqueeze(-1), torch.ones_like(cluster_list.unsqueeze(-1), dtype=torch.float))
    count_per_cluster[count_per_cluster == 0] = 1
    cluster_means = cluster_sums / count_per_cluster
    return cluster_means

def augment_xy_data_by_8_fold(xy_data):
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)
    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    return aug_xy_data

def training_augment(xy_data):
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)
    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=2)
    return aug_xy_data

def load_pdp_pkl_instance(filepath, device="cpu"):
    with open(filepath, 'rb') as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and 'data' in payload:
        scale = payload.get('scale_factor', 1.0)
        dataset = payload['data']
    else:
        scale = 1.0
        dataset = payload

    instance = dataset[0]
    depot_coords, node_coords = instance[0], instance[1]

    depot_xy = torch.tensor([depot_coords], dtype=torch.float32, device=device).unsqueeze(0)
    node_xy = torch.tensor(node_coords, dtype=torch.float32, device=device).unsqueeze(0)

    problem_size = node_xy.size(1)
    if problem_size % 2 != 0:
        raise ValueError(f"Problem size from PKL must be even, but got {problem_size}")
    
    num_pairs = problem_size // 2

    pickup_clusters = torch.full((1, num_pairs), 1, dtype=torch.long, device=device)
    delivery_clusters = torch.full((1, num_pairs), 2, dtype=torch.long, device=device)
    cluster_list = torch.cat([pickup_clusters, delivery_clusters], dim=1)

    pairing = torch.zeros(1, problem_size + 1, dtype=torch.long, device=device)
    pickup_indices = torch.arange(1, num_pairs + 1, device=device)
    delivery_indices = torch.arange(num_pairs + 1, problem_size + 1, device=device)
    
    pairing[:, pickup_indices] = delivery_indices
    pairing[:, delivery_indices] = pickup_indices
    
    node_demand = torch.zeros(size=(1, problem_size), device=device, dtype=torch.float32)

    return depot_xy, node_xy, node_demand, cluster_list, pairing, scale
