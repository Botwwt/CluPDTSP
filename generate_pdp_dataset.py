import os
import numpy as np
import torch
import pickle
import argparse

# ============================================================================
# Function Helpers
# ============================================================================

def generate_clustered_pdp_data(dataset_size, pdp_size, cluster_centers, cluster_std):
    """
    Generate PDTSP data with a clear clustered structure.
    Pickup and delivery nodes are sampled from different Gaussian clusters.
    """
    assert pdp_size % 2 == 0, "pdp_size must be even"
    num_pairs = pdp_size // 2
    
    dataset = []
    for _ in range(dataset_size):
        # Keep the depot fixed at the center.
        depot_xy = [0.5, 0.5]
        
        # Sample pickup nodes from the first Gaussian cluster.
        pickup_center = cluster_centers[0]
        pickups = pickup_center + np.random.randn(num_pairs, 2) * cluster_std
        
        # Sample delivery nodes from the second Gaussian cluster.
        delivery_center = cluster_centers[1]
        deliveries = delivery_center + np.random.randn(num_pairs, 2) * cluster_std
        
        # Concatenate node coordinates and keep them inside [0, 1].
        # The order is all pickup nodes first, then all delivery nodes.
        node_coords = np.concatenate((pickups, deliveries), axis=0)
        node_coords = np.clip(node_coords, 0, 1)
        
        dataset.append((depot_xy, node_coords.tolist()))
        
    return dataset


def generate_uniform_pdp_data(dataset_size, pdp_size):
    """
    Generate uniformly distributed PDTSP data using the original method.
    """
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, pdp_size, 2)).tolist()
    ))

# Utility helpers from the existing codebase.
def save_dataset(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

# ============================================================================
# Main Execution Logic
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Output filename (not recommended when generating multiple files)")
    parser.add_argument("--data_dir", default='data', help="Output directory")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='pdp', choices=['pdp'])
    
    parser.add_argument('--data_distribution', type=str, default='uniform', choices=['uniform', 'clustered'],
                        help="Choose data distribution: 'uniform' or 'clustered'")

    # dataset_size now means the number of instances per file and is typically 1 for evaluation.
    parser.add_argument("--dataset_size", type=int, default=1, help="Number of instances per .pkl file (should be 1 for evaluation).")
    
    # Control how many individual files are generated.
    parser.add_argument('--num_files', type=int, default=100, help="Number of individual instance files to generate for the test set.")
    
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[10, 20, 50],
                        help="Sizes of problem instances")
    
    parser.add_argument('--cluster_std', type=float, default=0.1, help="Standard deviation for clustered data")

    parser.add_argument("-f", action='store_true', help="Overwrite existing files")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()
    
    # Do not set a global seed here; each file receives its own reproducible seed inside the loop.

    for graph_size in opts.graph_sizes:
        
        datadir = os.path.join(opts.data_dir, opts.problem)
        os.makedirs(datadir, exist_ok=True)
        
        print(f"Generating {opts.num_files} instance files for graph size {graph_size} with '{opts.data_distribution}' distribution...")

        # Generate multiple independent files.
        for i in range(opts.num_files):
            
            # Assign a unique reproducible seed to each file.
            current_seed = opts.seed + i
            np.random.seed(current_seed)

            # Choose the generator according to the requested distribution.
            if opts.data_distribution == 'clustered':
                cluster_centers = [np.array([0.25, 0.25]), np.array([0.75, 0.75])]
                # Generate dataset_size instances, which is usually 1 for evaluation.
                dataset = generate_clustered_pdp_data(opts.dataset_size, graph_size, cluster_centers, opts.cluster_std)
                # Include the current seed in the filename to keep each file unique.
                filename = os.path.join(datadir, f"{opts.problem}{graph_size}_{opts.name}_clustered_std{opts.cluster_std}_seed{current_seed}.pkl")
                
            elif opts.data_distribution == 'uniform':
                dataset = generate_uniform_pdp_data(opts.dataset_size, graph_size)
                filename = os.path.join(datadir, f"{opts.problem}{graph_size}_{opts.name}_uniform_seed{current_seed}.pkl")
            
            else:
                raise ValueError(f"Unknown data distribution: {opts.data_distribution}")

            if opts.filename:
                # If filename is specified explicitly, generate one file and exit the loop.
                filename = check_extension(opts.filename)
                save_dataset(dataset, filename)
                print(f"Single dataset saved to {filename}")
                break  # Exit the loop.

            assert opts.f or not os.path.isfile(check_extension(filename)), \
                f"File already exists: {filename}. Use -f to overwrite."

            save_dataset(dataset, filename)
        
        if not opts.filename:
            print(f"Successfully generated {opts.num_files} files in {datadir}")
