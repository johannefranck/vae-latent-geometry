# Load latents (e.g. from artifacts/latents_*.npy)
# Build latent grid or k-NN graph
# Use Dijkstra to find shortest paths between point pairs
# Optionally visualize paths
# Save the paths for downstream spline initialization

import os
import numpy as np
import torch
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
from matplotlib import cm
from itertools import combinations



def create_latent_grid(n_points_per_axis=50, z_range=3.0):
    """Creates a 2D grid in latent space for discretization."""
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-z_range, z_range, n_points_per_axis),
        torch.linspace(-z_range, z_range, n_points_per_axis),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)
    flat_grid = grid.view(-1, 2)
    return flat_grid, (n_points_per_axis, n_points_per_axis)



def create_latent_grid_from_data(latents, n_points_per_axis=50, margin=0.1):
    """
    Create a grid that spans the min/max of the given latent data,
    with a small margin on each side.
    """
    latents = torch.tensor(latents) if not isinstance(latents, torch.Tensor) else latents
    z_min = latents.min(dim=0).values
    z_max = latents.max(dim=0).values

    z_range = z_max - z_min
    z_min -= margin * z_range
    z_max += margin * z_range

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(z_min[0], z_max[0], n_points_per_axis),
        torch.linspace(z_min[1], z_max[1], n_points_per_axis),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)
    flat_grid = grid.view(-1, 2)
    return flat_grid, (n_points_per_axis, n_points_per_axis)



def build_grid_graph(grid, k=8):
    """Builds a k-NN graph on the flattened latent grid."""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()

    tree = KDTree(grid)
    n_nodes = len(grid)
    graph = lil_matrix((n_nodes, n_nodes))

    for i in range(n_nodes):
        distances, indices = tree.query(grid[i], k=k + 1)
        for j, dist in zip(indices[1:], distances[1:]):
            graph[i, j] = dist
            graph[j, i] = dist
    return graph


def compute_shortest_paths(graph, source_indices):
    """Uses Dijkstra to compute shortest paths from given source indices."""
    dist_matrix, predecessors = dijkstra(
        csgraph=graph,
        directed=False,
        indices=source_indices,
        return_predecessors=True
    )
    return dist_matrix, predecessors


def reconstruct_path(predecessors, source, target):
    """Backtracks through Dijkstra predecessor array to build path."""
    path = []
    i = target
    while i != source:
        if i == -9999:
            return []
        path.append(i)
        i = predecessors[i]
    path.append(source)
    path.reverse()
    return path



def plot_paths_on_latents(grid, paths, labels, filename):
    """Overlays Dijkstra paths on latent space grid colored by labels."""
    grid_np = grid.cpu().numpy() if isinstance(grid, torch.Tensor) else grid
    plt.figure(figsize=(6, 6))

    # Use seaborn for colored scatter by label (same size as grid assumed)
    sns.scatterplot(
        x=grid_np[:, 0], y=grid_np[:, 1],
        hue=labels[:len(grid_np)],  # trim if needed
        s=5, alpha=0.6,
        palette="tab20",
        legend=False
    )

    cmap = plt.colormaps["tab10"]
    for idx, path in enumerate(paths):
        coords = grid_np[path]
        plt.plot(coords[:, 0], coords[:, 1], linewidth=1.5, color=cmap(idx % 10))

    plt.title("Dijkstra Paths on Grid (colored by label)")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def plot_paths_on_grid(grid, paths, filename):
    """Plots paths directly on the grid."""
    grid_np = grid.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(grid_np[:, 0], grid_np[:, 1], s=1, alpha=0.3, color="gray")
    cmap = plt.colormaps["tab10"]

    for idx, path in enumerate(paths):
        coords = grid_np[path]
        plt.plot(coords[:, 0], coords[:, 1], linewidth=1.5, color=cmap(idx % 10))

    plt.title("Dijkstra Paths on Grid")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_path_with_true_endpoints(grid, path, z_start, z_end, filename):
    """Plot Dijkstra path on grid, plus true latent start/end points."""
    grid_np = grid.cpu().numpy() if isinstance(grid, torch.Tensor) else grid
    path_coords = grid_np[path]

    plt.figure(figsize=(6, 6))
    # Grid background
    plt.scatter(grid_np[:, 0], grid_np[:, 1], s=1, alpha=0.3, color="lightgray")

    # Dijkstra path
    plt.plot(path_coords[:, 0], path_coords[:, 1], linewidth=2, color='blue', label="Dijkstra path")

    # True endpoints
    plt.scatter(*z_start, color="green", s=40, label="Start (true latent)")
    plt.scatter(*z_end, color="red", s=40, label="End (true latent)")

    plt.title("Dijkstra Path with True Latent Endpoints")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_all_paths_with_true_endpoints(grid, all_paths, endpoint_pairs, filename):
    """
    Plot all Dijkstra paths and all true start/end latent points in one figure.
    
    Args:
        grid: 2D latent grid (Tensor or ndarray)
        all_paths: list of lists of grid indices from Dijkstra
        endpoint_pairs: list of (z_start, z_end) latent coordinates
        filename: path to save the plot
    """
    grid_np = grid.cpu().numpy() if isinstance(grid, torch.Tensor) else grid

    plt.figure(figsize=(6, 6))
    plt.scatter(grid_np[:, 0], grid_np[:, 1], s=1, alpha=0.3, color="lightgray", label="Grid")

    # Plot all Dijkstra paths
    cmap = plt.colormaps["tab10"]
    for idx, path in enumerate(all_paths):
        path_coords = grid_np[path]
        plt.plot(path_coords[:, 0], path_coords[:, 1], linewidth=1.5, color=cmap(idx % 10))

    # Plot true latent start/end points
    all_sources = np.array([z[0] for z in endpoint_pairs])
    all_targets = np.array([z[1] for z in endpoint_pairs])

    plt.scatter(all_sources[:, 0], all_sources[:, 1], color="green", s=50, label="Start points")
    plt.scatter(all_targets[:, 0], all_targets[:, 1], color="red", s=50, label="End points")

    plt.title("All Dijkstra Paths with True Latent Start/End Points")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_latents_grid_and_paths(latents, labels, grid, all_paths, endpoint_pairs, filename, show_endpoints=False):
    grid_np = grid.cpu().numpy() if isinstance(grid, torch.Tensor) else grid

    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        x=latents[:, 0], y=latents[:, 1],
        hue=labels, palette="tab20", s=4, alpha=0.5, legend=False
    )
    plt.scatter(grid_np[:, 0], grid_np[:, 1], s=5, alpha=0.2, color="lightblue")

    cmap = plt.colormaps["tab10"]
    for idx, path in enumerate(all_paths):
        path_coords = grid_np[path]
        # print("Sample path indices:", path[:5])
        plt.plot(path_coords[:, 0], path_coords[:, 1], linewidth=1.5, color=cmap(idx % 10))

    all_sources = np.array([z[0] for z in endpoint_pairs])
    all_targets = np.array([z[1] for z in endpoint_pairs])

    if show_endpoints:
        plt.scatter(all_sources[:, 0], all_sources[:, 1], color="green", s=40, label="Start")
        plt.scatter(all_targets[:, 0], all_targets[:, 1], color="red", s=40, label="End")

    plt.title(f"Latents, Grid, and {len(all_paths)} Dijkstra Paths")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def select_label_representatives(latents, labels, n_labels=5, seed=12):
    """
    Selects one representative latent per label (closest to mean),
    then returns all pairwise combinations between them.

    Args:
        latents (np.ndarray): shape (N, D), latent representations
        labels (np.ndarray): shape (N,), class labels
        n_labels (int): number of distinct labels to use
        seed (int): random seed for reproducibility

    Returns:
        latent_pairs (List[Tuple[int, int]]): index pairs (i, j)
    """
    np.random.seed(seed)
    unique_labels = np.unique(labels)

    if n_labels > len(unique_labels):
        raise ValueError(f"Requested {n_labels} labels but only {len(unique_labels)} unique labels available.")

    selected_labels = np.random.choice(unique_labels, size=n_labels, replace=False)
    representatives = []

    for lbl in selected_labels:
        cluster_indices = np.where(labels == lbl)[0]
        cluster_latents = latents[cluster_indices]
        cluster_center = cluster_latents.mean(axis=0)
        closest_idx = cluster_indices[np.argmin(np.linalg.norm(cluster_latents - cluster_center, axis=1))]
        representatives.append(closest_idx)

    return list(combinations(representatives, 2))


def main():
    random.seed(12)
    np.random.seed(12)

    os.makedirs("src/plots", exist_ok=True)
    os.makedirs("src/artifacts", exist_ok=True)

    latents = np.load("src/artifacts/latents_ld2_ep30_bs64_lr1e-03.npy")
    labels = np.load("data/tasic-ttypes.npy")


    # The grid is synthetic; labels are only used for visualization
    grid, grid_shape = create_latent_grid_from_data(latents, n_points_per_axis=50)
    np.save("src/artifacts/grid.npy", grid.cpu().numpy())
    graph = build_grid_graph(grid, k=8)

    index_pairs = select_label_representatives(latents, labels, n_labels=10, seed=12)

    source_latents = latents[[i for i, _ in index_pairs]]  # shape: (5, 2)
    target_latents = latents[[j for _, j in index_pairs]]  # shape: (5, 2)

    tree = KDTree(grid)
    all_paths = []
    endpoint_pairs = []

    for i, (z_start, z_end) in enumerate(zip(source_latents, target_latents)):
        start_idx = tree.query(z_start)[1]
        end_idx = tree.query(z_end)[1]

        if start_idx == end_idx:
            print(f"Skipping pair {i}: identical grid points.")
            continue

        dist_matrix, predecessors = compute_shortest_paths(graph, [start_idx])
        path = reconstruct_path(predecessors[0], start_idx, end_idx)

        # print(f"Pair {i}: z_start={z_start}, z_end={z_end}")
        # print(f"Path length (grid steps): {len(path)}")
        # print(f"Grid indices (first 5): {path[:5]}")

        if path:
            all_paths.append(path)
            endpoint_pairs.append((z_start, z_end))


    plot_latents_grid_and_paths(
        latents,
        labels,
        grid,
        all_paths,
        endpoint_pairs,
        "src/plots/latents_grid_and_paths.pdf",
        show_endpoints=False
    )

    print(f"Saved visualizations for {len(all_paths)} shortest paths.")
    # Print shape of each path in latent space
    for i, path in enumerate(all_paths):
        path_coords = grid[path]
        print(f"Path {i} shape: {path_coords.shape}")

    # summary
    lengths = [len(p) for p in all_paths]
    print(f"Min path length: {min(lengths)}, Max path length: {max(lengths)}, Avg: {np.mean(lengths):.2f}")
        
    with open("src/artifacts/dijkstra_paths.pkl", "wb") as f:
        pickle.dump(all_paths, f)



if __name__ == "__main__":
    main()
