import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

from src.optimize_energy import GeodesicSpline, construct_nullspace_basis

def set_seed(seed=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
set_seed(12)

def create_latent_grid_from_data(latents, n_points_per_axis=150, margin=0.1):
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
    return graph, tree

def reconstruct_path(predecessors, start, end):
    path = []
    i = end
    while i != start:
        if i == -9999:
            return []
        path.append(i)
        i = predecessors[i]
    path.append(start)
    return path[::-1]

def select_representative_pairs(latents, labels, max_labels=133):
    unique_labels = np.unique(labels)
    selected_labels = unique_labels[:max_labels]
    representatives = []
    for lbl in selected_labels:
        inds = np.where(labels == lbl)[0]
        cluster = latents[inds]
        center = cluster.mean(axis=0)
        closest = inds[np.argmin(np.linalg.norm(cluster - center, axis=1))]
        representatives.append(closest)
    return list(combinations(representatives, 2))

def main():
    latent_path = "src/artifacts/latents_VAE_ld2_ep100_bs64_lr1e-03.npy"
    label_path = "data/tasic-ttypes.npy"
    os.makedirs("src/plots", exist_ok=True)

    n_poly = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latents = np.load(latent_path)
    labels = np.load(label_path)

    grid, _ = create_latent_grid_from_data(latents, n_points_per_axis=150)
    graph, tree = build_grid_graph(grid, k=8)

    index_pairs = select_representative_pairs(latents, labels, max_labels=5)
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)

    source_latents = latents[[i for i, _ in index_pairs]]
    target_latents = latents[[j for _, j in index_pairs]]

    plt.figure(figsize=(8, 8))
    plt.scatter(latents[:, 0], latents[:, 1], c='lightgray', s=10, label='Latents')

    for i, (z_start, z_end) in enumerate(zip(source_latents, target_latents)):
        start_idx = tree.query(z_start)[1]
        end_idx = tree.query(z_end)[1]

        if start_idx == end_idx:
            print(f"Skipping pair {i}: identical grid points.")
            continue

        dist_matrix, predecessors = dijkstra(graph, directed=False, indices=[start_idx], return_predecessors=True)
        path_indices = reconstruct_path(predecessors[0], start_idx, end_idx)
        if not path_indices:
            print(f"Path {i} failed.")
            continue

        path_coords = grid[path_indices]
        if not isinstance(path_coords, torch.Tensor):
            target = torch.tensor(path_coords, dtype=torch.float32, device=device)
        else:
            target = path_coords.clone().detach().to(dtype=torch.float32, device=device)

        a, b = target[0], target[-1]
        spline = GeodesicSpline((a, b), basis, n_poly=n_poly).to(device)

        t_vals = torch.linspace(0, 1, len(target), device=device)
        optimizer = torch.optim.LBFGS([spline.omega], max_iter=50)

        def closure():
            optimizer.zero_grad()
            pred = spline(t_vals)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Plot both the Dijkstra path and spline
        t = torch.linspace(0, 1, 1000, device=device)
        z = spline(t).detach().cpu().numpy()

        plt.plot(path_coords[:, 0], path_coords[:, 1], 'k--', alpha=0.6, linewidth=1.5, label='Dijkstra Path' if i == 0 else None)
        plt.plot(z[:, 0], z[:, 1], '-', alpha=1.0, linewidth=2, label='Fitted Spline' if i == 0 else None)

        # Store spline info for later use in optimize_energy.py
        if "spline_data" not in locals():
            spline_data = []

        spline_data.append({
            "a": a.detach().cpu(),
            "b": b.detach().cpu(),
            "n_poly": n_poly,
            "basis": basis.detach().cpu(),
            "omega_init": spline.omega.detach().cpu()
        })


    plt.title("Latents with Dijkstra Paths and Fitted Splines")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("src/plots/all_splines_combined.png", dpi=300)
    plt.close()

    # Save all splines to disk in a compatible format
    torch.save(spline_data, "src/artifacts/spline_batch.pt")
    print("Saved all fitted splines to src/artifacts/spline_batch.pt")

if __name__ == "__main__":
    main()
