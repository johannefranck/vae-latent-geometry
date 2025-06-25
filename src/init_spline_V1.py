import os
import numpy as np
import torch
import pickle
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

def build_knn_graph(points, k=8):
    tree = KDTree(points)
    n = len(points)
    graph = lil_matrix((n, n))
    for i in range(n):
        dists, inds = tree.query(points[i], k=k + 1)
        for j, dist in zip(inds[1:], dists[1:]):
            graph[i, j] = dist
            graph[j, i] = dist
    return graph, tree

def reconstruct_path(predecessors, start, end):
    path = []
    i = end
    while i != start:
        if i == -9999: return []
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

def plot_path_and_spline(path_coords, spline, filename):
    t = torch.linspace(0, 1, 1000, device=spline.omega.device)
    z = spline(t).detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(path_coords[:, 0], path_coords[:, 1], 'k--', label='Dijkstra Path', alpha=0.7)
    plt.plot(z[:, 0], z[:, 1], 'r-', label='Initial Spline')
    plt.scatter([path_coords[0, 0], path_coords[-1, 0]],
                [path_coords[0, 1], path_coords[-1, 1]], c='blue', label='Endpoints')
    plt.title("Initial Spline vs. Dijkstra Path")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def main():
    # --- Config ---
    latent_path = "src/artifacts/latents_VAE_ld2_ep100_bs64_lr1e-03.npy"
    label_path = "data/tasic-ttypes.npy"
    save_path = "src/artifacts/initial_splines.pt"
    os.makedirs("src/plots", exist_ok=True)
    os.makedirs("src/artifacts", exist_ok=True)

    n_poly = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load data ---
    latents = np.load(latent_path)
    labels = np.load(label_path)

    # --- Build kNN graph ---
    graph, tree = build_knn_graph(latents, k=8)
    print("k=8")

    # --- Select point pairs ---
    index_pairs = select_representative_pairs(latents, labels, max_labels=3)

    # --- Build basis ---
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)

    # # --- Dijkstra + Spline fit ---
    # spline_data = []
    # for i, (i_start, i_end) in enumerate(index_pairs):
    #     z0 = latents[i_start]
    #     z1 = latents[i_end]
    #     s_idx = tree.query(z0)[1]
    #     t_idx = tree.query(z1)[1]

    #     dist_matrix, predecessors = dijkstra(graph, directed=False, indices=[s_idx], return_predecessors=True)
    #     path_indices = reconstruct_path(predecessors[0], s_idx, t_idx)
    #     if not path_indices:
    #         print(f"Path {i} failed.")
    #         continue

    #     path_coords = tree.data[path_indices]
    #     a = torch.tensor(path_coords[0], dtype=torch.float32, device=device)
    #     b = torch.tensor(path_coords[-1], dtype=torch.float32, device=device)
    #     spline = GeodesicSpline((a, b), basis, n_poly=n_poly).to(device)

    #     # Fit spline ω to match path
    #     target = torch.tensor(path_coords, dtype=torch.float32, device=device)
    #     t_vals = torch.linspace(0, 1, len(target), device=device)

    #     optimizer = torch.optim.LBFGS([spline.omega], max_iter=50)

    #     def closure():
    #         optimizer.zero_grad()
    #         pred = spline(t_vals)
    #         loss = torch.nn.functional.mse_loss(pred, target)
    #         loss.backward()
    #         return loss

    #     optimizer.step(closure)


    #     # --- Plot ---
    #     plot_file = f"src/plots/spline_vs_path_{i:03d}.png"
    #     plot_path_and_spline(path_coords, spline, plot_file)
# --- Dijkstra + Spline fit ---
    for i, (i_start, i_end) in enumerate(index_pairs):
        s_idx = i_start
        t_idx = i_end

        if s_idx == t_idx:
            print(f"Path {i} skipped: identical points.")
            continue

        dist_matrix, predecessors = dijkstra(
            graph, directed=False, indices=[s_idx], return_predecessors=True
        )
        path_indices = reconstruct_path(predecessors[0], s_idx, t_idx)
        if not path_indices:
            print(f"Path {i} failed.")
            continue

        # Get the actual latent path (these are the graph nodes)
        path_coords = latents[path_indices]
        target = torch.tensor(path_coords, dtype=torch.float32, device=device)

        # Fit spline to follow this path
        a = target[0]
        b = target[-1]
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

        # --- Plot ---
        plot_file = f"src/plots/spline_vs_path_{i:03d}.png"
        plot_path_and_spline(path_coords, spline, plot_file)

    #     spline_data.append({
    #         "a": a.cpu(),
    #         "b": b.cpu(),
    #         "omega_init": omega_init.cpu(),
    #         "basis": basis.cpu(),
    #         "n_poly": n_poly
    #     })

    #     print(f"Plotted and prepared spline {i+1}/{len(index_pairs)}")

    # # --- Save after all plotting ---
    # torch.save(spline_data, save_path)
    # print(f"\nSaved {len(spline_data)} initial splines to {save_path}")

if __name__ == "__main__":
    main()
