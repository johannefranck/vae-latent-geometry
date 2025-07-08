import os
import json
import numpy as np
import torch
import argparse
import re
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

from src.select_representative_pairs import load_pairs
from src.single_decoder.optimize_energy import GeodesicSpline, construct_nullspace_basis

def set_seed(seed=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def create_latent_grid_from_data(latents, n_points_per_axis=150, margin=0.1):
    if not isinstance(latents, torch.Tensor):
        latents = torch.tensor(latents)
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
    return grid.view(-1, 2), (n_points_per_axis, n_points_per_axis)

def build_grid_graph(grid, k=8):
    grid_np = grid.cpu().numpy() if isinstance(grid, torch.Tensor) else grid
    tree = KDTree(grid_np)
    n_nodes = len(grid_np)
    graph = lil_matrix((n_nodes, n_nodes))

    for i in range(n_nodes):
        dists, indices = tree.query(grid_np[i], k=k + 1)
        graph.rows[i] = list(indices[1:])
        graph.data[i] = list(dists[1:])
    return graph.tocsr(), tree

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

def extract_seed_from_path(path):
    match = re.search(r"seed(\d+)", str(path))
    return match.group(1) if match else "unknown"

def main(seed, pairfile):
    set_seed(12)
    pairs_path = f"src/artifacts/{pairfile}"
    pair_name = Path(pairfile).stem.replace("selected_pairs_", "")
    os.makedirs("src/plots", exist_ok=True)

    n_poly = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    latents = np.load(f"src/artifacts/latents_VAE_ld2_ep100_bs64_lr1e-03_seed{seed}.npy")
    representatives, pairs = load_pairs(pairs_path)

    grid, _ = create_latent_grid_from_data(latents, n_points_per_axis=200)
    graph, tree = build_grid_graph(grid, k=8)
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)

    # plt.figure(figsize=(8, 8))
    # plt.scatter(latents[:, 0], latents[:, 1], c='lightgray', s=10, label='Latents')

    spline_data = []
    skipped_same = 0
    skipped_path = 0

    for i, (idx_a, idx_b) in enumerate(pairs):
        z_start = latents[idx_a]
        z_end = latents[idx_b]

        start_idx = tree.query(z_start)[1]
        end_idx = tree.query(z_end)[1]

        if start_idx == end_idx:
            skipped_same += 1
            continue

        _, predecessors = dijkstra(graph, directed=False, indices=[start_idx], return_predecessors=True)
        path_indices = reconstruct_path(predecessors[0], start_idx, end_idx)
        if not path_indices:
            skipped_path += 1
            continue

        # if skipped_path > 0 or skipped_same > 0:
        #     print(f"Skipped {skipped_same} due to identical grid points")
        #     print(f"Skipped {skipped_path} due to Dijkstra path failure")

        path_coords = grid[path_indices]
        if not isinstance(path_coords, torch.Tensor):
            target = torch.tensor(path_coords, dtype=torch.float32, device=device)
        else:
            target = path_coords.clone().detach().to(dtype=torch.float32, device=device)

        a, b = target[0], target[-1] # maybe should be z_start, z_end! TODO: check
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

        # t = torch.linspace(0, 1, 1000, device=device)
        # z = spline(t).detach().cpu().numpy()

        # plt.plot(path_coords[:, 0], path_coords[:, 1], 'k--', alpha=0.6, linewidth=1.5, label='Dijkstra' if i == 0 else None)
        # plt.plot(z[:, 0], z[:, 1], '-', alpha=1.0, linewidth=2, label='Fitted Spline' if i == 0 else None)

        spline_data.append({
            "a": a.detach().cpu(),
            "b": b.detach().cpu(),
            "a_index": idx_a,
            "b_index": idx_b,
            "a_label": next(rep["label"] for rep in representatives if rep["index"] == idx_a),
            "b_label": next(rep["label"] for rep in representatives if rep["index"] == idx_b),
            "n_poly": n_poly,
            "basis": basis.detach().cpu(),
            "omega_init": spline.omega.detach().cpu()
        })

    # plt.title("Latents with Dijkstra Paths and Fitted Splines")
    # plt.axis("equal")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"src/plots/splines_init_dijkstra_seed{seed}.png", dpi=300)
    # plt.close()

    print(f"Saving {len(spline_data)} fitted splines out of {len(pairs)} total pairs")
    if len(spline_data) == 0:
        print("[ERROR] No splines were fitted! Exiting without saving.")
        return

    output_file = f"src/artifacts/spline_batch_seed{seed}_p{pair_name}.pt"
    torch.save({
        "spline_data": spline_data,
        "representatives": representatives,
        "pairs": pairs
    }, output_file)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairfile", type=str, required=True, help="selected_pairs_*.json")
    args = parser.parse_args()
    main(args.seed, args.pairfile)
