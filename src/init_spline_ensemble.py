import os
import re
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
from src.select_representative_pairs import load_pairs
from src.vae import VAE, EVAE
from src.single_decoder.optimize_energy import GeodesicSpline, construct_nullspace_basis

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def compute_entropy(outputs, eps=1e-8):
    std = outputs.std(dim=0)
    std = torch.clamp(std, min=eps)
    entropy_proxy = std.mean(dim=1)
    entropy_proxy -= entropy_proxy.min()
    entropy_proxy /= entropy_proxy.max() + eps
    return entropy_proxy

def build_entropy_weighted_graph(grid, decoders, k=6):
    with torch.no_grad():
        outputs = [decoder(grid).mean for decoder in decoders]
        outputs = torch.stack(outputs)
        entropy = compute_entropy(outputs)
    tree = KDTree(grid.cpu().numpy())
    n = grid.size(0)
    graph = lil_matrix((n, n))
    for i in range(n):
        dists, idxs = tree.query(grid[i].cpu().numpy(), k=k + 1)
        for j, dist in zip(idxs[1:], dists[1:]):
            weight = dist * (entropy[i].item() + entropy[j].item()) / 2
            graph[i, j] = graph[j, i] = weight
    return graph.tocsr(), tree

def create_latent_grid(latents, n=200, margin=0.1):
    z_min = latents.min(axis=0)
    z_max = latents.max(axis=0)
    z_range = z_max - z_min
    z_min -= margin * z_range
    z_max += margin * z_range
    x = np.linspace(z_min[0], z_max[0], n)
    y = np.linspace(z_min[1], z_max[1], n)
    mesh_x, mesh_y = np.meshgrid(x, y)
    grid = np.stack([mesh_x, mesh_y], axis=-1).reshape(-1, 2)
    return torch.tensor(grid, dtype=torch.float32), (n, n)

def main(seed, pairfile, num_decoders):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load data -----
    data = np.load("data/tasic-pca50.npy")  # (N, 50)
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # ----- Init model -----
    if num_decoders == 1:
        suffix = f"VAE_ld2_d1_ep100_bs64_lr1e-03_seed{seed}"
        model = VAE(input_dim=50, latent_dim=2).to(device)
        decoders = [model.decoder]
    else:
        suffix = f"EVAE_ld2_d{num_decoders}_ep100_bs64_lr1e-03_seed{seed}"
        model = EVAE(input_dim=50, latent_dim=2, num_decoders=num_decoders).to(device)
        decoders = model.decoders

    model.load_state_dict(torch.load(f"src/artifacts/{suffix}_best.pth", map_location=device))
    model.eval()

    # ----- Compute latents dynamically -----
    with torch.no_grad():
        latents = model.encoder(data_tensor).mean  # (N, 2)

    grid, _ = create_latent_grid(latents.cpu().numpy(), n=100)
    grid = grid.to(device)
    graph, tree = build_entropy_weighted_graph(grid, decoders, k=8)
    basis, _ = construct_nullspace_basis(n_poly=4, device=device)

    pairs_path = f"src/artifacts/{pairfile}"
    pair_tag = Path(pairfile).stem.replace("selected_pairs_", "")
    representatives, pairs = load_pairs(pairs_path)

    spline_data = []
    for idx_a, idx_b in tqdm(pairs, desc="Fitting splines"):
        z_a = latents[idx_a]
        z_b = latents[idx_b]

        start_idx = tree.query(z_a.cpu().numpy())[1]
        end_idx = tree.query(z_b.cpu().numpy())[1]
        if start_idx == end_idx:
            continue

        _, pred = dijkstra(graph, indices=[start_idx], return_predecessors=True)
        path_idx = []
        node = end_idx
        while node != start_idx:
            if node == -9999:
                path_idx = []
                break
            path_idx.append(node)
            node = pred[0][node]
        path_idx = [start_idx] + path_idx[::-1]
        if not path_idx:
            continue

        path_coords = grid[path_idx].to(device)
        a, b = z_a, z_b
        spline = GeodesicSpline((a, b), basis, n_poly=4).to(device)

        t_vals = torch.linspace(0, 1, len(path_coords), device=device)
        optimizer = torch.optim.LBFGS([spline.omega], max_iter=50)

        def closure():
            optimizer.zero_grad()
            pred = spline(t_vals)
            loss = torch.nn.functional.mse_loss(pred, path_coords)
            loss.backward()
            return loss

        optimizer.step(closure)

        spline_data.append({
            "a": z_a.cpu(), "b": z_b.cpu(),
            "a_index": idx_a, "b_index": idx_b,
            "a_label": next(r["label"] for r in representatives if r["index"] == idx_a),
            "b_label": next(r["label"] for r in representatives if r["index"] == idx_b),
            "n_poly": 4, "basis": basis.cpu(),
            "omega_init": spline.omega.detach().cpu()
        })

    if len(spline_data) == 0:
        print("No splines fitted. Exiting.")
        return

    out_path = f"src/artifacts/spline_ensemble_seed{seed}_p{pair_tag}_d{num_decoders}.pt"
    torch.save({"spline_data": spline_data, "pairs": pairs, "representatives": representatives}, out_path)
    print(f"[INFO] Saved: {out_path} | Total: {len(spline_data)} splines")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairfile", type=str, required=True)
    parser.add_argument("--num_decoders", type=int, required=True)
    args = parser.parse_args()
    main(args.seed, args.pairfile, args.num_decoders)
