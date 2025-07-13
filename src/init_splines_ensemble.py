import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from pathlib import Path
from scipy.sparse.csgraph import dijkstra
from src.select_representative_pairs import load_pairs
from src.single_decoder.optimize_energy import GeodesicSpline

from src.single_decoder.optimize_energy import construct_nullspace_basis
from src.single_decoder.optimize_energy_batched import GeodesicSplineBatch
from src.plotting import plot_latents_with_selected, plot_initialized_splines
from src.train import EVAE, GaussianEncoder, GaussianDecoder, GaussianPrior, make_encoder_net, make_decoder_net



# ====== INITIALIZE SPLINES AND GRID CREATION ======
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


def build_entropy_weighted_graph(grid, decoders, k=8, eps=1e-8):
    """
    Construct a graph over the latent grid with edge weights based purely on
    decoder ensemble disagreement (entropy proxy).
    """
    device = next(decoders[0].parameters()).device
    grid = grid.to(device)

    with torch.no_grad():
        # Decoder outputs: (num_decoders, N, D)
        outputs = torch.stack([decoder(grid).mean for decoder in decoders])  
        std = outputs.std(dim=0)  # (N, D)
        entropy_proxy = std.norm(dim=1)  # (N,), higher norm = more disagreement

    # Normalize entropy to [0, 1]
    entropy_proxy = (entropy_proxy - entropy_proxy.min()) / (entropy_proxy.max() - entropy_proxy.min() + eps)

    grid_np = grid.cpu().numpy()
    tree = KDTree(grid_np)
    n_nodes = len(grid_np)
    graph = lil_matrix((n_nodes, n_nodes))

    for i in range(n_nodes):
        dists, indices = tree.query(grid_np[i], k=k + 1)
        for j in indices[1:]:
            avg_entropy = 0.5 * (entropy_proxy[i].item() + entropy_proxy[j].item())
            graph[i, j] = avg_entropy
            graph[j, i] = avg_entropy

    return graph.tocsr(), tree



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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--pairfile", type=str, required=True)  # e.g. experiment/pairs/selected_pairs_10.json
    parser.add_argument("--use-entropy", action="store_true", help="Use decoder ensemble uncertainty for graph weights")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--plot-latents", action="store_true")
    parser.add_argument("--n-poly", type=int, default=4)
    args = parser.parse_args()

    # === Infer model ID and create save folder ===
    model_name = Path(args.model_path).stem  # e.g. "model_seed12"
    if args.save_dir is None:
        save_dir = Path("experiment") / f"splines_init_{model_name}"
    else:
        save_dir = Path(args.save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    latent_dim = 2
    input_dim = 50
    encoder = GaussianEncoder(make_encoder_net(input_dim, latent_dim))
    decoder = GaussianDecoder(make_decoder_net(latent_dim, input_dim))
    prior = GaussianPrior(latent_dim)
    model = EVAE(prior, encoder, decoder, num_decoders=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    assert len(model.decoder) == 10, "[ERROR] Expected 10 decoders in ensemble."

    # Load data
    data = np.load("data/tasic-pca50.npy").astype(np.float32)
    data_tensor = torch.from_numpy(data).to(device)
    with torch.no_grad():
        latents = model.encoder(data_tensor).base_dist.loc.cpu().numpy()

    # Optional: plot latent space with selected centers
    if args.plot_latents:
        plot_latents_with_selected(
            model,
            data_tensor,
            selected_indices_path=args.pairfile,
            save_path="experiment/latent_with_selected_model123.png"
        )

    # Prepare Dijkstra init
    representatives, pairs = load_pairs(args.pairfile)
    grid, _ = create_latent_grid_from_data(latents, n_points_per_axis=200)
    if args.use_entropy:
        print("[INFO] Building entropy-weighted graph...")
        graph, tree = build_entropy_weighted_graph(grid, model.decoder)
    else:
        print("[INFO] Building Euclidean graph...")
        graph, tree = build_grid_graph(grid, k=8)

    basis, _ = construct_nullspace_basis(n_poly=args.n_poly, device=device)

    spline_data = []
    for idx_a, idx_b in tqdm(pairs, desc="Initializing splines"):
        z_start = latents[idx_a]
        z_end = latents[idx_b]
        start_idx = tree.query(z_start)[1]
        end_idx = tree.query(z_end)[1]
        if start_idx == end_idx:
            continue

        _, preds = dijkstra(graph, indices=start_idx, return_predecessors=True)
        path = reconstruct_path(preds, start_idx, end_idx)
        if not path:
            continue

        target = grid[path].to(device)
        a, b = target[0], target[-1]
        # spline = GeodesicSpline((a, b), basis, n_poly=4).to(device)
        spline = GeodesicSplineBatch(
                            a.unsqueeze(0),  # (1, D)
                            b.unsqueeze(0),  # (1, D)
                            basis,
                            omega=torch.zeros((1, basis.shape[1], a.shape[0]), device=device),
                            n_poly=args.n_poly
                        ).to(device)

        t_vals = torch.linspace(0, 1, len(target), device=device)
        optimizer = torch.optim.LBFGS([spline.omega], max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(spline(t_vals).squeeze(1), target)
            loss.backward()
            return loss

        optimizer.step(closure)
        assert spline.omega.shape == (1, basis.shape[1], a.shape[0]), f"omega shape bad: {spline.omega.shape}"

        spline_data.append({
            "a": a.detach().cpu(),
            "b": b.detach().cpu(),
            "a_index": idx_a,
            "b_index": idx_b,
            "a_label": next(rep["label"] for rep in representatives if rep["index"] == idx_a),
            "b_label": next(rep["label"] for rep in representatives if rep["index"] == idx_b),
            "n_poly": args.n_poly,
            "basis": basis.detach().cpu(),
            "omega_init": spline.omega.squeeze(0).detach().cpu()
        })

    # Save spline batch
    pairname = Path(args.pairfile).stem.replace("selected_pairs_", "")
    graph_type = "entropy" if args.use_entropy else "euclidean"
    save_path = save_dir / f"spline_batch_init_{graph_type}_{pairname}.pt"

    torch.save({
        "spline_data": spline_data,
        "representatives": representatives,
        "pairs": pairs
    }, save_path)
    print(f"[✓] Saved {len(spline_data)} initialized splines to: {save_path}")

    # optional plotting of intiial splines
    plot_initialized_splines(
        latents=latents,
        spline_data=spline_data,
        basis=basis,
        representatives=representatives,
        save_path=save_dir / f"spline_plot_init_{graph_type}_{pairname}.png",
        device=device
    )
    print(f"[✓] Saved plot of initialized splines to: {save_dir}/spline_plot_init_{graph_type}_{pairname}.png")
