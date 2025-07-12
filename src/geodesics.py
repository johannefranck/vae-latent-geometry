import torch
import torch.nn as nn
import random
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix

from src.single_decoder.optimize_energy import construct_nullspace_basis
from src.plotting import plot_latents_with_selected, plot_initialized_splines
from src.train import EVAE, GaussianEncoder, GaussianDecoder, GaussianPrior, make_encoder_net, make_decoder_net



class GeodesicSplineBatch(nn.Module):
    def __init__(self, a, b, basis, omega, n_poly):
        super().__init__()
        self.a = a  # (B, D)
        self.b = b  # (B, D)
        self.basis = basis  # (4n, K)
        self.omega = nn.Parameter(omega)  # (B, K, D)
        self.n_poly = n_poly

    def forward(self, t):
        B, K, D = self.omega.shape
        coeffs = torch.einsum("nk,bkd->nbd", self.basis, self.omega)
        coeffs = coeffs.view(self.n_poly, 4, B, D)
        seg_idx = torch.clamp((t * self.n_poly).floor().long(), max=self.n_poly - 1)
        local_t = t * self.n_poly - seg_idx.float()
        powers = torch.stack([local_t**i for i in range(4)], dim=1).to(t.device)
        coeffs_selected = coeffs[seg_idx]
        poly = torch.einsum("ti,tibd->tbd", powers, coeffs_selected)
        linear = (1 - t[:, None, None]) * self.a[None] + t[:, None, None] * self.b[None]
        return linear + poly



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


# ====== ENERGY COMPUTATION ======
def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)
    x = decoder(z.view(-1, z.shape[-1])).mean
    x = x.view(t_vals.shape[0], z.shape[1], -1)
    diffs = x[1:] - x[:-1]
    return (diffs**2).sum(dim=2).sum(dim=0)

def compute_energy_ensemble_batched(spline, decoders, t_vals, M=2):
    """
    Optimized batched energy computation with sampling decoder pairs from ensemble.
    """
    B = spline.a.shape[0]
    T = len(t_vals)
    D = spline.a.shape[1]
    device = t_vals.device
    N_dec = len(decoders)

    z = spline(t_vals)  # (T, B, D)
    z_flat = z.view(T * B, D)  # (TB, D)

    decoded_all = []
    for decoder in decoders:
        out = decoder(z_flat).mean  # (TB, X)
        decoded_all.append(out.view(T, B, -1))

    decoded_all = torch.stack(decoded_all)  # (N_dec, T, B, X)

    total_energy = torch.zeros(B, device=device)

    if N_dec == 1:
        return compute_energy(spline, decoders[0], t_vals)
    for _ in range(M):
        decoder_1 = random.choice(decoders)
        decoder_2 = random.choice(decoders)

        x1 = decoder_1(z_flat).mean.view(T, B, -1)
        x2 = decoder_2(z_flat).mean.view(T, B, -1)

        diffs = x2[1:] - x1[:-1]
        total_energy += (diffs ** 2).sum(dim=2).sum(dim=0)

    return total_energy / M


# Lengths

@torch.no_grad()
def compute_geodesic_lengths(spline, decoder, t_vals):
    T = len(t_vals)
    dt = 1.0 / (T - 1) # scaling
    z = spline(t_vals)
    x = decoder(z.view(-1, z.shape[-1])).mean
    x = x.view(t_vals.shape[0], z.shape[1], -1)
    diffs = x[1:] - x[:-1]
    return (torch.norm(diffs, dim=2).sum(dim=0) * dt).cpu() # scaled by dt

@torch.no_grad()
def compute_geodesic_lengths_ensemble(spline, decoders, t_vals, M=3):
    """
    Estimate geodesic lengths by integrating speed over the curve using
    samples from the decoder ensemble. Each sample uses one fixed decoder.
    """
    B = spline.a.shape[0]
    T = len(t_vals)
    D = spline.a.shape[1]
    device = t_vals.device
    dt = 1.0 / (T - 1)

    z = spline(t_vals)  # (T, B, D)
    z_flat = z.view(T * B, D)

    lengths = torch.zeros(B, device=device)

    for _ in range(M):
        decoder = random.choice(decoders)  # Sample one decoder
        x = decoder(z_flat).mean.view(T, B, -1)  # (T, B, X)
        diffs = x[1:] - x[:-1]  # (T-1, B, X)
        lengths += torch.norm(diffs, dim=2).sum(dim=0) * dt

    return (lengths / M).cpu()


# ====== OPTIMIZATION FUNCTION ======

def optimize_energy(
    a, b, omega_init, basis, decoders, n_poly, t_vals,
    steps=500, lr=1e-3,
    ensemble=False, M=2
):
    model = GeodesicSplineBatch(a, b, basis, omega_init, n_poly).to(a.device)
    optimizer = torch.optim.Adam([model.omega], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        if ensemble:
            energy = compute_energy_ensemble_batched(model, decoders, t_vals, M = M)
        else:
            energy = compute_energy(model, decoders, t_vals)

        loss = energy
        loss.sum().backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[Step {step}] Energy: {energy.mean():.4f}")

        # Early stopping??

    if ensemble:
        final_decoder = decoders
        lengths = compute_geodesic_lengths_ensemble(model, final_decoder, t_vals, M=M)
    else:
        lengths = compute_geodesic_lengths(model, decoders, t_vals)

    return model, lengths, model.omega.detach()


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from scipy.sparse.csgraph import dijkstra
    from src.select_representative_pairs import load_pairs
    from src.single_decoder.optimize_energy import GeodesicSpline

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--pairfile", type=str, required=True)  # e.g. experiment/pairs/selected_pairs_10.json
    parser.add_argument("--use-entropy", action="store_true", help="Use decoder ensemble uncertainty for graph weights")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--plot-latents", action="store_true")
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

    basis, _ = construct_nullspace_basis(n_poly=4, device=device)

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
        spline = GeodesicSpline((a, b), basis, n_poly=4).to(device)

        t_vals = torch.linspace(0, 1, len(target), device=device)
        optimizer = torch.optim.LBFGS([spline.omega], max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(spline(t_vals), target)
            loss.backward()
            return loss

        optimizer.step(closure)

        spline_data.append({
            "a": a.detach().cpu(),
            "b": b.detach().cpu(),
            "a_index": idx_a,
            "b_index": idx_b,
            "a_label": next(rep["label"] for rep in representatives if rep["index"] == idx_a),
            "b_label": next(rep["label"] for rep in representatives if rep["index"] == idx_b),
            "n_poly": 4,
            "basis": basis.detach().cpu(),
            "omega_init": spline.omega.detach().cpu()
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
    # plot_initialized_splines(
    #     latents=latents,
    #     spline_data=spline_data,
    #     basis=basis,
    #     representatives=representatives,
    #     save_path=save_dir / f"spline_plot_init_{graph_type}_{pairname}.png",
    #     device=device
    # )
    # print(f"[✓] Saved plot of initialized splines to: {save_dir}/spline_plot_init_{graph_type}_{pairname}.png")
