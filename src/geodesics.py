import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import random
from pathlib import Path
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix

from src.vae import VAE
from src.single_decoder.optimize_energy import construct_nullspace_basis


def set_seed(seed=12):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


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
    Construct a graph over the latent grid with edge weights based on
    both Euclidean distance and decoder ensemble disagreement (entropy proxy).
    """
    device = next(decoders[0].parameters()).device
    grid = grid.to(device)

    with torch.no_grad():
        # Get decoder outputs
        outputs = [decoder(grid).mean for decoder in decoders]  # list of (N, D)
        outputs = torch.stack(outputs)  # shape: (num_decoders, N, D)
        std = outputs.std(dim=0)  # (N, D)
        std = torch.clamp(std, min=eps)
        entropy_proxy = std.mean(dim=1)  # (N,)
        entropy_proxy = (entropy_proxy - entropy_proxy.min()) / (entropy_proxy.max() + eps)  # [0, 1]

    grid_np = grid.cpu().numpy()
    tree = KDTree(grid_np)
    n_nodes = len(grid_np)
    graph = lil_matrix((n_nodes, n_nodes))

    for i in range(n_nodes):
        dists, indices = tree.query(grid_np[i], k=k + 1)
        for j, dist in zip(indices[1:], dists[1:]):
            avg_entropy = 0.5 * (entropy_proxy[i].item() + entropy_proxy[j].item())
            weight = dist * (1.0 + avg_entropy)
            graph[i, j] = weight
            graph[j, i] = weight

    return graph.tocsr(), tree


# ====== ENERGY COMPUTATION ======
def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)
    x = decoder(z.view(-1, z.shape[-1])).mean
    x = x.view(t_vals.shape[0], z.shape[1], -1)
    diffs = x[1:] - x[:-1]
    return (diffs**2).sum(dim=2).sum(dim=0)


def compute_energy_ensemble(spline, decoder, t_vals, M=2):
    B = spline.a.shape[0]
    T = len(t_vals)
    D = spline.a.shape[1]
    device = t_vals.device

    z = spline(t_vals)  # (T, B, D)
    z_flat = z.view(T * B, D)

    total_energy = torch.zeros(B, device=device)
    rng = random.Random(456)  # fixed seed
    for _ in range(M):
        # DO NOT DISABLE GRADIENT HERE!
        d1, d2 = rng.choice(decoder), rng.choice(decoder)
        X1 = d1(z_flat).mean.view(T, B, -1)  # decoder returns distribution
        X2 = d2(z_flat).mean.view(T, B, -1)
        diffs = X2[1:] - X1[:-1]
        total_energy += (diffs ** 2).sum(dim=2).sum(dim=0)

    return total_energy / M

# def compute_energy_ensemble_batched(spline, decoders, t_vals, M=2):
#     """
#     Batched energy computation for B splines, using M ensemble samples per spline.
#     Each spline gets its own decoder pair samples.
#     """
#     B = spline.a.shape[0]
#     T = len(t_vals)
#     D = spline.a.shape[1]
#     device = t_vals.device

#     z = spline(t_vals)  # (T, B, D)
#     z_flat = z.view(T * B, D)

#     total_energy = torch.zeros(B, device=device)
#     rng = random.Random(456)

#     for _ in range(M):
#         d1_list = [rng.choice(decoders) for _ in range(B)]
#         d2_list = [rng.choice(decoders) for _ in range(B)]

#         X1 = torch.stack([d1(z_flat[i::B]).mean for i, d1 in enumerate(d1_list)], dim=1)  # (T, B, X)
#         X2 = torch.stack([d2(z_flat[i::B]).mean for i, d2 in enumerate(d2_list)], dim=1)  # (T, B, X)

#         diffs = X2[1:] - X1[:-1]
#         total_energy += (diffs ** 2).sum(dim=2).sum(dim=0)

#     return total_energy / M

def standardize(x, eps=1e-5):
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + eps)


def compute_energy_ensemble_batched(spline, decoders, t_vals, M=2):
    """
    Optimized batched energy computation with decoder output normalization.
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
    rng = torch.Generator(device=device).manual_seed(456)

    for _ in range(M):
        d1_idx = torch.randint(N_dec, (B,), generator=rng, device=device)
        d2_idx = torch.randint(N_dec, (B,), generator=rng, device=device)

        t_idx = torch.arange(T, device=device).view(-1, 1)  # (T, 1)
        b_idx = torch.arange(B, device=device).view(1, -1)  # (1, B)

        X1 = decoded_all[d1_idx, t_idx, b_idx]  # (T, B, X)
        X2 = decoded_all[d2_idx, t_idx, b_idx]

        diffs = X2[1:] - X1[:-1]  # (T-1, B, X)
        total_energy += (diffs ** 2).sum(dim=2).sum(dim=0)

    return total_energy / M



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
    B = spline.a.shape[0]
    T = len(t_vals)
    D = spline.a.shape[1]
    device = t_vals.device
    N_dec = len(decoders)
    dt = 1.0 / (T - 1)  # Add dt for proper arc length

    z = spline(t_vals)  # (T, B, D)
    z_flat = z.view(T * B, D)

    lengths = torch.zeros(B, device=device)
    rng = torch.Generator(device=device).manual_seed(456)

    for _ in range(M):
        d1_idx = torch.randint(N_dec, (1,), generator=rng, device=device).item()
        decoder = decoders[d1_idx]
        x = decoder(z_flat).mean.view(T, B, -1)
        diffs = x[1:] - x[:-1]
        lengths += torch.norm(diffs, dim=2).sum(dim=0) * dt  # dt scaled

    return (lengths / M).cpu()



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
            energy = compute_energy_ensemble_batched(model, decoders, t_vals, M)
        else:
            energy = compute_energy(model, decoders, t_vals)

        loss = energy
        loss.sum().backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[Step {step}] Energy: {energy.mean():.4f}")

        # Early stopping condition if no improvement in last 50 steps


    if ensemble:
        final_decoder = decoders
        lengths = compute_geodesic_lengths_ensemble(model, final_decoder, t_vals, M=M)
    else:
        lengths = compute_geodesic_lengths(model, decoders, t_vals)

    return model, lengths, model.omega.detach()



def main():
    set_seed(12)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairfile", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--ensemble", action="store_true", help="Enable ensemble decoding")
    args = parser.parse_args()

    seed = args.seed
    pair_tag = Path(args.pairfile).stem.replace("selected_pairs_", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)

    spline_path = f"src/artifacts/spline_batch_seed{seed}_p{pair_tag}.pt"
    model_path = f"src/artifacts/vae_best_seed{seed}.pth"
    output_path = f"src/artifacts/optimized_geodesics_seed{seed}_p{pair_tag}.pt"

    vae = VAE(input_dim=50, latent_dim=2).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    decoders = list(vae.decoder) if args.ensemble else vae.decoder

    spline_data = torch.load(spline_path, map_location=device)["spline_data"]
    n_poly = spline_data[0]["n_poly"]
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)
    t_vals = torch.linspace(0, 1, 200, device=device)

    all_outputs = []
    for start in range(0, len(spline_data), args.batch_size):
        end = min(start + args.batch_size, len(spline_data))
        chunk = spline_data[start:end]
        print(f"Optimizing splines {start} to {end - 1}")

        a = torch.stack([d["a"] for d in chunk]).to(device)
        b = torch.stack([d["b"] for d in chunk]).to(device)
        omega = torch.stack([d["omega_init"] for d in chunk]).to(device)
        cluster_pairs = [(d["a_label"], d["b_label"]) for d in chunk]

        model, lengths, omega_optimized = optimize_energy(
            a, b, omega, basis, decoders, n_poly, t_vals,
            ensemble=args.ensemble, M=10
        )

        for i in range(len(chunk)):
            all_outputs.append({
                "a": a[i].cpu(),
                "b": b[i].cpu(),
                "cluster_pair": cluster_pairs[i],
                "n_poly": n_poly,
                "basis": basis.cpu(),
                "omega_init": omega[i].cpu(),
                "omega_optimized": omega_optimized[i].cpu(),
                "length_geodesic": lengths[i].item(),
                "length_euclidean": torch.norm(a[i] - b[i]).item(),
            })

        del a, b, omega, model
        torch.cuda.empty_cache()

    torch.save(all_outputs, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
