import torch
import torch.nn as nn
import random
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix

from src.single_decoder.optimize_energy import construct_nullspace_basis



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


