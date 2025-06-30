import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
from pathlib import Path
from src.vae_good import VAE
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
        T = len(t)
        device = t.device

        coeffs = torch.einsum("nk,bkd->nbd", self.basis, self.omega)
        coeffs = coeffs.view(self.n_poly, 4, B, D)

        seg_idx = torch.clamp((t * self.n_poly).floor().long(), max=self.n_poly - 1)
        local_t = t * self.n_poly - seg_idx.float()
        powers = torch.stack([local_t ** i for i in range(4)], dim=1).to(device)

        coeffs_selected = coeffs[seg_idx]  # (T, 4, B, D)
        poly = torch.einsum("ti,tibd->tbd", powers, coeffs_selected)  # (T, B, D)

        linear = (1 - t[:, None, None]) * self.a[None, :, :] + t[:, None, None] * self.b[None, :, :]
        return linear + poly

@torch.no_grad()
def compute_geodesic_lengths(spline, decoder, t_vals):
    z = spline(t_vals)
    x = decoder(z.view(-1, z.shape[-1])).mean
    x = x.view(t_vals.shape[0], z.shape[1], -1)

    diffs = x[1:] - x[:-1]
    lengths = torch.norm(diffs, dim=2).sum(dim=0)
    return lengths.cpu()

def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)
    x = decoder(z.view(-1, z.shape[-1])).mean
    x = x.view(t_vals.shape[0], z.shape[1], -1)

    diffs = x[1:] - x[:-1]
    energy = (diffs ** 2).sum(dim=2).sum(dim=0)
    return energy

def main(seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    set_seed(seed)

    spline_path = f"src/artifacts/spline_batch_seed{seed}.pt"
    decoder_path = f"src/artifacts/vae_best_seed{seed}.pth"
    output_path = f"src/artifacts/spline_batch_optimized_batched_seed{seed}.pt"

    vae = VAE(input_dim=50, latent_dim=2).to(device)
    vae.load_state_dict(torch.load(decoder_path, map_location=device))
    vae.eval()
    decoder = vae.decoder

    spline_data = torch.load(spline_path, map_location=device)["spline_data"]
    n_poly = spline_data[0]["n_poly"]

    a = torch.stack([d["a"] for d in spline_data]).to(device)
    b = torch.stack([d["b"] for d in spline_data]).to(device)
    omega = torch.stack([d["omega_init"] for d in spline_data]).to(device)
    cluster_pairs = [(d["a_label"], d["b_label"]) for d in spline_data]

    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)
    model = GeodesicSplineBatch(a, b, basis, omega, n_poly).to(device)

    optimizer = optim.Adam([model.omega], lr=1e-3)
    t_vals = torch.linspace(0, 1, 2000, device=device)

    for step in range(500):
        optimizer.zero_grad()
        energy = compute_energy(model, decoder, t_vals)
        endpoint_error = (model(t_vals[-1:]) - b[None]) ** 2
        endpoint_loss = endpoint_error.sum(dim=(0, 2))
        loss = energy + 1000 * endpoint_loss
        loss.sum().backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}")

    geodesic_lengths = compute_geodesic_lengths(model, decoder, t_vals)
    output = []
    for i in range(len(spline_data)):
        output.append({
            "a": a[i].cpu(),
            "b": b[i].cpu(),
            "cluster_pair": cluster_pairs[i],
            "n_poly": n_poly,
            "basis": basis.cpu(),
            "omega_init": omega[i].cpu(),
            "omega_optimized": model.omega.data[i].cpu(),
            "length_geodesic": geodesic_lengths[i].item(),
            "length_euclidean": torch.norm(a[i] - b[i]).item()
        })

    torch.save(output, output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(args.seed)
