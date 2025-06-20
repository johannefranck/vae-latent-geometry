import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle

from src.advanced_vae import AdvancedVAE
from src.catmull_init_spline import CatmullRom, construct_basis


def load_decoder(path, input_dim=50, latent_dim=2, device="cpu"):
    model = AdvancedVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.decoder


def compute_jacobian(decoder, z):
    z = z.detach().clone().requires_grad_(True)
    x = decoder(z).mean.view(-1)
    J_rows = []
    for i in range(x.shape[0]):
        grad_i = torch.autograd.grad(x[i], z, retain_graph=True, create_graph=True)[0]
        J_rows.append(grad_i)
    return torch.cat(J_rows, dim=0)  # (output_dim, 2)


class OmegaSpline(nn.Module):
    def __init__(self, z0, z1, basis, omega_init):
        super().__init__()
        self.z0 = z0.detach().clone()
        self.z1 = z1.detach().clone()
        self.basis = basis  # (4n, 4n-2)
        self.omega = nn.Parameter(omega_init.clone())  # (4n-2, 2)

    def forward(self, t):
        line = (1 - t)[:, None] * self.z0 + t[:, None] * self.z1  # (N, 2)
        coefs = self.basis @ self.omega                          # (4n, 2)
        n_poly = coefs.shape[0] // 4
        t_scaled = (t * n_poly - 1e-6).clamp(0, n_poly - 1)
        k = t_scaled.floor().long()
        u = t_scaled - k
        c = coefs.view(n_poly, 4, 2)[k]  # (N, 4, 2)
        u_vec = torch.stack([u**i for i in range(4)], dim=1)  # (N, 4)
        return line + torch.einsum("nu,nud->nd", u_vec, c)


def fit_omega_to_catmull(catmull_spline, basis, t_fit):
    with torch.no_grad():
        Z = catmull_spline(t_fit)  # (4n, 2)
        z0, z1 = catmull_spline.p[0], catmull_spline.p[-1]
        line = (1 - t_fit)[:, None] * z0 + t_fit[:, None] * z1
        dev = Z - line
        omega = torch.linalg.lstsq(basis, dev).solution
    return omega


def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)  # (N, 2)
    dz = (z[1:] - z[:-1]) * t_vals.shape[0]
    dz = dz.unsqueeze(1)
    G_all = []
    for zi in z[:-1]:
        J = compute_jacobian(decoder, zi.unsqueeze(0))  # (output_dim, 2)
        G = J.T @ J  # (2, 2)
        G_all.append(G.unsqueeze(0))
    G_all = torch.cat(G_all, dim=0)  # (N-1, 2, 2)
    energy = torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2))  # (N-1, 1, 1)
    return energy.mean()


def optimize_omega(spline, decoder, t_vals, steps=1000, lr=1e-2):
    optimizer = optim.Adam([spline.omega], lr=lr)
    for step in range(steps):
        optimizer.zero_grad()
        energy = compute_energy(spline, decoder, t_vals)
        energy.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}: Energy = {energy.item():.4f}")
    return spline


def plot_spline(spline, out_path="src/plots/spline_omega_optimized.png"):
    t = torch.linspace(0, 1, 400, device=spline.z0.device)
    with torch.no_grad():
        curve = spline(t).cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.plot(curve[:, 0], curve[:, 1], 'r-', label="Optimized ω-Spline", lw=2)
    plt.scatter([spline.z0[0].item(), spline.z1[0].item()],
                [spline.z0[1].item(), spline.z1[1].item()],
                c='b', s=20, label='Endpoints')
    plt.axis("equal")
    plt.tight_layout()
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print("Saved plot to", out_path)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spline_data = torch.load("src/artifacts/spline_init_syrota.pt", map_location=device)
    knots = spline_data["knots"].to(device)

    catmull = CatmullRom(knots).to(device)
    t_vals = torch.linspace(0, 1, 64, device=device)
    t_basis = torch.linspace(0, 1, 4 * (knots.shape[0] - 1), device=device)
    decoder = load_decoder("src/artifacts/vae_best_avae.pth", device=device)

    basis = construct_basis(knots.shape[0] - 1).to(device)  # (4n, 4n-2)
    omega_init = fit_omega_to_catmull(catmull, basis, t_basis)

    spline = OmegaSpline(knots[0], knots[-1], basis, omega_init).to(device)
    spline = optimize_omega(spline, decoder, t_vals, steps=1000, lr=1e-2)

    plot_spline(spline)
