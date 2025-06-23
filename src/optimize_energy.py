# optimize_omega_energy.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.vae import VAE

# -------------------------- Spline Definition ----------------------------

class SyrotaSpline(nn.Module):
    def __init__(self, point_a, point_b, basis):
        super().__init__()
        self.register_buffer("a", point_a)
        self.register_buffer("b", point_b)
        self.register_buffer("basis", basis)  # (4n, d_free)
        self.omega = nn.Parameter(torch.zeros(basis.shape[1], point_a.shape[0]))  # (d_free, dim)

    def forward(self, t):
        # evaluate straight line
        line = (1 - t)[:, None] * self.a + t[:, None] * self.b  # (N, dim)

        # evaluate polynomial deviation
        n_poly = self.basis.shape[0] // 4
        seg = torch.clamp((t * n_poly).floor().long(), max=n_poly - 1)
        local_t = t * n_poly - seg.float()
        tp = torch.stack([torch.ones_like(local_t), local_t, local_t**2, local_t**3], dim=1)  # (N, 4)

        coefs = self.basis @ self.omega  # (4n, dim)
        coefs = coefs.view(n_poly, 4, -1)
        segment_coefs = coefs[seg]  # (N, 4, dim)

        poly = torch.einsum("ni,nid->nd", tp, segment_coefs)  # (N, dim)
        return line + poly


# -------------------------- Energy Calculation --------------------------

def compute_jacobian(decoder, z):
    z = z.detach().clone().requires_grad_(True)
    x = decoder(z).mean.view(-1)
    J_rows = [torch.autograd.grad(x[i], z, retain_graph=True, create_graph=True)[0] for i in range(x.shape[0])]
    return torch.cat(J_rows, dim=0)  # (output_dim, latent_dim)

def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)  # (N, dim)
    dz = (z[1:] - z[:-1]) * t_vals.shape[0]
    dz = dz.unsqueeze(1)  # (N-1, 1, dim)

    G_all = []
    for zi in z[:-1]:
        J = compute_jacobian(decoder, zi.unsqueeze(0))  # (output_dim, dim)
        G = J.T @ J
        G_all.append(G.unsqueeze(0))
    G_all = torch.cat(G_all, dim=0)  # (N-1, dim, dim)

    energy = torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2))  # (N-1, 1, 1)
    return energy.mean()


# -------------------------- Optimization Loop ---------------------------

def optimize_omega(spline, decoder, steps=1000, lr=1e-2):
    optimizer = optim.Adam([spline.omega], lr=lr)
    t_vals = torch.linspace(0, 1, 64, device=spline.a.device)

    best_energy = compute_energy(spline, decoder, t_vals).item()
    print(f"Initial Energy: {best_energy:.4f}")

    for step in range(steps):
        omega_backup = spline.omega.data.clone()

        optimizer.zero_grad()
        energy = compute_energy(spline, decoder, t_vals)
        energy.backward()
        optimizer.step()

        new_energy = energy.item()
        if new_energy > best_energy:
            spline.omega.data.copy_(omega_backup)
        else:
            best_energy = new_energy

        if step % 50 == 0:
            print(f"Step {step:4d}: Energy = {new_energy:.2f} | ω grad norm = {spline.omega.grad.norm():.4f}")

    return spline


# -------------------------- Main Entry Point ----------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder_path = "src/artifacts/vae_best_avae.pth"

    vae = VAE(input_dim=50, latent_dim=2).to(device)
    vae.load_state_dict(torch.load(decoder_path, map_location=device))
    vae.eval()
    decoder = vae.decoder

    # endpoints in latent space (2D assumed)
    a = torch.tensor([-1.5, 0.0], device=device)
    b = torch.tensor([1.5, 0.0], device=device)

    # construct spline basis
    def construct_basis(n_poly):
        tc = torch.linspace(0, 1, n_poly + 1)[1:-1]
        rows = []
        e0, e1 = torch.zeros(4 * n_poly), torch.zeros(4 * n_poly)
        e0[0] = 1
        e1[-4:] = 1
        rows += [e0, e1]
        for i, t in enumerate(tc):
            s = 4 * i
            p = torch.tensor([1, t, t**2, t**3])
            d1 = torch.tensor([0, 1, 2*t, 3*t**2])
            d2 = torch.tensor([0, 0, 2, 6*t])
            for v in (p, d1, d2):
                r = torch.zeros(4 * n_poly)
                r[s:s+4] = v
                r[s+4:s+8] = -v
                rows.append(r)
        C = torch.stack(rows).float()
        _, _, Vh = torch.linalg.svd(C)
        return Vh.T[:, C.size(0):].contiguous()

    n_poly = 8
    basis = construct_basis(n_poly).to(device)
    spline = SyrotaSpline(a, b, basis).to(device)

    spline = optimize_omega(spline, decoder, steps=1000, lr=1e-2)

    # Plot
    with torch.no_grad():
        t = torch.linspace(0, 1, 400, device=device)
        z = spline(t).cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.plot(z[:, 0], z[:, 1], 'r-', lw=2, label='Optimized Spline')
        plt.scatter([a[0].cpu(), b[0].cpu()], [a[1].cpu(), b[1].cpu()], c='black', label='Endpoints')
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.savefig("src/plots/spline_syrota_omega.png", dpi=300)
        print("Saved plot: src/plots/spline_syrota_omega.png")
