import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.vae import VAE

import random
import os

def set_seed(seed=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Needed for deterministic CUDA ops

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(12)  


class GeodesicSpline(nn.Module):
    def __init__(self, point_pair, basis, n_poly):
        super().__init__()
        self.a, self.b = point_pair
        self.n_poly = n_poly
        self.basis = basis
        gen = torch.Generator(device=self.a.device).manual_seed(12)
        self.omega = nn.Parameter(0.01* torch.randn(basis.shape[1], self.a.shape[0], generator=gen, device=self.a.device))

    def eval_piecewise_poly(self, t, coeffs):
        t = t.flatten()
        seg_idx = torch.clamp((t * self.n_poly).floor().long(), max=self.n_poly - 1)
        local_t = t * self.n_poly - seg_idx.float()
        powers = torch.stack([local_t**i for i in range(4)], dim=1)  # (T, 4)
        seg_coefs = coeffs[seg_idx]  # (T, 4, dim)
        return torch.einsum("ti,tid->td", powers, seg_coefs)

    def forward(self, t):
        coeffs = self.basis @ self.omega  # (4n, dim)
        coeffs = coeffs.view(self.n_poly, 4, -1)  # (n_poly, 4, dim)

        poly = self.eval_piecewise_poly(t, coeffs)
        linear = (1 - t[:, None]) * self.a + t[:, None] * self.b
        return linear + poly


def nullspace(C, rtol=1e-10):
    C = C.to(torch.float64)  
    U, S, Vh = torch.linalg.svd(C, full_matrices=True)
    rank = (S > rtol * S[0]).sum()
    return Vh.T[:, rank:].contiguous()


def construct_nullspace_basis(n_poly, device):
    rows = []

    # Boundary: spline offset(0) = 0 and offset(1) = 0
    B = torch.zeros((2, 4 * n_poly), device=device, dtype=torch.float64)
    B[0, 0] = 1.0   # first segment at t=0
    B[1, -4:] = 1.0 # last segment at t=1
    rows.append(B[0])
    rows.append(B[1])
    tc = torch.linspace(0, 1, n_poly + 1, device=device, dtype=torch.float64)[1:-1] # time cutoffs between polynomials

    # C0, C1, C2 continuity at internal knots
    for i in range(n_poly - 1):
        si = 4 * i  # start index
        # Local coordinate continuity: tL=1.0 (end of left), tR=0.0 (start of right)
        tL, tR = 1.0, 0.0

        # C0: continuity of position
        c0 = torch.zeros(4 * n_poly, dtype=torch.float64, device=device)
        c0[si:si+4] = torch.tensor([1, tL, tL**2, tL**3], device=device)
        c0[si+4:si+8] = -torch.tensor([1, tR, tR**2, tR**3], device=device)
        rows.append(c0)

        # C1: continuity of first derivative
        c1 = torch.zeros(4 * n_poly, dtype=torch.float64, device=device)
        c1[si:si+4] = torch.tensor([0, 1, 2*tL, 3*tL**2], device=device)
        c1[si+4:si+8] = -torch.tensor([0, 1, 2*tR, 3*tR**2], device=device)
        rows.append(c1)

        # C2: continuity of second derivative
        c2 = torch.zeros(4 * n_poly, dtype=torch.float64, device=device)
        c2[si:si+4] = torch.tensor([0, 0, 2, 6*tL], device=device)
        c2[si+4:si+8] = -torch.tensor([0, 0, 2, 6*tR], device=device)
        rows.append(c2)

    C = torch.stack(rows)

    basis = nullspace(C)
    basis = torch.linalg.qr(basis)[0]
    
    print("||C @ basis|| =", torch.norm(C @ basis.double()).item())
    print(f"New residual: {torch.norm(C @ basis):.2e}")
    print(f"rank of C: {torch.linalg.matrix_rank(C)}")
    print(f"expected rank: {C.shape[0]}")
    return basis.to(dtype=torch.float32), C.to(dtype=torch.float32)



def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)  # (T, latent_dim)
    x = decoder(z).mean  # (T, data_dim)
    x_flat = x.view(x.size(0), -1)  # Flatten to (T, obs_dim)

    diffs = x_flat[1:] - x_flat[:-1]
    dist_sq = diffs.pow(2).sum(dim=1)
    energy = dist_sq.sum()
    return energy



# ------------------ Optimization ------------------
def optimize_spline(spline, decoder, C, steps=1000, lr=1e-2, patience=500, delta=1e-6):
    # Automatically get the correct parameter tensor
    param = spline.omega if hasattr(spline, "omega") else spline.params
    optimizer = optim.Adam([param], lr=lr)

    t_vals = torch.linspace(0, 1, 2000, device=param.device)

    best_energy = compute_energy(spline, decoder, t_vals).item()
    best_params = param.data.clone()
    patience_counter = 0

    for step in range(steps):
        optimizer.zero_grad()
        energy = compute_energy(spline, decoder, t_vals)

        
        # Add penalty on deviation from b at t=1
        t_end = torch.tensor([1.0], device=param.device)
        end_error = (spline(t_end) - spline.b).pow(2).sum()
        full_loss = energy + 1000.0 * end_error  # weight penalty

        full_loss.backward()

        # if step >= 2500:
        #     torch.nn.utils.clip_grad_value_([spline.omega], clip_value=0.1)

        optimizer.step()

        new_energy = energy.item()
        rel_improvement = (best_energy - new_energy) / best_energy
        if rel_improvement > delta:
            best_energy = new_energy
            best_params = param.data.clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if step % 50 == 0:
            print(f"Step {step:4d}: Energy = {new_energy:.4f} | ω grad norm = {param.grad.norm():.4f}")

        if patience_counter > patience:
            print("Early stopping.")
            break

    param.data.copy_(best_params)
    print(f"omegas: {param.data.cpu().numpy()}")
    return spline


# ------------------ Main ------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder_path = "src/artifacts/vae_best.pth"
    vae = VAE(input_dim=50, latent_dim=2).to(device)
    vae.load_state_dict(torch.load(decoder_path, map_location=device))
    vae.eval()
    decoder = vae.decoder

    # some points
    a = torch.tensor([-0.5, -0.5], device=device)
    b = torch.tensor([0.5, 0.5], device=device)
    pair = (a, b)

    n_poly = 4
    basis, C = construct_nullspace_basis(n_poly, device)

    spline = GeodesicSpline(pair, basis, n_poly).to(device)
    omega_init = spline.omega.data.clone()
    spline = optimize_spline(spline, decoder, C, steps=2500, lr=1e-3, patience=500)
    spline_init = GeodesicSpline(pair, basis, n_poly).to(device)
    spline_init.omega.data.copy_(omega_init)


    # 
    t = torch.linspace(0, 1, 2000, device=device, requires_grad=True)
    t = t[:-1]  # remove last point to avoid duplicate at t=1
    z = spline(t)

    # Derivatives
    dz = torch.zeros_like(z)
    for i in range(z.shape[1]):
        dz[:, i] = torch.autograd.grad(z[:, i], t, grad_outputs=torch.ones_like(t), create_graph=True)[0] * n_poly

    ddz = torch.zeros_like(z)
    for i in range(dz.shape[1]):
        ddz[:, i] = torch.autograd.grad(dz[:, i], t, grad_outputs=torch.ones_like(t), create_graph=True)[0] * (n_poly**2)

    print("Spline shape:", z.shape)
    print("Velocity shape:", dz.shape)
    print("Acceleration shape:", ddz.shape)

    # Detach for plotting
    z_np = z.detach().cpu().numpy()
    dz_np = dz.detach().cpu().numpy()
    ddz_np = ddz.detach().cpu().numpy()

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(6, 12))

    axs[0].plot(z_np[:, 0], z_np[:, 1], 'r-', lw=2, label='Geodesic Spline')
    axs[0].scatter([a[0].item(), b[0].item()], [a[1].item(), b[1].item()], c='black', label='Endpoints')
    axs[0].set_title("Spline in Latent Space")
    axs[0].axis("equal")
    axs[0].legend()

    axs[1].plot(dz_np[:, 0], label="dx/dt")
    axs[1].plot(dz_np[:, 1], label="dy/dt")
    axs[1].set_title("Velocity (First Derivative)")
    axs[1].legend()

    axs[2].plot(ddz_np[:, 0], label="d²x/dt²")
    axs[2].plot(ddz_np[:, 1], label="d²y/dt²")
    axs[2].set_title("Acceleration (Second Derivative)")
    axs[2].legend()
    print("Spline start:", z_np[0])
    print("Spline end:  ", z_np[-1])
    print("Expected a:  ", a.cpu().numpy())
    print("Expected b:  ", b.cpu().numpy())

    # Compute and plot knot locations
    knot_times = torch.linspace(0, 1, n_poly + 1, device=device)[:-1]  # Exclude final t=1 to avoid duplication
    with torch.no_grad():
        z_knots = spline(knot_times).cpu().numpy()
        z_init = spline_init(t).cpu().numpy()

    axs[0].plot(z_init[:, 0], z_init[:, 1], 'k--', lw=1.5, label='Initial Spline')
    axs[0].scatter(z_knots[:, 0], z_knots[:, 1], c='blue', marker='x', s=60, label='Knots')
    for i, (x, y) in enumerate(z_knots):
        axs[0].text(x, y, f'k{i}', fontsize=8, ha='left', va='bottom', color='blue')


    plt.tight_layout()
    plt.savefig("src/plots/spline_smoothness.png", dpi=300)
    print("Saved plot to src/plots/spline_smoothness.png")
