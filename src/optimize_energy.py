import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.vae import VAE




class GeodesicSpline(nn.Module):
    def __init__(self, point_pair, basis, n_poly):
        super().__init__()
        self.a, self.b = point_pair
        self.n_poly = n_poly
        self.basis = basis
        self.omega = nn.Parameter(10 * torch.randn(basis.shape[1], self.a.shape[0]))

    def eval_piecewise_poly(self, t, coeffs):
        """
        Evaluate piecewise cubic spline.

        t: (T,) time values in [0,1]
        coeffs: (n_poly, 4, d) spline coefficients
        returns: (T, d)
        """
        T = t.shape[0]
        n_poly = coeffs.shape[0]
        d = coeffs.shape[2]

        seg_idx = torch.clamp((t * n_poly).floor().long(), max=n_poly - 1)  # (T,)
        local_t = t * n_poly - seg_idx.float()  # (T,)
        powers = torch.stack([local_t**i for i in range(4)], dim=1)  # (T, 4)

        segment_coefs = coeffs[seg_idx]  # (T, 4, d)
        return torch.einsum("ti,tid->td", powers, segment_coefs)

    def forward(self, t):
        coeffs = self.basis @ self.omega  # (4n, dim)
        #print("Coefs shape (before view):", coeffs.shape)

        coeffs = coeffs.view(self.n_poly, 4, -1)  # (n_poly, 4, dim)
        #print("Segment coefs shape (after view):", coeffs.shape)

        poly = self.eval_piecewise_poly(t, coeffs)
        #print(poly)
        linear = (1 - t[:, None]) * self.a + t[:, None] * self.b
        return linear + poly



def construct_nullspace_basis(n_poly, device):
    t_knots = torch.linspace(0, 1, n_poly + 1, device=device)[1:-1]  # (n-1,) internal knots

    rows = []

    # Boundary conditions: gamma(0) = 0, gamma(1) = 0
    B = torch.zeros((2, 4 * n_poly), device=device)
    B[0, 0] = 1.0
    B[1, -4:] = 1.0
    rows.append(B[0])
    rows.append(B[1])

    # Continuity constraints: C0, C1, C2 at internal knots
    for i, t in enumerate(t_knots):
        si = 4 * i

        c0 = torch.zeros(4 * n_poly, device=device)
        c0[si:si+4] = torch.tensor([1.0, t, t**2, t**3], device=device)
        c0[si+4:si+8] = -c0[si:si+4]
        #print(f"c0 shape: {c0.shape}")
        rows.append(c0)

        c1 = torch.zeros(4 * n_poly, device=device)
        c1[si:si+4] = torch.tensor([0.0, 1.0, 2*t, 3*t**2], device=device)
        c1[si+4:si+8] = -c1[si:si+4]
        rows.append(c1)

        c2 = torch.zeros(4 * n_poly, device=device)
        c2[si:si+4] = torch.tensor([0.0, 0.0, 2.0, 6*t], device=device)
        c2[si+4:si+8] = -c2[si:si+4]
        rows.append(c2)

    C = torch.stack(rows)  # shape: (3(n-1) + 2, 4n) = (11, 16) for n=4

    _, S, Vh = torch.linalg.svd(C)
    basis = Vh.T[:, C.shape[0]:].contiguous()
    #print("Constructed constraint matrix C with shape:", C.shape)
    #print("SVD singular values:", S)
    # Check nullspace property
    #print("Nullspace check (C @ basis):", torch.norm(C @ basis).item())
    #print("Basis shape:", basis.shape)
    return basis, C  # shape: (4n, d_free)


# ------------------ Jacobian & Energy ------------------

# def compute_jacobian(decoder, z):
#     z = z.detach().clone().requires_grad_(True)
#     x = decoder(z).mean.view(-1)
#     grads = [torch.autograd.grad(x[i], z, retain_graph=True, create_graph=True)[0] for i in range(x.shape[0])]
#     return torch.cat(grads, dim=0)  # (output_dim, dim)


# def compute_energy(spline, decoder, t_vals):
#     z = spline(t_vals)
#     dz = (z[1:] - z[:-1]) * t_vals.shape[0]
#     dz = dz.unsqueeze(1)  # (T-1, 1, dim)

#     G_all = []
#     for zi in z[:-1]:
#         J = compute_jacobian(decoder, zi.unsqueeze(0))  # (output_dim, dim)
#         identity = torch.eye(J.shape[1], device=J.device)
#         deviation = torch.norm(J.T @ J - identity)
#         #print("Deviation from Euclidean metric:", deviation.item())
#         if J.norm().item() < 1:
#             print("Jacobian norm:", J.norm().item())

#         G = J.T @ J  # (dim, dim)
#         G_all.append(G.unsqueeze(0))
#     G_all = torch.cat(G_all, dim=0)  # (T-1, dim, dim)

#     energy = torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2))  # (T-1, 1, 1)
#     return energy.mean()
def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)  # (T, latent_dim)
    x = decoder(z).mean  # (T, data_dim) — you might need to flatten if needed
    x_flat = x.view(x.size(0), -1)  # Flatten to (T, obs_dim)

    diffs = x_flat[1:] - x_flat[:-1]
    dist_sq = diffs.pow(2).sum(dim=1)
    energy = dist_sq.sum()
    return energy



# ------------------ Optimization ------------------

def optimize_spline(spline, decoder, C, steps=1000, lr=1e-2, patience=500, delta=1e-6):
    optimizer = optim.Adam([spline.omega], lr=lr)
    t_vals = torch.linspace(0, 1, 1000, device=spline.omega.device)

    best_energy = compute_energy(spline, decoder, t_vals).item()
    best_params = spline.omega.data.clone()
    patience_counter = 0

    for step in range(steps):
        optimizer.zero_grad()
        energy = compute_energy(spline, decoder, t_vals)
        energy.backward()

        #residual = torch.norm(spline.basis.new_tensor(C) @ (spline.basis @ spline.omega))
        #print(f"[Step {step:4d}] Constraint residual: {residual.item():.4e}")
        
        if step >= 2500:
            torch.nn.utils.clip_grad_value_([spline.omega], clip_value=1.0)

        optimizer.step()

        new_energy = energy.item()
        rel_improvement = (best_energy - new_energy) / best_energy
        if rel_improvement > delta:
            best_energy = new_energy
            best_params = spline.omega.data.clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if step % 50 == 0:
            print(f"Step {step:4d}: Energy = {new_energy:.4f} | ω grad norm = {spline.omega.grad.norm():.4f}")

        if patience_counter > patience:
            print("Early stopping.")
            break

    spline.omega.data.copy_(best_params)
    return spline


# ------------------ Main ------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder_path = "src/artifacts/vae_best_avae.pth"
    vae = VAE(input_dim=50, latent_dim=2).to(device)
    vae.load_state_dict(torch.load(decoder_path, map_location=device))
    vae.eval()
    decoder = vae.decoder

    # Define point pair (e.g. two samples in latent space)
    a = torch.tensor([-20, 25], device=device)
    b = torch.tensor([20, -55], device=device)
    pair = (a, b)

    n_poly = 4
    basis, C = construct_nullspace_basis(n_poly, device)
    spline = GeodesicSpline(pair, basis, n_poly).to(device)

    spline = optimize_spline(spline, decoder, C, steps=5000, lr=1e-3, patience=500)


with torch.no_grad():
    t = torch.linspace(0, 1, 1000, device=device)
    z = spline(t).cpu().numpy()

    # Compute first and second derivatives
    dz = np.gradient(z, axis=0) * 1000
    ddz = np.gradient(dz, axis=0) * 1000

    fig, axs = plt.subplots(3, 1, figsize=(6, 12))

    # Plot spline path
    axs[0].plot(z[:, 0], z[:, 1], 'r-', lw=2, label='Geodesic Spline')
    axs[0].scatter([a[0].cpu(), b[0].cpu()], [a[1].cpu(), b[1].cpu()], c='black', label='Endpoints')
    axs[0].set_title("Spline in Latent Space")
    axs[0].axis("equal")
    axs[0].legend()

    # Velocity
    axs[1].plot(dz[:, 0], label="dx/dt")
    axs[1].plot(dz[:, 1], label="dy/dt")
    axs[1].set_title("Velocity (First Derivative)")
    axs[1].legend()

    # Acceleration
    axs[2].plot(ddz[:, 0], label="d²x/dt²")
    axs[2].plot(ddz[:, 1], label="d²y/dt²")
    axs[2].set_title("Acceleration (Second Derivative)")
    axs[2].legend()

    plt.tight_layout()
    Path("src/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("src/plots/spline_diagnostics.png", dpi=300)
    print("Saved: src/plots/spline_diagnostics.png")
