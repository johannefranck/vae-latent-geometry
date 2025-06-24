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
        self.omega = nn.Parameter(torch.randn(basis.shape[1], self.a.shape[0]))

    # def eval_piecewise_poly(self, t, coeffs):
    #     if t.ndim == 2 and t.shape[1] == 1:
    #         t = t.squeeze(1)  # (T,)
    #     T = t.shape[0]
    #     n_poly = coeffs.shape[0]

    #     seg_idx = torch.clamp((t * n_poly).floor().long(), max=n_poly - 1)  # (T,)
    #     local_t = t * n_poly - seg_idx.float()  # (T,)
    #     powers = torch.stack([local_t**i for i in range(4)], dim=1)  # (T, 4)

    #     segment_coefs = coeffs[seg_idx]  # (T, 4, d)
    #     return torch.einsum("ti,tid->td", powers, segment_coefs)
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
    C = C.to(torch.float64)  # important!
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
        t = tc[i] # global t at knot i
        t=1

        # C0 continuity
        c0 = torch.zeros(4 * n_poly, device=device, dtype=torch.float64)
        c0[si:si+4]     = torch.tensor([1.0, t, t**2, t**3], dtype=torch.float64, device=device)   # left at t=1
        c0[si+4:si+8]   = -torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)       # right at t=0
        rows.append(c0)

        # C1 continuity
        c1 = torch.zeros(4 * n_poly, device=device, dtype=torch.float64)
        c1[si:si+4]     = torch.tensor([0.0, 1.0, 2.0*t, 3.0*t**2], dtype=torch.float64, device=device)        # d/dt left at t=1
        c1[si+4:si+8]   = -torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64, device=device)       # d/dt right at t=0
        rows.append(c1)

        # C2 continuity
        c2 = torch.zeros(4 * n_poly, device=device, dtype=torch.float64)
        c2[si:si+4]     = torch.tensor([0.0, 0.0, 2.0, 6.0*t], dtype=torch.float64, device=device)        # d²/dt² left at t=1
        c2[si+4:si+8]   = -torch.tensor([0.0, 0.0, 2.0, 0.0], dtype=torch.float64, device=device)       # d²/dt² right at t=0
        rows.append(c2)

    C = torch.stack(rows)

    basis = nullspace(C)
    basis = torch.linalg.qr(basis)[0]
    
    print("||C @ basis|| =", torch.norm(C @ basis.double()).item())
    print(f"New residual: {torch.norm(C @ basis):.2e}")
    print(f"rank of C: {torch.linalg.matrix_rank(C)}")
    print(f"expected rank: {C.shape[0]}")
    return basis.to(dtype=torch.float32), C.to(dtype=torch.float32)


# def construct_nullspace_basis(n_poly, device):
#     t_knots = torch.linspace(0, 1, n_poly + 1, device=device, dtype=torch.float64)[1:-1]  # (n-1,) internal knots

#     rows = []

#     # Boundary conditions: gamma(0) = 0, gamma(1) = 0
#     B = torch.zeros((2, 4 * n_poly), device=device, dtype=torch.float64)
#     B[0, 0] = 1.0
#     B[1, -4:] = 1.0
#     rows.append(B[0])
#     rows.append(B[1])

#     # Continuity constraints: C0, C1, C2 at internal knots
#     for i, t in enumerate(t_knots):
#         si = 4 * i

#         c0 = torch.zeros(4 * n_poly, device=device, dtype=torch.float64)
#         c0[si:si+4] = torch.tensor([1.0, t, t**2, t**3], device=device, dtype=torch.float64)
#         c0[si+4:si+8] = -c0[si:si+4]
#         #print(f"c0 shape: {c0.shape}")
#         rows.append(c0)

#         c1 = torch.zeros(4 * n_poly, device=device, dtype=torch.float64)
#         c1[si:si+4] = torch.tensor([0.0, 1.0, 2*t, 3*t**2], device=device, dtype=torch.float64)
#         c1[si+4:si+8] = -c1[si:si+4]
#         rows.append(c1)

#         c2 = torch.zeros(4 * n_poly, device=device, dtype=torch.float64)
#         c2[si:si+4] = torch.tensor([0.0, 0.0, 2.0, 6*t], device=device, dtype=torch.float64)
#         c2[si+4:si+8] = -c2[si:si+4]
#         rows.append(c2)

#     C = torch.stack(rows)  # shape: (3(n-1) + 2, 4n) = (11, 16) for n=4

#     #_, S, Vh = torch.linalg.svd(C)
#     #basis = Vh.T[:, C.shape[0]:].contiguous()
#     basis = nullspace(C)  # will now be high precision
#     basis = torch.linalg.qr(basis)[0]  # makes it orthonormal
#     residual = torch.norm(C @ basis)
#     print(f"New residual: {residual:.2e}")
#     print(f"rank of C: {torch.linalg.matrix_rank(C)}")
#     print(f"expected rank: {C.shape[0]}")

#     #print(f"norm: {torch.norm(C @ basis).item()}")
#     #print("Constructed constraint matrix C with shape:", C.shape)
#     #print("SVD singular values:", S)
#     # Check nullspace property
#     #print("Nullspace check (C @ basis):", torch.norm(C @ basis).item())
#     #print("Basis shape:", basis.shape)
#     basis = basis.to(dtype=torch.float32)
#     C = C.to(dtype=torch.float32)
#     return basis, C  # shape: (4n, d_free)




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
        energy.backward()

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
    # spline = TorchGeodesic(decoder, n_poly, torch.stack(pair).to(device), device=device)
    spline = optimize_spline(spline, decoder, C, steps=500, lr=1e-3, patience=500)

    # 
    t = torch.linspace(0, 1, 2000, device=device, requires_grad=True)
    t = t[:-1]  # remove last point to avoid duplicate at t=1
    z = spline(t)

    # Derivatives
    dz = torch.zeros_like(z)
    for i in range(z.shape[1]):
        dz[:, i] = torch.autograd.grad(z[:, i], t, grad_outputs=torch.ones_like(t), create_graph=True)[0]

    ddz = torch.zeros_like(z)
    for i in range(dz.shape[1]):
        ddz[:, i] = torch.autograd.grad(dz[:, i], t, grad_outputs=torch.ones_like(t), create_graph=True)[0]

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


    plt.tight_layout()
    plt.savefig("src/plots/spline_loaded_2.png", dpi=300)
    print("Saved plot to src/plots/spline_loaded_2.png")
