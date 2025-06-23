import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from torch.optim import Adam
from src.vae import VAE
from src.spline_pytorch import TorchCubicSpline  # needed for pickle unpickling

# === Construct basis (Syrota-style) ===
def construct_basis(n_poly, dim):
    tc = torch.linspace(0, 1, n_poly + 1)[1:-1]
    boundary = torch.zeros((2, 4 * n_poly))
    boundary[0, 0] = 1
    boundary[1, -4:] = 1

    zeroth, first, second = torch.zeros((3, n_poly - 1, 4 * n_poly))
    for i in range(n_poly - 1):
        si = 4 * i
        t = tc[i]
        f0 = torch.tensor([1.0, t, t**2, t**3])
        f1 = torch.tensor([0.0, 1.0, 2.0*t, 3.0*t**2])
        f2 = torch.tensor([0.0, 0.0, 2.0, 6.0*t])
        zeroth[i, si:si+4] = f0
        zeroth[i, si+4:si+8] = -f0
        first[i, si:si+4] = f1
        first[i, si+4:si+8] = -f1
        second[i, si:si+4] = f2
        second[i, si+4:si+8] = -f2

    constraints = torch.cat([boundary, zeroth, first, second], dim=0)
    _, S, Vh = torch.linalg.svd(constraints)
    rank = (S > 1e-10).sum().item()
    return Vh.T[:, rank:]

# === Geodesic Spline Module ===
class GeodesicSpline(nn.Module):
    def __init__(self, point_pair, basis, omega_init):
        super().__init__()
        self.z0 = point_pair[0].requires_grad_(False)
        self.z1 = point_pair[1].requires_grad_(False)
        self.basis = basis
        self.n_poly = basis.shape[0] // 4
        self.dim = point_pair.shape[1]
        self.omega = nn.Parameter(omega_init.clone())  # optimize this!

    def _eval_line(self, t):
        return (1 - t).unsqueeze(1) * self.z0 + t.unsqueeze(1) * self.z1

    def _eval_poly(self, t):
        coefs = self.basis @ self.omega
        coefs = coefs.view(self.n_poly, 4, self.dim)

        # Avoid discontinuity artifacts
        eps = 1e-6
        t_scaled = (t * self.n_poly - eps).clamp(0, self.n_poly - 1)
        segment_idx = t_scaled.floor().long()
        local_t = t_scaled - segment_idx.float()

        # t^0 to t^3
        t_powers = torch.stack([local_t ** i for i in range(4)], dim=1)
        coefs_idx = coefs[segment_idx]  # select coefficients per sample

        return torch.einsum("nf,nfd->nd", t_powers, coefs_idx)


    def forward(self, t):
        return self._eval_line(t) + self._eval_poly(t)

# === Fit omega from control points ===
def fit_initial_omega(ctrl_pts, basis, point_pair, n_poly):
    t_vals = torch.linspace(0, 1, n_poly + 1)
    a, b = point_pair[0], point_pair[1]
    straight = (1 - t_vals).unsqueeze(1) * a + t_vals.unsqueeze(1) * b
    path_tensor = torch.tensor(ctrl_pts, dtype=torch.float32)
    deviation = path_tensor - straight
    D = deviation.view(-1, deviation.shape[-1])
    D_upsampled = F.interpolate(D.T.unsqueeze(0), size=4*n_poly, mode='linear', align_corners=True).squeeze(0).T
    return torch.linalg.lstsq(basis, D_upsampled).solution

# === Compute Riemannian Energy ===
def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)  # (T, latent_dim)
    dz = (z[1:] - z[:-1]) * t_vals.shape[0]  # (T-1, latent_dim)
    dz = dz.unsqueeze(1)  # (T-1, 1, latent_dim)

    G_all = []
    for zi in z[:-1]:
        zi = zi.unsqueeze(0).clone().detach().requires_grad_(True)  # (1, latent_dim)
        x = decoder(zi)  # (1, x_dim)
        J = torch.zeros(x.shape[-1], zi.shape[-1])  # (x_dim, latent_dim)

        for j in range(x.shape[-1]):
            grad = torch.autograd.grad(x[0, j], zi, retain_graph=True, create_graph=True)[0]
            J[j] = grad.squeeze(0)

        G = J.T @ J  # (latent_dim, latent_dim)
        G_all.append(G.unsqueeze(0))

    G_all = torch.cat(G_all, dim=0)  # (T-1, latent_dim, latent_dim)
    energy = torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2)).mean()
    return energy







# === Plotting ===
def plot_optimized_paths(latents, labels, grid, splines, filename):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.5, legend=False)
    plt.scatter(grid[:, 0], grid[:, 1], s=5, alpha=0.2, color="lightblue")
    t_vals = torch.linspace(0, 1, 300)
    for spline in splines:
        with torch.no_grad():
            pts = spline(t_vals).cpu().numpy()
        
        print(f"[PLOT] spline ω norm: {spline.omega.norm().item():.4f}")
        plt.plot(pts[:, 0], pts[:, 1], '-', color='red', alpha=0.8)
    plt.title("Optimized Geodesic Splines")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_comparison(ctrl_pts_list, optimized_splines, t_vals, filename):
    plt.figure(figsize=(8, 8))
    for ctrl_pts, spline in zip(ctrl_pts_list, optimized_splines):
        ctrl_pts = np.array(ctrl_pts)
        with torch.no_grad():
            optimized_path = spline(t_vals).cpu().numpy()
        plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'r--', label='initial')
        plt.plot(optimized_path[:, 0], optimized_path[:, 1], 'g-', label='optimized')
        print(f"[PLOT] ω norm: {spline.omega.norm().item():.4f}")

    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("Initial vs Optimized Geodesic Splines")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_spline_derivatives(spline, t_vals, name="spline_debug"):
    with torch.no_grad():
        z = spline(t_vals)
        dz = (z[1:] - z[:-1]) / (t_vals[1] - t_vals[0])
        ddz = (dz[1:] - dz[:-1]) / (t_vals[1] - t_vals[0])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(dz.cpu().numpy())
    plt.title("1st Derivative (dz/dt)")

    plt.subplot(1, 2, 2)
    plt.plot(ddz.cpu().numpy())
    plt.title("2nd Derivative (d²z/dt²)")

    plt.suptitle(f"Smoothness Check: {name}")
    plt.tight_layout()
    plt.savefig(f"src/plots/{name}_derivatives.png")
    plt.close()


# === Optimization ===
def optimize_splines(ctrl_pts_list, decoder, t_vals, steps=30, lr=1e-2):
    optimized = []
    n_poly = len(ctrl_pts_list[0]) - 1
    basis = construct_basis(n_poly, dim=2)

    for i, ctrl_pts in enumerate(ctrl_pts_list[:2]):
        z0 = torch.tensor(ctrl_pts[0], dtype=torch.float32)
        z1 = torch.tensor(ctrl_pts[-1], dtype=torch.float32)
        point_pair = torch.stack([z0, z1], dim=0)
        omega_init = fit_initial_omega(ctrl_pts, basis, point_pair, n_poly)
        spline = GeodesicSpline(point_pair, basis, omega_init)
        optimizer = Adam([spline.omega], lr=lr)

        print(f"\n--- Optimizing spline {i+1}/{len(ctrl_pts_list[:2])} ---")
        for step in range(steps):
            optimizer.zero_grad()
            energy = compute_energy(spline, decoder, t_vals)
            energy.backward()
            optimizer.step()
            print(f"Step {step+1:3d} | Energy: {energy.item():.6f}")

        optimized.append(spline)
        print(f"Spline {i+1} optimized. Final energy: {energy.item():.4f}")
        # Plot the derivatives for inspection
        plot_spline_derivatives(spline, t_vals, name=f"spline_{i+1}")
    return optimized

class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        Induces curvature in the latent space: non-linear map z -> x.
        """
        return torch.cat([
            torch.sin(2 * z[:, :1]),
            torch.exp(z[:, 1:])
        ], dim=1)

def generate_smooth_test_paths(n_paths=2, n_points=9):
    """
    Generate known smooth curves in latent space for testing.
    Paths are sinusoids or spirals to ensure C2 continuity.
    """
    t = np.linspace(0, 1, n_points)
    paths = []

    # Example 1: Half sine wave from left to right
    x1 = np.linspace(-2, 2, n_points)
    y1 = np.sin(np.pi * x1 / 4)
    paths.append(np.stack([x1, y1], axis=1))

    # Example 2: Log spiral
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = np.exp(0.1 * theta)
    x2 = r * np.cos(theta)
    y2 = r * np.sin(theta)
    paths.append(np.stack([x2, y2], axis=1))

    return paths[:n_paths]



T_VALS = torch.linspace(0, 1, 300)  # global discretization

# === Main ===
def main():
    import sys
    import src.spline_pytorch
    sys.modules['__main__'].TorchCubicSpline = src.spline_pytorch.TorchCubicSpline

    # decoder = VAE(input_dim=50, latent_dim=2).decoder
    # decoder.load_state_dict(torch.load("src/artifacts/decoder_ld2_ep600_bs64_lr1e-03.pth", map_location='cpu'))

    decoder = DummyDecoder()
    decoder.eval()

    z = torch.randn(100, 2)
    x = decoder(z)
    print("Decoder output std:", x.std().item())

    # with open("src/artifacts/spline_inits_torch.pkl", "rb") as f:
    #     cubic_splines = pickle.load(f)
    # ctrl_pts_list = [spline.ctrl_pts.detach().cpu().numpy() for spline in cubic_splines[:2]]
    ctrl_pts_list = generate_smooth_test_paths(n_paths=2, n_points=9)


    optimized_splines = optimize_splines(ctrl_pts_list, decoder, T_VALS, steps=20, lr=1e-2)

    with open("src/artifacts/spline_optimized_torch.pkl", "wb") as f:
        pickle.dump(optimized_splines, f)

    latents = np.load("src/artifacts/latents_ld2_ep600_bs64_lr1e-03.npy")
    labels = np.load("data/tasic-ttypes.npy")
    grid = np.load("src/artifacts/grid.npy")

    # Plot just optimized
    plot_optimized_paths(latents, labels, grid, optimized_splines, "src/plots/latents_optimized_splines.png")

    # Plot comparison
    plot_comparison(ctrl_pts_list, optimized_splines, T_VALS, "src/plots/all_splines_comparison.png")

if __name__ == "__main__":
    main()
