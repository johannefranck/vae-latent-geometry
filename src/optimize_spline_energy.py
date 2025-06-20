import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from torch.optim import Adam
from src.spline_pytorch import TorchCubicSpline
from src.advanced_vae import AdvancedVAE

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

class GeodesicSpline(nn.Module):
    def __init__(self, point_pair, basis, omega_init):
        super().__init__()
        self.z0 = point_pair[0].requires_grad_(False)
        self.z1 = point_pair[1].requires_grad_(False)
        self.basis = basis
        self.n_poly = basis.shape[0] // 4
        self.dim = point_pair.shape[1]
        self.omega = nn.Parameter(omega_init.clone())

    def _eval_line(self, t):
        return (1 - t).unsqueeze(1) * self.z0 + t.unsqueeze(1) * self.z1

    def _eval_poly(self, t):
        coefs = self.basis @ self.omega
        coefs = coefs.view(self.n_poly, 4, self.dim)
        eps = 1e-6
        t_scaled = (t * self.n_poly - eps).clamp(0, self.n_poly - 1)
        segment_idx = t_scaled.floor().long()
        local_t = t_scaled - segment_idx.float()
        t_powers = torch.stack([local_t ** i for i in range(4)], dim=1)
        return torch.einsum("nf,nfd->nd", t_powers, coefs[segment_idx])

    def forward(self, t):
        return self._eval_line(t) + self._eval_poly(t)

# def fit_initial_omega(ctrl_pts, basis, n_poly):
#     path_tensor = torch.tensor(ctrl_pts, dtype=torch.float32)
#     D_upsampled = F.interpolate(path_tensor.T.unsqueeze(0), size=4*n_poly, mode='linear', align_corners=True).squeeze(0).T
#     return torch.linalg.lstsq(basis, D_upsampled).solution
def fit_initial_omega(ctrl_pts, basis, n_poly, alpha=1e-4):
    path_tensor = torch.tensor(ctrl_pts, dtype=torch.float32)  # shape: [N, 2]
    path_tensor = path_tensor.T.unsqueeze(0)  # shape: [1, 2, N]

    # Interpolate to 4*n_poly points (must match basis size)
    D_upsampled = F.interpolate(path_tensor, size=4*n_poly, mode='linear', align_corners=True)
    D_upsampled = D_upsampled.squeeze(0).T  # shape: [4*n_poly, 2]

    # Regularized least squares solution
    A = basis
    AtA = A.T @ A + alpha * torch.eye(A.shape[1])
    Atb = A.T @ D_upsampled
    omega = torch.linalg.solve(AtA, Atb)
    return omega



def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)
    dz = (z[1:] - z[:-1]) * t_vals.shape[0]
    dz = dz.unsqueeze(1)
    G_all = []
    for zi in z[:-1]:
        zi = zi.unsqueeze(0).clone().detach().requires_grad_(True)
        x = decoder(zi).mean
        J = torch.stack([torch.autograd.grad(x[0, j], zi, retain_graph=True, create_graph=True)[0].squeeze(0)
                         for j in range(x.shape[-1])], dim=0)
        G = J.T @ J
        G_all.append(G.unsqueeze(0))
    print(f"Jacobian norm: {J.norm().item():.2e}")
    G_all = torch.cat(G_all, dim=0)
    return torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2)).mean()

def optimize_splines(ctrl_pts_list, decoder, t_vals, basis, n_poly, steps=30, lr=1e-2, init_mode = "fit"):
    optimized = []
    for i, ctrl_pts in enumerate(ctrl_pts_list):
        z0 = torch.tensor(ctrl_pts[0], dtype=torch.float32)
        z1 = torch.tensor(ctrl_pts[-1], dtype=torch.float32)
        point_pair = torch.stack([z0, z1])

        # possible init modes: 
        if init_mode == "fit":
                    omega_init = fit_initial_omega(ctrl_pts, basis, n_poly)
        elif init_mode == "zeros": # do "zeros" like syrota.
            omega_init = torch.zeros((basis.shape[1],2), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown initialization mode: {init_mode}")
        
        spline = GeodesicSpline(point_pair, basis, omega_init)
        optimizer = Adam([spline.omega], lr=lr)
        print(f"\n--- Optimizing spline {i+1}/{len(ctrl_pts_list)} ---")
        for step in range(steps):
            optimizer.zero_grad()
            energy = compute_energy(spline, decoder, t_vals)
            energy.backward()
            optimizer.step()
            print(f"Step {step+1:3d} | Energy: {energy.item():.6f}")
            print(f"gradient norm: {spline.omega.grad.norm().item() if spline.omega.grad is not None else 'N/A'}")
    
        optimized.append(spline)
        plot_spline_derivatives(spline, t_vals, name=f"spline_{i+1}")
    return optimized

def plot_input_paths(ctrl_pts_list, t_vals, filename):
    plt.figure(figsize=(8, 8))
    for path in ctrl_pts_list:
        plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2)
    plt.title("Input Smooth Paths")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_optimized_paths(latents, labels, grid, splines, t_vals, filename):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.5, legend=False)
    plt.scatter(grid[:, 0], grid[:, 1], s=5, alpha=0.2, color="lightblue")
    for spline in splines:
        with torch.no_grad():
            pts = spline(t_vals).cpu().numpy()
        plt.plot(pts[:, 0], pts[:, 1], '-', color='red', alpha=0.8)
    plt.title("Optimized Geodesic Splines")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_comparison(ctrl_pts_list, optimized_splines, basis, n_poly, t_vals, filename):
    plt.figure(figsize=(8, 8))
    for i, (ctrl_pts, spline) in enumerate(zip(ctrl_pts_list, optimized_splines)):
        z0 = torch.tensor(ctrl_pts[0], dtype=torch.float32)
        z1 = torch.tensor(ctrl_pts[-1], dtype=torch.float32)
        point_pair = torch.stack([z0, z1])
        omega_init = fit_initial_omega(ctrl_pts, basis, n_poly)
        initial_spline = GeodesicSpline(point_pair, basis, omega_init)

        with torch.no_grad():
            optimized_path = spline(t_vals).cpu().numpy()
            initial_path = initial_spline(t_vals).cpu().numpy()

        plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'k--', alpha=0.3)
        plt.plot(initial_path[:, 0], initial_path[:, 1], 'r--', label='initial')
        plt.plot(optimized_path[:, 0], optimized_path[:, 1], 'g-', label='optimized')

    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("Initial vs Optimized Geodesic Splines")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_spline_derivatives(spline, t_vals, name="spline"):
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



def main():
    import sys
    import src.spline_pytorch
    sys.modules['__main__'].TorchCubicSpline = src.spline_pytorch.TorchCubicSpline

    n_poly = 3 # try fewer => fewer parameters!
    t_vals = torch.linspace(0, 1, 2000)

    # decoder = CrazyDecoder(output_dim=50).eval()
    vae = AdvancedVAE(input_dim=50, latent_dim=2)
    vae.load_state_dict(torch.load("src/artifacts/vae_best_avae.pth", map_location="cpu"))
    vae.eval()
    decoder = vae.decoder

    # ctrl_pts_list = generate_smooth_test_paths(n_poly)
    with open("src/artifacts/spline_inits_torch_avae.pkl", "rb") as f:
        spline_inits = pickle.load(f)

    ctrl_pts_list = [spline(t_vals).detach().cpu().numpy() for spline in spline_inits[:1]]

    #print(len(ctrl_pts_list), ctrl_pts_list[0].shape) 2000 er mange at lave en spline fra

    plot_input_paths(ctrl_pts_list, t_vals, "src/plots/input_paths_avae.png")

    # ---- now optimize the splines ----
    basis = construct_basis(n_poly, dim=2)
    print(f"Using basis: {basis.shape} for n_poly={n_poly}")

    optimized_splines = optimize_splines(ctrl_pts_list, decoder, t_vals, basis=basis, n_poly=n_poly, steps=10, lr=1e-2, init_mode="fit")

    with open("src/artifacts/spline_optimized_torch_avae.pkl", "wb") as f:
        pickle.dump(optimized_splines, f)

    # ---- plotting ----

    latents = np.load("src/artifacts/latents_Avae_ld2_ep800_bs64_lr1e-03.npy")
    labels = np.load("data/tasic-ttypes.npy")
    grid = np.load("src/artifacts/grid.npy")

    plot_optimized_paths(latents, labels, grid, optimized_splines, t_vals, "src/plots/latents_optimized_splines_avae.png")
    
    plot_comparison(ctrl_pts_list, optimized_splines, basis, n_poly, t_vals, "src/plots/all_splines_comparison_avae.png")

if __name__ == "__main__":
    main()
