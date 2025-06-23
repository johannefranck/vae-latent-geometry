import torch
import torch.nn as nn
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from src.vae import VAE


def set_seed(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(12)


def resample_path(path, n_poly):
    path = np.array(path)
    deltas = np.diff(path, axis=0)
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    cumdist = np.concatenate([[0], np.cumsum(dists)])
    t_orig = cumdist / cumdist[-1]
    t_new = np.linspace(0, 1, n_poly + 1)
    x_new = np.interp(t_new, t_orig, path[:, 0])
    y_new = np.interp(t_new, t_orig, path[:, 1])
    return np.stack([x_new, y_new], axis=1)


def build_constraint_matrix(n):
    A = []
    t_knots = torch.linspace(0, 1, n + 1)
    row_start = torch.zeros(4 * n)
    row_start[0] = 1.0
    A.append(row_start)

    row_end = torch.zeros(4 * n)
    row_end[-4:] = torch.tensor([t_knots[-1] ** i for i in range(4)])
    A.append(row_end)

    for i in range(1, n):
        t = t_knots[i]
        for order in range(3):
            row = torch.zeros(4 * n)
            if order == 0:
                powers = torch.tensor([t ** j for j in range(4)])
            elif order == 1:
                powers = torch.tensor([0.0, 1.0, 2.0 * t, 3.0 * t ** 2])
            else:
                powers = torch.tensor([0.0, 0.0, 2.0, 6.0 * t])
            row[4 * (i - 1):4 * i] = powers
            row[4 * i:4 * (i + 1)] = -powers
            A.append(row)

    return torch.stack(A)


def nullspace(A, rtol=1e-5):
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    rank = (s > rtol * s[0]).sum()
    return vh[rank:].T


class GeodesicSpline(nn.Module):
    def __init__(self, point_pair, basis, omega_init):
        super().__init__()
        self.point_pair = point_pair
        self.basis = basis
        self.omega = nn.Parameter(omega_init.clone().detach())
        self.n_poly = basis.shape[0] // 4
        self.dim = point_pair.shape[1]

    def _eval_line(self, t):
        a, b = self.point_pair[0], self.point_pair[1]
        return (1 - t).unsqueeze(1) * a + t.unsqueeze(1) * b

    def _eval_poly(self, t):
        coeffs = self.basis @ self.omega
        coeffs = coeffs.view(self.n_poly, 4, self.dim)
        eps = 1e-6
        t_idx = (t * self.n_poly - eps).clamp(0, self.n_poly - 1)
        idx = t_idx.floor().long()
        t_local = (t * self.n_poly - idx.float()).clamp(0.0, 1.0)
        t_powers = torch.stack([t_local**i for i in range(4)], dim=1)
        coeffs_idx = coeffs[idx]
        return torch.einsum('nf,nfd->nd', t_powers, coeffs_idx)

    def forward(self, t):
        return self._eval_line(t) + self._eval_poly(t)


def fit_initial_omega(path, basis, point_pair, n_poly):
    t_vals = torch.linspace(0, 1, n_poly + 1)
    a, b = point_pair[0], point_pair[1]
    straight = (1 - t_vals).unsqueeze(1) * a + t_vals.unsqueeze(1) * b
    deviation = torch.tensor(path, dtype=torch.float32) - straight

    B = basis  # shape (4n, n_free)
    D = deviation.view(-1, deviation.shape[-1])  # shape (n_poly+1, d)

    # Project into nullspace with least squares
    # We need to match dimensions: we upsample D to shape (4n, d)
    D_upsampled = torch.nn.functional.interpolate(
        D.T.unsqueeze(0), size=(4 * n_poly), mode='linear', align_corners=True
    ).squeeze(0).T  # shape (4n, d)

    omega_init = torch.linalg.lstsq(B, D_upsampled).solution
    return omega_init



def optimize_nullspace_spline(spline, decoder, t_dense, n_steps=200, lr=1e-2):
    optimizer = torch.optim.Adam([spline.omega], lr=lr)
    path_history, losses = [], []

    for step in range(n_steps):
        optimizer.zero_grad()

        z_path = spline(t_dense)  # shape (T, latent_dim)
        dz = z_path[1:] - z_path[:-1]  # shape (T-1, latent_dim)
        dz = dz.unsqueeze(1)  # shape (T-1, 1, latent_dim)

        # --- Pullback metric ---
        G_all = []

        for i in range(len(z_path) - 1):
            z = z_path[i:i+1]  # keep batched
            z.requires_grad_(True)  # enables tracing back to omega
            x = decoder(z)[0]  # (x_dim,)
            J_rows = []

            for j in range(x.shape[0]):
                grad_j = torch.autograd.grad(x[j], z, retain_graph=True, create_graph=True)[0]
                J_rows.append(grad_j)
            J = torch.stack(J_rows)  # (x_dim, latent_dim)
            G = J.T @ J  # (latent_dim, latent_dim)
            G_all.append(G.unsqueeze(0))

        G_all = torch.cat(G_all, dim=0)  # (T-1, latent_dim, latent_dim)

        energy = torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2)).mean()
        energy.backward()
        optimizer.step()

        with torch.no_grad():
            path_history.append(z_path.detach().cpu().numpy())
            losses.append(energy.item())

        print(f"Step {step+1:03d} | Energy: {energy.item():.6f} | ω norm: {spline.omega.norm().item():.4f} | Grad norm: {spline.omega.grad.norm().item():.6f}")

    return losses, path_history









def plot_all_splines(latents, labels, ctrl_pts_list, splines_opt, filename):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.4, legend=False)
    t_dense = torch.linspace(0, 1, 500)

    for ctrl_pts, spline in zip(ctrl_pts_list, splines_opt):
        plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'r--', linewidth=1.2, alpha=0.7)
        with torch.no_grad():
            path = spline(t_dense).cpu().numpy()
        plt.plot(path[:, 0], path[:, 1], 'g-', linewidth=1.5, alpha=0.9)

    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.title("Initial (red dashed) vs Optimized (green) Splines")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # --- Hyperparameters ---
    n_poly = 5  # number of polynomial segments in spline
    t_resolution = 500  # how many t points on spline

    vae = VAE()
    vae.load_state_dict(torch.load("src/artifacts/vae_best.pth", map_location="cpu"))
    vae.eval()
    decoder = vae.decoder

    # SANITY CHECK
    with torch.no_grad():
        z_test = torch.randn(100, 2)
        x_test = decoder(z_test)
        print("Decoder output std (should be > 0.1):", x_test.std().item())

    latents = np.load("src/artifacts/latents_ld2_ep600_bs64_lr1e-03.npy")
    labels = np.load("data/tasic-ttypes.npy")

    with open("src/artifacts/dijkstra_paths.pkl", "rb") as f:
        dijkstra_paths = pickle.load(f)

    A = build_constraint_matrix(n_poly)
    basis = nullspace(A)
    t_dense = torch.linspace(0, 1, t_resolution)

    ctrl_pts_list = []
    splines_opt = []

    for path in dijkstra_paths[:2]:
        ctrl_pts = resample_path(path, n_poly)
        a = torch.tensor(ctrl_pts[0], dtype=torch.float32)
        b = torch.tensor(ctrl_pts[-1], dtype=torch.float32)
        point_pair = torch.stack([a, b], dim=0)

        omega_init = fit_initial_omega(ctrl_pts, basis, point_pair, n_poly=5)
        spline = GeodesicSpline(point_pair, basis, omega_init)
        assert spline.omega.grad is not None and spline.omega.grad.norm().item() > 0, "No gradient flow to ω!"

        # Kick the spline out of flat minimum early
        spline.omega.data += 0.01 * torch.randn_like(spline.omega)

        ctrl_pts_list.append(ctrl_pts)
        losses, _ = optimize_nullspace_spline(spline, decoder, t_dense, n_steps=10, lr=5e-3)
        print("Final ω norm:", spline.omega.norm().item())

        splines_opt.append(spline)


    with torch.no_grad():
        path = spline(t_dense).cpu().numpy()
        start, end = path[0], path[-1]
        midpoint = path[len(path) // 2]
        deviation = np.linalg.norm(midpoint - 0.5 * (start + end))
        print(f"Midpoint deviation from line: {deviation:.4f}")


    plot_all_splines(latents, labels, ctrl_pts_list, splines_opt, "src/plots/all_splines_comparison.png")


if __name__ == "__main__":
    main()
