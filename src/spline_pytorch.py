import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from src.vae import VAE


# ---- Utility ----

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


# ---- PyTorch Cubic Spline Class ----

class TorchCubicSpline(torch.nn.Module):
    def __init__(self, ctrl_pts, knots=None):
        super().__init__()
        self.n_pts = len(ctrl_pts)
        self.register_parameter('ctrl_pts', torch.nn.Parameter(torch.tensor(ctrl_pts, dtype=torch.float32)))

        self.t = torch.linspace(0, 1, self.n_pts)
        if knots is None:
            self.knots = self.t
        else:
            self.knots = torch.tensor(knots, dtype=torch.float32)

    def forward(self, t_vals):
        # Evaluate cubic spline using PyTorch's natural cubic spline via basis matrix
        #t_vals = torch.tensor(t_vals, dtype=torch.float32).unsqueeze(-1)  # shape (n_eval, 1)
        t_vals = t_vals.clone().detach().float().unsqueeze(-1) if isinstance(t_vals, torch.Tensor) else torch.tensor(t_vals, dtype=torch.float32).unsqueeze(-1)

        x_spline = self.eval_spline(t_vals, self.ctrl_pts[:, 0])
        y_spline = self.eval_spline(t_vals, self.ctrl_pts[:, 1])
        return torch.stack([x_spline, y_spline], dim=1)

    def eval_spline(self, t_vals, y):
        # Spline basis matrix for natural cubic spline
        coeffs = self.natural_cubic_spline_coeffs(self.t, y)
        t_vals = t_vals.squeeze()
        out = torch.zeros_like(t_vals)
        for i in range(self.n_pts - 1):
            mask = (t_vals >= self.t[i]) & (t_vals <= self.t[i+1])
            h = self.t[i+1] - self.t[i]
            xi = t_vals[mask] - self.t[i]
            a, b, c, d = coeffs[i]
            out[mask] = a + b*xi + c*xi**2 + d*xi**3
        return out

    def natural_cubic_spline_coeffs(self, x, y):
        n = len(x)
        h = x[1:] - x[:-1]
        alpha = (3 / h[1:]) * (y[2:] - y[1:-1]) - (3 / h[:-1]) * (y[1:-1] - y[:-2])

        A = torch.zeros((n, n), dtype=torch.float32)
        rhs = torch.zeros(n, dtype=torch.float32)

        A[0, 0] = A[-1, -1] = 1
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            rhs[i] = alpha[i-1]

        c = torch.linalg.solve(A, rhs)
        b, d = [], []
        for i in range(n-1):
            b_i = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2*c[i]) / 3
            d_i = (c[i+1] - c[i]) / (3*h[i])
            b.append(b_i)
            d.append(d_i)

        coeffs = []
        for i in range(n-1):
            a = y[i]
            coeffs.append((a, b[i], c[i], d[i]))
        return coeffs


# ---- Plotting ----

def plot_latents_with_splines(latents, labels, grid, splines, filename):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.5, legend=False)
    plt.scatter(grid[:, 0], grid[:, 1], s=5, alpha=0.2, color="lightblue")
    t_dense = torch.linspace(0, 1, 200)

    for i, spline in enumerate(splines):
        dijkstra = np.array(dijkstra_paths[i])
        plt.plot(dijkstra[:, 0], dijkstra[:, 1], 'o-', linewidth=1.0, markersize=2, color='green', alpha=0.6)

        ctrl_pts = resample_path(dijkstra_paths[i], n_poly=8)
        plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'o--', linewidth=1.2, markersize=4, color='blue', alpha=0.7)

        with torch.no_grad():
            points = spline(t_dense)
        x, y = points[:, 0], points[:, 1]
        plt.plot(x, y, linewidth=1.8, color='red', alpha=0.9)

    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("Latents, Grid, and PyTorch Splines")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---- Main ----

with open("src/artifacts/dijkstra_paths_avae.pkl", "rb") as f:
    dijkstra_paths = pickle.load(f)


def main():
    grid = np.load("src/artifacts/grid.npy")
    latents = np.load("src/artifacts/latents_Avae_ld2_ep800_bs64_lr1e-03.npy")
    labels = np.load("data/tasic-ttypes.npy")

    splines = []
    for i, path in enumerate(dijkstra_paths):
        ctrl_pts = resample_path(path, n_poly=8)
        spline = TorchCubicSpline(ctrl_pts)
        splines.append(spline)
        print(f"Initialized spline {i+1}/{len(dijkstra_paths)}")

    plot_latents_with_splines(latents, labels, grid, splines, "src/plots/latents_grid_splinepaths_pytorch_avae.png")

    with open("src/artifacts/spline_inits_torch_avae.pkl", "wb") as f:
        pickle.dump(splines, f)

    print(f"Saved {len(splines)} PyTorch splines to src/artifacts/spline_inits_torch_avae.pkl")


if __name__ == "__main__":
    main()
