import torch
import numpy as np
import torch.nn as nn
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from src.vae import VAE
from scipy.interpolate import CubicSpline


# Load Dijkstra paths
with open("src/artifacts/dijkstra_paths.pkl", "rb") as f:
    dijkstra_paths = pickle.load(f)

# Fit spline to a path using natural cubic spline ("original")
def fit_spline_to_path(path):
    path_np = np.array(path)
    # sort path based on x values, preserving y association
    sorted_indices = np.argsort(path_np[:, 0])
    sorted_path = path_np[sorted_indices]
    x = sorted_path[:, 0]
    y = sorted_path[:, 1]
    # remove duplicate x values
    x_unique, unique_indices = np.unique(x, return_index=True)
    y_unique = y[unique_indices]
    # Fit natural cubic spline
    return CubicSpline(x_unique, y_unique, bc_type='natural')

# Parametric Natural Cubic Splines
def fit_parametric_spline(path):
    path_np = np.array(path)
    
    # Use index as parameter (or optionally arc-length)
    t = np.linspace(0, 1, len(path_np))
    
    x = path_np[:, 0]
    y = path_np[:, 1]

    # Fit spline x(t), y(t)
    spline_x = CubicSpline(t, x, bc_type='natural')
    spline_y = CubicSpline(t, y, bc_type='natural')
    
    return spline_x, spline_y

def resample_path(path, n_poly):
    path = np.array(path)
    deltas = np.diff(path, axis=0)
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    cumdist = np.concatenate([[0], np.cumsum(dists)])
    t_orig = cumdist / cumdist[-1]
    t_new = np.linspace(0, 1, n_poly + 1)
    spline_x = CubicSpline(t_orig, path[:, 0])
    spline_y = CubicSpline(t_orig, path[:, 1])
    x_new = spline_x(t_new)
    y_new = spline_y(t_new)
    return np.stack([x_new, y_new], axis=1)

def fit_parametric_spline_fixed(path, n_poly):
    ctrl_pts = resample_path(path, n_poly)
    t = np.linspace(0, 1, n_poly + 1)
    spline_x = CubicSpline(t, ctrl_pts[:, 0], bc_type='natural')
    spline_y = CubicSpline(t, ctrl_pts[:, 1], bc_type='natural')
    return spline_x, spline_y

def plot_latents_grid_and_splines(latents, labels, grid, splines, filename):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x=latents[:, 0], y=latents[:, 1],
        hue=labels, palette="tab20", s=4, alpha=0.5, legend=False
    )
    plt.scatter(grid[:, 0], grid[:, 1], s=5, alpha=0.2, color="lightblue")

    t_dense = np.linspace(0, 1, 200)
    cmap = plt.colormaps["tab10"]

    for i, (spline_x, spline_y) in enumerate(splines):
        # Dijkstra path points (green)
        dijkstra = np.array(dijkstra_paths[i])
        plt.plot(dijkstra[:, 0], dijkstra[:, 1], 'o-', linewidth=1.0, markersize=2, color='green', alpha=0.6)

        # Resampled control points
        ctrl_pts = resample_path(dijkstra_paths[i], n_poly=8)
        plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'o--', linewidth=1.2, markersize=4, color='blue', alpha=0.7)

        # Fitted spline
        x = spline_x(t_dense)
        y = spline_y(t_dense)
        plt.plot(x, y, linewidth=1.8, color='red', alpha=0.9)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("Latents, Grid, and Spline Paths")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def main():


    # Load latent grid (for reference background)
    grid = np.load("src/artifacts/grid.npy")
    # load data (for visualization)
    latents = np.load("src/artifacts/latents_ld2_ep600_bs64_lr1e-03.npy")
    labels = np.load("data/tasic-ttypes.npy")

    # Fit splines to all Dijkstra paths
    splines = []
    for i, path in enumerate(dijkstra_paths):
        spline_x, spline_y = fit_parametric_spline_fixed(path, n_poly=8)
        splines.append((spline_x, spline_y))
        print(f"Initialized spline {i+1}/{len(dijkstra_paths)}")

    # Plot
    plot_latents_grid_and_splines(latents, labels, grid, splines, "src/plots/latents_grid_splinepaths.png")

    # Save splines
    with open("src/artifacts/spline_inits.pkl", "wb") as f:
        pickle.dump(splines, f)

    print(f"Saved {len(splines)} spline inits to src/artifacts/spline_inits.pkl")

if __name__ == "__main__":
    main()
