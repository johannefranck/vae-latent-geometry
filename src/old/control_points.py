import torch
import numpy as np
import torch.nn as nn
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.vae import VAE

# import first djikstra paths as points
with open("src/artifacts/dijkstra_paths.pkl", "rb") as f:
    dijkstra_paths = pickle.load(f)

print("Loaded Dijkstra paths:", dijkstra_paths[0])


# fit an initialization of a natural cubic spline to the first path via the control points
from scipy.interpolate import CubicSpline
def fit_spline_to_path(path):
    path_np = np.array(path)
    
    # Sort path based on x values, preserving y association
    sorted_indices = np.argsort(path_np[:, 0])
    sorted_path = path_np[sorted_indices]
    
    x = sorted_path[:, 0]
    y = sorted_path[:, 1]

    # Fit natural cubic spline
    cs = CubicSpline(x, y, bc_type='natural')
    return cs


# Fit a spline to the first Dijkstra path
spline = fit_spline_to_path(dijkstra_paths[0])

# Plot the fitted spline on top of the Dijkstra points
def plot_spline_with_dijkstra(path, spline):
    path_np = np.array(path)
    x = path_np[:, 0]
    y = path_np[:, 1]

    # Generate dense points for the spline
    x_dense = np.linspace(x.min(), x.max(), 100)
    y_dense = spline(x_dense)

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=3, alpha=1, color="blue", label="Dijkstra Path")
    plt.plot(x_dense, y_dense, color='red', label='Fitted Spline')
    plt.title("Dijkstra Path with Fitted Spline")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.savefig("src/plots/spline_fitted_to_dijkstra_path.png")
    plt.close()

plot_spline_with_dijkstra(dijkstra_paths[0], spline)

