import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.vae_good import VAE
from src.select_representative_pairs import load_pairs
from src.optimize_energy import GeodesicSpline
from src.plotting import plot_latent_density_with_splines

# ---- CONFIG ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_DIR = Path("src/artifacts")
PLOT_DIR = Path("src/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = PLOT_DIR / "density_with_splines.png"

# ---- LOAD LATENTS + LABELS ----
latents = np.load(ARTIFACT_DIR / "latents_VAE_ld2_ep100_bs64_lr1e-03.npy")
labels = np.load("data/tasic-ttypes.npy")

# ---- LOAD VAE + DECODER ----
vae = VAE(input_dim=50, latent_dim=2).to(device)
vae.load_state_dict(torch.load(ARTIFACT_DIR / "vae_best_avae.pth", map_location=device))
vae.eval()

# ---- LOAD SPLINE BATCH ----
batch_data = torch.load(ARTIFACT_DIR / "spline_batch_optimized.pt", map_location=device)
representatives, _ = load_pairs(ARTIFACT_DIR / "selected_pairs.json")
cluster_ids = [rep["label"] for rep in representatives]
spline_pairs = []

for entry in batch_data:
    a = entry["a"].to(device)
    b = entry["b"].to(device)
    n_poly = entry["n_poly"]
    basis = entry["basis"].to(device)
    omega_init = entry["omega_init"].to(device)
    omega_opt = entry["omega_optimized"].to(device)

    pair = (a, b)

    spline_init = GeodesicSpline(pair, basis, n_poly).to(device)
    spline_init.omega.data.copy_(omega_init)

    spline_opt = GeodesicSpline(pair, basis, n_poly).to(device)
    spline_opt.omega.data.copy_(omega_opt)

    spline_pairs.append((spline_init, spline_opt))

# ---- CONSTRUCT GEODESIC DISTANCE MATRIX ----
# Map point tensors to indices for matrix layout
point_set = []
index_map = {}
current_index = 0

# First collect all unique points
for entry in batch_data:
    a_tuple = tuple(entry["a"].view(-1).tolist())
    b_tuple = tuple(entry["b"].view(-1).tolist())
    for pt in [a_tuple, b_tuple]:
        if pt not in index_map:
            index_map[pt] = current_index
            point_set.append(pt)
            current_index += 1

n_points = len(point_set)
distance_matrix = np.full((n_points, n_points), np.nan)

# Fill in distances
for entry in batch_data:
    a_idx = index_map[tuple(entry["a"].view(-1).tolist())]
    b_idx = index_map[tuple(entry["b"].view(-1).tolist())]
    dist = entry["length_geodesic"]
    distance_matrix[a_idx, b_idx] = dist
    distance_matrix[b_idx, a_idx] = dist  # symmetric

# ---- PLOT DISTANCE MATRIX ----
# ---- PLOT with cluster IDs ----
plt.figure(figsize=(8, 6))
sns.heatmap(distance_matrix, 
            annot=True, fmt=".1f", cmap="viridis",
            xticklabels=cluster_ids, yticklabels=cluster_ids,
            cbar_kws={"label": "Geodesic Distance"})
plt.title("Geodesic Distance Matrix (Decoder Space)")
plt.xlabel("Cluster ID")
plt.ylabel("Cluster ID")
plt.tight_layout()
plt.savefig(PLOT_DIR / "geodesic_distance_matrix_labeled.png", dpi=300)
print("Saved: geodesic_distance_matrix_labeled.png")


# ---- PLOT DENSITY WITH SPLINES INIT AND OPT ----
plot_latent_density_with_splines(
    latents=latents,
    labels=labels,
    splines=spline_pairs,
    filename=str(PLOT_PATH)
)

print(f"Saved: {PLOT_PATH}")
