import torch
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.single_decoder.vae import VAE
from src.single_decoder.optimize_energy import GeodesicSpline
from src.plotting import plot_latent_density_with_splines

# ---- PARSE ARGUMENTS ----
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Seed used to identify model/data")
parser.add_argument("--pairs_path", type=str, default="src/artifacts/selected_pairs.json", help="Path to selected pairs JSON")
args = parser.parse_args()
seed = args.seed
pairs_path = Path(args.pairs_path)
pair_tag = pairs_path.stem.replace("selected_pairs_", "")

# ---- CONFIG ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_DIR = Path("src/artifacts")
PLOT_DIR = Path("src/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

latents_path = ARTIFACT_DIR / f"latents_VAE_ld2_ep100_bs64_lr1e-03_seed{seed}.npy"
decoder_path = ARTIFACT_DIR / f"vae_best_seed{seed}.pth"
spline_path = ARTIFACT_DIR / f"spline_batch_optimized_batched_seed{seed}_p{pair_tag}.pt"
plot_path_density = PLOT_DIR / f"density_with_splines_seed{seed}_p{pair_tag}.png"
plot_path_matrix = PLOT_DIR / f"geodesic_distance_seed{seed}_p{pair_tag}.png"
json_path = ARTIFACT_DIR / f"geodesic_distances_seed{seed}_p{pair_tag}.json"


# ---- LOAD LATENTS + LABELS ----
latents = np.load(latents_path)
labels = np.load("data/tasic-ttypes.npy")

# ---- LOAD VAE + DECODER ----
vae = VAE(input_dim=50, latent_dim=2).to(device)
vae.load_state_dict(torch.load(decoder_path, map_location=device))
vae.eval()
decoder = vae.decoder

# ---- LOAD SPLINE BATCH ----
batch_data = torch.load(spline_path, map_location=device)
print(f"Loaded {len(batch_data)} optimized splines from: {spline_path}")

spline_pairs = []
point_set = []
index_map = {}
label_map = {}
current_index = 0

for entry in batch_data:
    a = entry["a"].to(device)
    b = entry["b"].to(device)
    a_label, b_label = entry["cluster_pair"]
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

    for pt, label in zip([a, b], [a_label, b_label]):
        pt_key = tuple(pt.view(-1).tolist())
        if pt_key not in index_map:
            index_map[pt_key] = current_index
            label_map[pt_key] = label
            point_set.append(pt_key)
            current_index += 1
cluster_ids = [label_map[pt] for pt in point_set]


# ---- CONSTRUCT GEODESIC DISTANCE MATRIX ----
point_set = []
index_map = {}
current_index = 0

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
assert len(cluster_ids) == distance_matrix.shape[0], "Label count mismatch with distance matrix!"


for entry in batch_data:
    a_idx = index_map[tuple(entry["a"].view(-1).tolist())]
    b_idx = index_map[tuple(entry["b"].view(-1).tolist())]
    dist = entry["length_geodesic"]
    distance_matrix[a_idx, b_idx] = dist
    distance_matrix[b_idx, a_idx] = dist

# ---- PLOT GEODESIC DISTANCE MATRIX ----
np.fill_diagonal(distance_matrix, 0.0)
print(f"Distance matrix shape: {distance_matrix.shape}")
plt.figure(figsize=(12, 12))
sns.heatmap(
    distance_matrix,
    cmap="copper",
    square=True,
    annot=False,
    # fmt=".1f",
    xticklabels=cluster_ids,
    yticklabels=cluster_ids,
    cbar=False,
    # cbar_kws={"label": "Geodesic Distance"},
)
plt.xticks(rotation=90, fontsize=3)
plt.yticks(rotation=0, fontsize=3)
plt.title(f"Geodesic Distance Matrix Single Decoder (seed {seed})")
plt.xlabel("Cluster ID")
plt.ylabel("Cluster ID")
plt.tight_layout()
plt.savefig(plot_path_matrix, dpi=300)
print(f"Saved: {plot_path_matrix}")

# ---- SAVE DISTANCE MATRIX AS JSON ----
json_matrix = {
    "seed": seed,
    "cluster_ids": cluster_ids,
    "distance_matrix": distance_matrix.tolist()
}
with open(json_path, "w") as f:
    json.dump(json_matrix, f, indent=2)
print(f"Saved: {json_path}")

# ---- PLOT LATENT DENSITY WITH SPLINES ----
plot_latent_density_with_splines(
    latents=latents,
    labels=labels,
    splines=spline_pairs,
    seed=seed,
    filename=str(plot_path_density)
)

print(f"Saved: {plot_path_density}")
