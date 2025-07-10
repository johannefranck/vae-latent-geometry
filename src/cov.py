import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.vae import EVAE
from src.select_representative_pairs import load_pairs
from src.geodesics import (
    GeodesicSplineBatch,
    construct_nullspace_basis,
    optimize_energy,
)

def set_seed(seed=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def compute_cov(values):
    values = np.array(values)
    return float(values.std() / values.mean()) if values.mean() != 0 else float("inf")


def cov_mode_ensemble():
    set_seed(12)

    # --- CONFIG ---
    pairfile = "src/artifacts/selected_pairs_10.json"
    data_path = "data/tasic-pca50.npy"
    model_root = "models_v103"
    save_dir = "models_v103/cov_results"
    latent_dim = 2
    decoder_counts = [1, 2, 3, 4, 5, 6]
    reruns = list(range(10))
    n_poly = 4
    batch_size = 500

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.load(data_path), dtype=torch.float32).to(device)
    reps, pairs = load_pairs(pairfile)
    t_vals = torch.linspace(0, 1, 200, device=device)
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)

    cov_geo = {d: [] for d in decoder_counts}
    cov_euc = {d: [] for d in decoder_counts}

    for num_decoders in decoder_counts:
        print(f"\nEvaluating CoV for {num_decoders} decoders")
        all_euc = {tuple(p): [] for p in pairs}
        all_geo = {tuple(p): [] for p in pairs}

        # Accumulate all (a, b) for all reruns for batching
        spline_jobs = []  # List of (a, b, pair_key)

        for rerun in reruns:
            model_path = Path(model_root) / f"dec{num_decoders}" / f"model_rerun{rerun}.pt"
            model = EVAE(input_dim=50, latent_dim=latent_dim, num_decoders=num_decoders).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            encoder = model.encoder
            decoders = model.decoders

            with torch.no_grad():
                for idx_a, idx_b in pairs:
                    z0 = encoder(data[idx_a].unsqueeze(0)).mean.squeeze(0)
                    z1 = encoder(data[idx_b].unsqueeze(0)).mean.squeeze(0)
                    euc = torch.norm(z0 - z1).item()
                    all_euc[tuple((idx_a, idx_b))].append(euc)
                    spline_jobs.append((z0, z1, (idx_a, idx_b), decoders))  # also carry decoders for now

        print(f"[{num_decoders} decoders] Total splines to optimize: {len(spline_jobs)}")

        # === Optimize in batches ===
        for i in range(0, len(spline_jobs), batch_size):
            batch = spline_jobs[i:i+batch_size]
            print(f"Optimizing splines {i} to {i + len(batch) - 1}")

            a = torch.stack([job[0] for job in batch]).to(device)
            b = torch.stack([job[1] for job in batch]).to(device)
            omega_init = torch.zeros((len(batch), basis.shape[1], latent_dim), device=device)
            decoders = batch[0][3]  # same across all jobs in batch

            _, lengths, _ = optimize_energy(
                a, b, omega_init, basis, decoders,
                n_poly=n_poly,
                t_vals=t_vals,
                steps=500,
                lr=1e-3,
                ensemble=True,
                M=10
            )

            for (job, L) in zip(batch, lengths):
                key = job[2]  # (idx_a, idx_b)
                all_geo[tuple(key)].append(L.item())

        # Compute CoV for each pair
        for key in pairs:
            cov_euc[num_decoders].append(compute_cov(all_euc[key]))
            cov_geo[num_decoders].append(compute_cov(all_geo[key]))

    with open(Path(save_dir) / "cov_geodesic.json", "w") as f:
        json.dump(cov_geo, f, indent=2)
    with open(Path(save_dir) / "cov_euclidean.json", "w") as f:
        json.dump(cov_euc, f, indent=2)

    print("CoV results saved.")
    return cov_geo, cov_euc



def plot_cov_results(cov_geo, cov_euc):
    geo_means = {int(k): np.mean(v) for k, v in cov_geo.items()}
    euc_means = {int(k): np.mean(v) for k, v in cov_euc.items()}

    decoders = sorted(geo_means.keys())
    geo_y = [geo_means[d] for d in decoders]
    euc_y = [euc_means[d] for d in decoders]

    plt.figure(figsize=(8, 6))
    plt.plot(decoders, geo_y, label="Geodesic CoV", marker="o")
    plt.plot(decoders, euc_y, label="Euclidean CoV", marker="s")
    plt.xlabel("Number of decoders")
    plt.ylabel("Coefficient of Variation")
    plt.xticks(decoders)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models_v103/cov_results/CoV_plot.png")
    print("Saved plot to models_v103/cov_results/CoV_plot.png")

if __name__ == "__main__":
    cov_geo, cov_euc = cov_mode_ensemble()
    plot_cov_results(cov_geo, cov_euc)
