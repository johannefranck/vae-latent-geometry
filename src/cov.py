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

def remap_old_decoder_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # If old key like 'decoder.0.xxx', turn into 'decoders.0.xxx'
        if k.startswith("decoder.") and k[8].isdigit():
            parts = k.split(".")
            parts[0] = "decoders"
            new_key = ".".join(parts)
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def cov_mode_ensemble():
    set_seed(12)

    # --- CONFIG ---
    pairfile = "src/artifacts/selected_pairs_10.json"
    data_path = "data/tasic-pca50.npy"
    model_root = "models_v103"
    save_dir = "models_v103/cov_results"
    latent_dim = 2
    decoder_counts = [1, 2, 3, 4, 5, 6]
    reruns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Reruns for each decoder count
    n_poly = 4

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.load(data_path), dtype=torch.float32).to(device)
    reps, pairs = load_pairs(pairfile)
    t_vals = torch.linspace(0, 1, 2000, device=device)
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)

    cov_geo = {d: [] for d in decoder_counts}
    cov_euc = {d: [] for d in decoder_counts}

    for num_decoders in decoder_counts:
        print(f"\nEvaluating CoV for {num_decoders} decoders")

        for idx_a, idx_b in pairs:
            geo_dists = []
            euc_dists = []

            for rerun in reruns:
                model_path = Path(model_root) / f"dec{num_decoders}" / f"model_rerun{rerun}.pt"
                model = EVAE(input_dim=50, latent_dim=latent_dim, num_decoders=num_decoders).to(device)
                state = torch.load(model_path, map_location=device)
                state = remap_old_decoder_keys(state)
                model.load_state_dict(state)

                model.eval()

                print(model.decoders)  # <- should print 3 decoders


                with torch.no_grad():
                    z0 = model.encoder(data[idx_a].unsqueeze(0)).mean.squeeze(0)
                    z1 = model.encoder(data[idx_b].unsqueeze(0)).mean.squeeze(0)

                euc_dist = torch.norm(z0 - z1, p=2).item()
                euc_dists.append(euc_dist)

                # a = z0.unsqueeze(0)
                # b = z1.unsqueeze(0)

                # omega = torch.zeros((1, basis.shape[1], latent_dim), device=device)
                # spline_init = GeodesicSplineBatch(a, b, basis, omega, n_poly)

                # # NEED TO DO BATCHED OPTIMIZATION LIKE IN MAIN OF GEODESICS.py!

                # spline_opt, lengths, _ = optimize_energy(
                #     a, b, omega, basis, model.decoders,
                #     n_poly=n_poly,
                #     t_vals=t_vals,
                #     steps=500,
                #     lr=1e-3,
                #     ensemble=True,
                #     M=10
                # )
                # geo_dists.append(lengths[0].item())


            cov_euc[num_decoders].append(compute_cov(euc_dists))
            # cov_geo[num_decoders].append(compute_cov(geo_dists))

    # Save results
    # with open(Path(save_dir) / "cov_geodesic.json", "w") as f:
    #     json.dump(cov_geo, f, indent=2)
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
