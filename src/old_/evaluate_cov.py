import os
import json
import torch
import numpy as np
from pathlib import Path
from src.vae import EVAE
from src.select_representative_pairs import load_pairs
from src.geodesics import construct_nullspace_basis, optimize_energy
from src.plotting import plot_cov_results


def compute_cov(values):
    values = np.array(values)
    return float(values.std() / values.mean()) if values.mean() != 0 else float("inf")


def run_cov_analysis_aml_style(
    model_root="models_v105",
    model_save_dir="models_cov_output",
    latent_dim=2,
    max_decoders=10,
    reruns=range(3),
    pairs_json="src/artifacts/selected_pairs_10.json",
    data_path="data/tasic-pca50.npy",
    steps=200,
):
    os.makedirs(f"{model_save_dir}/cov_results", exist_ok=True)
    geo_txt = Path(model_save_dir) / "cov_results/cov_geo_log.txt"
    euc_txt = Path(model_save_dir) / "cov_results/cov_euc_log.txt"
    open(geo_txt, "w").close()
    open(euc_txt, "w").close()

    data = torch.tensor(np.load(data_path), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    reps, pairs = load_pairs(pairs_json)
    t_vals = torch.linspace(0, 1, 1000, device=device)
    basis, _ = construct_nullspace_basis(n_poly=4, device=device)

    for rerun in reruns:
        print(f"[Rerun {rerun}]")
        model_path = Path(model_root) / f"dec10" / f"model_rerun{rerun}.pt"
        model = EVAE(input_dim=data.shape[1], latent_dim=latent_dim, num_decoders=10).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        encoder = model.encoder

        with torch.no_grad():
            embeddings = {
                (i, j): (
                    encoder(data[i].unsqueeze(0)).mean.squeeze(0),
                    encoder(data[j].unsqueeze(0)).mean.squeeze(0),
                )
                for i, j in pairs
            }

        for num_decoders in range(1, max_decoders + 1):
            print(f"  Testing with {num_decoders} decoders...")
            decoders = list(model.decoders[:num_decoders])
            spline_jobs = []

            for (i, j), (z0, z1) in embeddings.items():
                spline_jobs.append((z0, z1, (i, j), torch.norm(z0 - z1).item()))

            for batch in spline_jobs:
                z0, z1, (i, j), euc = batch
                omega_init = torch.zeros((1, basis.shape[1], latent_dim), device=device)

                try:
                    _, lengths, _ = optimize_energy(
                        z0.unsqueeze(0),
                        z1.unsqueeze(0),
                        omega_init,
                        basis,
                        decoders,
                        n_poly=4,
                        t_vals=t_vals,
                        steps=steps,
                        lr=1e-2,
                        ensemble=True,
                        M=10,
                    )
                    geo = lengths.item()
                except RuntimeError as e:
                    print(f"Skipping ({i}, {j}) due to error: {e}")
                    continue

                with open(geo_txt, "a") as f:
                    f.write(f"{num_decoders} {rerun} {i} {j} {geo}\n")
                with open(euc_txt, "a") as f:
                    f.write(f"{num_decoders} {rerun} {i} {j} {euc:.17g}\n")


def parse_cov_logs(model_save_dir, max_decoders=10):
    def parse_log(file_path):
        cov_data = {k: {} for k in range(1, max_decoders + 1)}
        with open(file_path, "r") as f:
            for line in f:
                d, r, i, j, val = line.strip().split()
                d = int(d)
                key = (int(i), int(j))
                val = float(val)
                if d not in cov_data:
                    continue
                cov_data[d].setdefault(key, []).append(val)
        return {d: [compute_cov(vs) for vs in cov_data[d].values()] for d in cov_data}

    cov_geo = parse_log(Path(model_save_dir) / "cov_results/cov_geo_log.txt")
    cov_euc = parse_log(Path(model_save_dir) / "cov_results/cov_euc_log.txt")

    with open(Path(model_save_dir) / "cov_results/cov_geodesic.json", "w") as f:
        json.dump(cov_geo, f, indent=2)
    with open(Path(model_save_dir) / "cov_results/cov_euclidean.json", "w") as f:
        json.dump(cov_euc, f, indent=2)

    return cov_geo, cov_euc


if __name__ == "__main__":
    reruns = list(range(2))
    run_cov_analysis_aml_style(
        model_root="models_v105",
        model_save_dir="models_cov_output",
        latent_dim=2,
        max_decoders=10,
        reruns=reruns,
        pairs_json="src/artifacts/selected_pairs_10.json",
        data_path="data/tasic-pca50.npy",
        steps=200,
    )

    cov_geo, cov_euc = parse_cov_logs("models_cov_output", max_decoders=10)
    plot_cov_results(cov_geo, cov_euc, model_save_dir="models_cov_output")
