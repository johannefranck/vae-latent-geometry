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
from src.plotting import (
    build_distance_matrices, 
    plot_cov_results, 
    plot_latents_from_reruns, 
    plot_latent_geodesics_from_saved_splines
)

def set_seed(seed=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def compute_cov(values):
    values = np.array(values)
    return float(values.std() / values.mean()) if values.mean() != 0 else float("inf")

def cov_mode_ensemble(pairs=10, num_decoders=6):
    set_seed(12)

    # --- CONFIG ---
    pairfile = f"src/artifacts/selected_pairs_{pairs}.json"
    data_path = "data/tasic-pca50.npy"
    model_root = "models_v103"
    model_save_dir = f"models_p{pairs}"
    save_dir = f"{model_save_dir}/cov_results"
    latent_dim = 2
    decoder_counts = list(range(1, num_decoders + 1))  # [1, 2, ..., num_decoders]
    reruns = list(range(10))
    n_poly = 4
    batch_size = 500
    n_segments = 1000 # segments pr spline, t discretization
    os.makedirs(save_dir, exist_ok=True)

    geo_txt = Path(save_dir) / "cov_geo_log.txt"
    euc_txt = Path(save_dir) / "cov_euc_log.txt"
    open(geo_txt, "w").close()
    open(euc_txt, "w").close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.load(data_path), dtype=torch.float32).to(device)
    reps, pairs = load_pairs(pairfile)
    t_vals = torch.linspace(0, 1, n_segments, device=device)
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)

    for num_decoders in decoder_counts:
        print(f"\nEvaluating CoV for {num_decoders} decoders")
        all_euc = {tuple(p): {} for p in pairs}
        spline_data = []

        # --- Store spline jobs per rerun ---
        all_spline_jobs = dict()  # rerun (spline_jobs, decoders)

        for rerun in reruns:
            print(f"Rerun {rerun} for {num_decoders} decoders")
            model_path = Path(model_root) / f"dec{num_decoders}" / f"model_rerun{rerun}.pt"
            model = EVAE(input_dim=50, latent_dim=latent_dim, num_decoders=num_decoders).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            encoder = model.encoder
            decoders = list(model.decoders)

            spline_jobs = []
            with torch.no_grad():
                for idx_a, idx_b in pairs:
                    z0 = encoder(data[idx_a].unsqueeze(0)).mean.squeeze(0)
                    z1 = encoder(data[idx_b].unsqueeze(0)).mean.squeeze(0)
                    euc = torch.norm(z0 - z1).item()
                    all_euc[(idx_a, idx_b)][rerun] = euc
                    spline_jobs.append((z0, z1, (idx_a, idx_b)))

            all_spline_jobs[rerun] = (spline_jobs, decoders)

        # --- Optimize per rerun with correct decoder ---
        for rerun, (spline_jobs, decoders) in all_spline_jobs.items():
            for i in range(0, len(spline_jobs), batch_size):
                batch = spline_jobs[i:i+batch_size]
                print(f"Optimizing splines (rerun {rerun}) {i} to {i + len(batch) - 1}")

                a = torch.stack([job[0] for job in batch]).to(device)
                b = torch.stack([job[1] for job in batch]).to(device)
                omega_init = torch.zeros((len(batch), basis.shape[1], latent_dim), device=device)

                try:
                    spline_model, lengths, omega_opt = optimize_energy(
                        a, b, omega_init, basis, decoders,
                        n_poly=n_poly, t_vals=t_vals,
                        steps=5000, lr=1e-2, ensemble=True, M=5
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("CUDA OOM. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                with open(geo_txt, "a") as f_geo, open(euc_txt, "a") as f_euc:
                    for (job, L, omega) in zip(batch, lengths, omega_opt):
                        idx_a, idx_b = job[2]
                        geo = L.item()
                        euc = all_euc[(idx_a, idx_b)].get(rerun)
                        if euc is None:
                            print(f"Missing or misaligned euc for rerun {rerun}, pair {(idx_a, idx_b)}")
                            continue

                        f_geo.write(f"{num_decoders} {rerun} {idx_a} {idx_b} {geo}\n")
                        f_euc.write(f"{num_decoders} {rerun} {idx_a} {idx_b} {euc}\n")

                        # Store spline data
                        spline_data.append({
                            "num_decoders": num_decoders,
                            "rerun": rerun,
                            "idx_a": idx_a,
                            "idx_b": idx_b,
                            "a": job[0].cpu().numpy().tolist(),
                            "b": job[1].cpu().numpy().tolist(),
                            "omega": omega.cpu().numpy().tolist(),
                            "geo_distance": geo,
                            "euc_distance": euc
                        })

                del a, b, omega_init, lengths, omega_opt
                torch.cuda.empty_cache()

        Path(f"{model_save_dir}/results").mkdir(parents=True, exist_ok=True)
        with open(f"{model_save_dir}/results/splines_dec{num_decoders}.json", "w") as f:
            json.dump(spline_data, f)

    # === Aggregate and compute CoV ===
    def parse_file(file_path):
        cov_data = {d: {} for d in decoder_counts}
        with open(file_path, "r") as f:
            for line in f:
                d, r, i, j, val = line.strip().split()
                d, i, j = int(d), int(i), int(j)
                val = float(val)
                cov_data[d].setdefault((i, j), []).append(val)
        return {d: [compute_cov(vals) for vals in cov_data[d].values()] for d in decoder_counts}

    cov_geo = parse_file(geo_txt)
    cov_euc = parse_file(euc_txt)

    with open(Path(save_dir) / "cov_geodesic.json", "w") as f:
        json.dump(cov_geo, f, indent=2)
    with open(Path(save_dir) / "cov_euclidean.json", "w") as f:
        json.dump(cov_euc, f, indent=2)

    print("CoV results saved.")
    return cov_geo, cov_euc


if __name__ == "__main__":
    pairs = 50 # corresponds to selected_pairs_10.json, choose 10,50,100,133
    rerun = 0
    num_decoders = 6
    model_save_dir = f"models_p{pairs}"

    cov_geo, cov_euc = cov_mode_ensemble(pairs=pairs)
    plot_cov_results(cov_geo, cov_euc, pairs=pairs)

    build_distance_matrices(
        spline_json_path=f"{model_save_dir}/results/splines_dec{num_decoders}.json",
        rerun=rerun,
        num_decoders=num_decoders,
        cluster_map_path=f"src/artifacts/selected_pairs_{pairs}.json",
        plot_path_geo=f"{model_save_dir}/results/geo_matrix_dec{num_decoders}_rerun{rerun}.png",
        plot_path_euc=f"{model_save_dir}/results/euc_matrix_dec{num_decoders}_rerun{rerun}.png",
        json_out_path=f"{model_save_dir}/results/distances_dec{num_decoders}_rerun{rerun}.json"
    )

    # plot_latents_from_reruns(
    #     model_root="models_v103",
    #     data_path="data/tasic-pca50.npy",
    #     label_path="data/tasic-ttypes.npy",
    #     reruns=range(10),
    #     num_decoders=num_decoders,
    #     save_path=f"{model_save_dir}/results/latent_encodings_by_rerun{rerun}.png"
    # )

    plot_latent_geodesics_from_saved_splines(
        model_path=f"models_v103/dec{num_decoders}/model_rerun{rerun}.pt",
        spline_path=f"{model_save_dir}/results/splines_dec{num_decoders}.json",
        data_path="data/tasic-pca50.npy",
        pairfile=f"src/artifacts/selected_pairs_{pairs}.json",
        save_path=f"{model_save_dir}/results/latent_geodesics_r{rerun}_dec{num_decoders}.png",
        num_decoders=num_decoders,
        max_pairs=5
    )
