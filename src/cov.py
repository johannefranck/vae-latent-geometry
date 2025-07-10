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
    batch_size = 100
    os.makedirs(save_dir, exist_ok=True)

    geo_txt = Path(save_dir) / "cov_geo_log.txt"
    euc_txt = Path(save_dir) / "cov_euc_log.txt"
    open(geo_txt, "w").close()
    open(euc_txt, "w").close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.load(data_path), dtype=torch.float32).to(device)
    reps, pairs = load_pairs(pairfile)
    t_vals = torch.linspace(0, 1, 1000, device=device)
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)

    for num_decoders in decoder_counts:
        print(f"\nEvaluating CoV for {num_decoders} decoders")
        all_euc = {tuple(p): [] for p in pairs}
        spline_jobs = []
        spline_data = []

        for rerun in reruns:
            print(f"Rerun {rerun} for {num_decoders} decoders")
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
                    all_euc[(idx_a, idx_b)].append(euc)
                    spline_jobs.append((z0, z1, (idx_a, idx_b), decoders, rerun))

        for i in range(0, len(spline_jobs), batch_size):
            batch = spline_jobs[i:i+batch_size]
            print(f"Optimizing splines {i} to {i + len(batch) - 1}")

            a = torch.stack([job[0] for job in batch]).to(device)
            b = torch.stack([job[1] for job in batch]).to(device)
            omega_init = torch.zeros((len(batch), basis.shape[1], latent_dim), device=device)
            decoders = batch[0][3]

            try:
                spline_model, lengths, omega_opt = optimize_energy(
                    a, b, omega_init, basis, decoders,
                    n_poly=n_poly, t_vals=t_vals,
                    steps=100, lr=1e-3, ensemble=True, M=3
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
                    rerun = job[4]
                    geo = L.item()
                    euc = all_euc[(idx_a, idx_b)][rerun]
                    f_geo.write(f"{num_decoders} {idx_a} {idx_b} {geo}\n")
                    f_euc.write(f"{num_decoders} {idx_a} {idx_b} {euc}\n")
                    spline_data.append({
                        "num_decoders": num_decoders,
                        "rerun": rerun,
                        "idx_a": idx_a,
                        "idx_b": idx_b,
                        "a": job[0].cpu().numpy().tolist(),
                        "b": job[1].cpu().numpy().tolist(),
                        "omega": omega.cpu().numpy().tolist()
                    })

            del a, b, omega_init, lengths, omega_opt
            torch.cuda.empty_cache()

        with open(Path(save_dir) / f"splines_dec{num_decoders}.json", "w") as f:
            json.dump(spline_data, f)

    # === Aggregate and compute CoV ===
    def parse_file(file_path):
        cov_data = {d: {} for d in decoder_counts}
        with open(file_path, "r") as f:
            for line in f:
                d, i, j, val = line.strip().split()
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

def plot_latents_from_reruns(
    model_root="models_v103",
    data_path="data/tasic-pca50.npy",
    label_path="data/tasic-ttypes.npy",
    reruns=range(10),
    num_decoders=6,
    latent_dim=2,
    save_path="models_v103/cov_results/latent_encodings_by_rerun.png"
):
    import seaborn as sns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.load(data_path), dtype=torch.float32).to(device)
    labels = np.load(label_path)

    fig, axes = plt.subplots(2, (len(reruns) + 1) // 2, figsize=(16, 8), squeeze=False)
    axes = axes.flatten()

    for i, rerun in enumerate(reruns):
        model_path = Path(model_root) / f"dec{num_decoders}" / f"model_rerun{rerun}.pt"
        model = EVAE(input_dim=50, latent_dim=latent_dim, num_decoders=num_decoders).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            latents = model.encoder(data).mean.cpu().numpy()

        ax = axes[i]
        sns.scatterplot(
            x=latents[:, 0], y=latents[:, 1], hue=labels,
            palette="tab20", s=4, alpha=0.5, legend=False, ax=ax
        )
        ax.set_title(f"Encoder Latents (rerun {rerun})")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")

    for j in range(len(reruns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved latent projection plots to: {save_path}")




def plot_latent_geodesics_from_saved_splines(
    model_path, spline_path, data_path, pairfile, save_path, max_pairs=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model ===
    model = EVAE(input_dim=50, latent_dim=2, num_decoders=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    encoder = model.encoder

    # === Load data and latent ===
    data = torch.tensor(np.load(data_path), dtype=torch.float32).to(device)
    all_z = encoder(data).mean.detach().cpu().numpy()

    # === Load selected pairs ===
    _, pairs = load_pairs(pairfile)
    pair_set = set(tuple(p) for p in pairs[:max_pairs])

    # === Load splines ===
    with open(spline_path, "r") as f:
        spline_data = json.load(f)

    t_vals = torch.linspace(0, 1, 1000, device=device)
    n_poly = 4
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)
    basis = basis.to(device)

    plt.figure(figsize=(8, 8))
    plt.scatter(all_z[:, 0], all_z[:, 1], s=2, alpha=0.4, label="Latents")

    shown = 0
    for entry in spline_data:
        pair = (entry["idx_a"], entry["idx_b"])
        if pair not in pair_set:
            continue

        a = torch.tensor(entry["a"], dtype=torch.float32, device=device).unsqueeze(0)
        b = torch.tensor(entry["b"], dtype=torch.float32, device=device).unsqueeze(0)
        omega = torch.tensor(entry["omega"], dtype=torch.float32, device=device).unsqueeze(0)

        spline = GeodesicSplineBatch(a, b, basis, omega, n_poly=n_poly)
        zs = spline(t_vals).squeeze(1).detach().cpu().numpy()

        plt.plot(zs[:, 0], zs[:, 1], lw=2)
        plt.scatter([a[0, 0].item(), b[0, 0].item()], [a[0, 1].item(), b[0, 1].item()],
                    c="red", s=20)
        shown += 1
        if shown >= max_pairs:
            break

    plt.title("Latent space with saved geodesics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved latent space plot with geodesics to: {save_path}")

if __name__ == "__main__":
    cov_geo, cov_euc = cov_mode_ensemble()
    plot_cov_results(cov_geo, cov_euc)

    plot_latents_from_reruns(
        model_root="models_v103",
        data_path="data/tasic-pca50.npy",
        label_path="data/tasic-ttypes.npy",
        reruns=range(10),
        num_decoders=3,
        save_path="models_v103/cov_results/latent_encodings_by_rerun.png"
    )

    plot_latent_geodesics_from_saved_splines(
        model_path="models_v103/dec3/model_rerun0.pt",
        spline_path="models_v103/cov_results/splines_dec3.json",
        data_path="data/tasic-pca50.npy",
        pairfile="src/artifacts/selected_pairs_10.json",
        save_path="models_v103/cov_results/latent_geodesics.png",
        max_pairs=5
    )
