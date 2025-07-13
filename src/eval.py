import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from src.train import EVAE, GaussianEncoder, GaussianDecoder, GaussianPrior, make_encoder_net, make_decoder_net
from src.single_decoder.optimize_energy import construct_nullspace_basis
from src.optimize import GeodesicSplineBatch, compute_energy_mc

def plot_geodesic_matrix(spline_blob, output_path, seed=None, init_type=None):
    spline_data = spline_blob["spline_data"]
    reps = spline_blob["representatives"]
    if reps is None:
        raise ValueError("Missing 'representatives' in spline blob. Cannot build label mapping.")

    # Map global index (from original data) to local rep index
    global_to_local = {r["index"]: i for i, r in enumerate(reps)}

    def get_label(r):
        return (
            r.get("cluster_label") or
            r.get("label") or
            r.get("index") or
            str(reps.index(r))
        )
    labels = [get_label(r) for r in reps]

    n = len(reps)
    dist_mat = np.full((n, n), np.nan)

    skipped = 0
    for d in spline_data:
        a_global = d["a_index"]
        b_global = d["b_index"]
        if a_global not in global_to_local or b_global not in global_to_local:
            skipped += 1
            continue
        a_local = global_to_local[a_global]
        b_local = global_to_local[b_global]
        dist = d["geodesic_length"]
        dist_mat[a_local, b_local] = dist
        dist_mat[b_local, a_local] = dist

    np.fill_diagonal(dist_mat, 0)
    if skipped:
        print(f"[INFO] Skipped {skipped} spline entries not in representative set")

    plt.figure(figsize=(10, 10))
    sns.heatmap(dist_mat, square=True, xticklabels=labels, yticklabels=labels, cmap="copper", cbar=False)
    plt.xticks(rotation=90, fontsize=4)
    plt.yticks(rotation=0, fontsize=4)
    plt.title(f"Geodesic Distance Matrix - seed {seed} (init by {init_type})")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Saved geodesic matrix plot to: {output_path}")



def compute_cov(values):
    values = np.array(values)
    return np.std(values) / np.mean(values) if np.mean(values) > 0 else 0.0

def run_cov_analysis(seeds, decoder_counts, pairfile, model_dir, data_path, output_plot):
    latent_dim = 2
    input_dim = 50
    num_t = 2000
    mc_samples = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and point pairs
    data = torch.tensor(np.load(data_path).astype(np.float32)).to(device)
    with open(pairfile, "r") as f:
        pair_data = json.load(f)
    pairs = pair_data["pairs"]

    cov_geodesic = {k: [] for k in decoder_counts}
    cov_euclidean = []

    for pair_idx, (idx_a, idx_b) in enumerate(tqdm(pairs, desc="Evaluating pairs")):
        geo_dist_by_decoder = {k: [] for k in decoder_counts}
        euc_dist = []

        for seed in seeds:
            encoder = GaussianEncoder(make_encoder_net(input_dim, latent_dim))
            decoder_template = GaussianDecoder(make_decoder_net(latent_dim, input_dim))
            prior = GaussianPrior(latent_dim)
            model = EVAE(prior, encoder, decoder_template, num_decoders=10).to(device)
            model.load_state_dict(torch.load(f"{model_dir}/model_seed{seed}.pt", map_location=device))
            model.eval()

            with torch.no_grad():
                z_a = model.encoder(data[idx_a:idx_a+1]).base_dist.loc.squeeze(0)
                z_b = model.encoder(data[idx_b:idx_b+1]).base_dist.loc.squeeze(0)

            euc_dist.append(torch.norm(z_a - z_b).item())

            basis, _ = construct_nullspace_basis(n_poly=4, device=device)
            omega_init = torch.zeros((1, basis.shape[1], latent_dim), device=device)
            t_vals = torch.linspace(0, 1, num_t, device=device)

            for k in decoder_counts:
                sub_decoders = [model.decoder[i] for i in range(k)]
                a = z_a.unsqueeze(0)
                b = z_b.unsqueeze(0)
                spline_model = GeodesicSplineBatch(a, b, basis, omega_init.clone(), n_poly=4).to(device)
                optimizer = torch.optim.Adam([spline_model.omega], lr=1e-3)

                for step in range(300):
                    optimizer.zero_grad()
                    energy = compute_energy_mc(spline_model, sub_decoders, t_vals, M=mc_samples)
                    endpoint_loss = ((spline_model(t_vals[-1:]) - b.unsqueeze(0)) ** 2).sum()
                    loss = energy + 1000 * endpoint_loss
                    loss.backward()
                    optimizer.step()

                geo_length = torch.sqrt(energy).item()
                geo_dist_by_decoder[k].append(geo_length)

        for k in decoder_counts:
            cov_k = compute_cov(geo_dist_by_decoder[k])
            cov_geodesic[k].append(cov_k)

        cov_euclidean.append(compute_cov(euc_dist))

    avg_cov_geo = {k: np.mean(cov_geodesic[k]) for k in decoder_counts}
    avg_cov_euc = np.mean(cov_euclidean)

    # === Save CoV results to JSON ===
    def convert_numpy(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.item() if obj.size == 1 else obj.tolist()
        return obj

    cov_data = {
        "avg_cov_geodesic": {str(k): convert_numpy(v) for k, v in avg_cov_geo.items()},
        "avg_cov_euclidean": convert_numpy(avg_cov_euc),
        "raw_cov_geodesic": {str(k): [convert_numpy(v) for v in vals] for k, vals in cov_geodesic.items()},
        "raw_cov_euclidean": [convert_numpy(v) for v in cov_euclidean],
        "seeds": seeds,
        "decoder_counts": decoder_counts,
        "num_pairs": len(pairs)
    }

    json_path = Path(output_plot).with_name(f"cov_values_{Path(output_plot).stem.split('_')[-1]}.json")
    with open(json_path, "w") as f:
        json.dump(cov_data, f, indent=2)

    print(f"[✓] Saved CoV values to: {json_path}")

    plt.figure(figsize=(8, 5))
    x = decoder_counts
    y_geo = [avg_cov_geo[k] for k in x]
    y_euc = [avg_cov_euc] * len(x)

    plt.plot(x, y_geo, marker='o', label='Geodesic CoV')
    plt.plot(x, y_euc, linestyle='--', label='Euclidean CoV')
    plt.xlabel("Number of Decoders")
    plt.xticks(x)
    plt.ylabel("Average Coefficient of Variation (CoV)")
    plt.title("CoV vs Number of Decoders")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"[✓] Saved CoV plot to: {output_plot}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["matrix", "cov"], required=True)
    parser.add_argument("--init-type", type=str, default="entropy", choices=["entropy", "euclidean"])
    parser.add_argument("--pair-count", type=int, default=10)
    parser.add_argument("--seed", type=int, help="Seed number, e.g., 123 for model_seed123.pt, for matrix mode")
    parser.add_argument("--seeds", nargs="*", type=int, default=[12, 123], help="Specify models for cov analysis")
    args = parser.parse_args()

    plot_dir = Path("experiment/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "matrix":
        spline_path = Path("experiment") / f"splines_opt_model_seed{args.seed}" / f"spline_batch_opt_{args.init_type}_{args.pair_count}.pt"
        if not spline_path.exists():
            print(f"[ERROR] File not found: {spline_path}")
            return
        spline_blob = torch.load(spline_path)
        plot_path = plot_dir / f"geodesic_matrix_seed{args.seed}_{args.init_type}_{args.pair_count}.png"
        plot_geodesic_matrix(spline_blob, plot_path, seed=args.seed, init_type=args.init_type)

    elif args.mode == "cov":
        pairfile = f"experiment/pairs/selected_pairs_{args.pair_count}.json"
        data_path = "data/tasic-pca50.npy"
        output_plot = f"experiment/plots/cov_plot_{args.pair_count}.png"
        run_cov_analysis(seeds=args.seeds, decoder_counts=[1, 2, 3], pairfile=pairfile, model_dir="experiment", data_path=data_path, output_plot=output_plot)

if __name__ == "__main__":
    main()
