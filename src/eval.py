import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_geodesic_matrix(spline_blob, output_path):
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
    plt.title("Geodesic Distance Matrix")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Saved geodesic matrix plot to: {output_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["matrix"], required=True)
    parser.add_argument("--init-type", type=str, default="entropy", choices=["entropy", "euclidean"])
    parser.add_argument("--pair-count", type=int, default=10)
    parser.add_argument("--seed", type=int, required=True, help="Seed number, e.g., 123 for model_seed123.pt")
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
        plot_geodesic_matrix(spline_blob, plot_path)

if __name__ == "__main__":
    main()
