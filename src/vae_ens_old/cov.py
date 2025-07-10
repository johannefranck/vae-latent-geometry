import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def compute_cov_across_reruns(global_seed, pair_tag, decoder_counts, reruns, artifact_dir="src/artifacts/ensemble"):
    results = {"euclidean": {}, "geodesic": {}}

    for num_decoders in decoder_counts:
        euclidean_dists = defaultdict(list)
        geodesic_dists = defaultdict(list)

        for rerun in reruns:
            for decoder_idx in range(num_decoders):
                path = f"{artifact_dir}/spline_ensemble_optimized_seed{global_seed}_p{pair_tag}_rerun{rerun}_dec{decoder_idx}.pt"
                if not os.path.exists(path):
                    print(f"[WARN] Missing file: {path}")
                    continue

                entries = torch.load(path)
                for entry in entries:
                    try:
                        cluster_pair = tuple(entry["cluster_pair"])
                        a = tuple(entry["a"].tolist())
                        b = tuple(entry["b"].tolist())
                        key = (cluster_pair, a, b)

                        euclidean_dists[key].append(entry["length_euclidean"])
                        geodesic_dists[key].append(entry["length_geodesic"])
                    except KeyError as e:
                        print(f"[ERROR] Missing key in entry from {path}: {e}")
                        continue

        def compute_cov(dist_dict):
            covs = []
            for dists in dist_dict.values():
                if len(dists) < 2:
                    continue
                dists = np.array(dists)
                mean = dists.mean()
                std = dists.std()
                if mean > 0:
                    covs.append(std / mean)
            return float(np.mean(covs)) if covs else float("nan")

        results["euclidean"][num_decoders] = compute_cov(euclidean_dists)
        results["geodesic"][num_decoders] = compute_cov(geodesic_dists)

    return results


if __name__ == "__main__":
    global_seed = 12
    pair_tag = "10"
    decoder_counts = [1, 2, 3]
    reruns = list(range(10))
    artifact_dir = "src/artifacts/ensemble"

    results = compute_cov_across_reruns(global_seed, pair_tag, decoder_counts, reruns, artifact_dir)

    # Save as JSON
    json_path = os.path.join(artifact_dir, f"cov_results_seed{global_seed}_p{pair_tag}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved results to {json_path}")

    # Plot
    plt.figure()
    plt.plot(decoder_counts, [results["euclidean"][d] for d in decoder_counts], label="Euclidean distance")
    plt.plot(decoder_counts, [results["geodesic"][d] for d in decoder_counts], label="Geodesic distance")
    plt.xlabel("Number of decoders")
    plt.ylabel("Coefficient of Variation")
    plt.title("CoV vs Number of Decoders")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(artifact_dir, f"cov_plot_seed{global_seed}_p{pair_tag}.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"[OK] Saved plot to {plot_path}")
