import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_lengths(path):
    data = torch.load(path, map_location="cpu")  # this is a list of dicts
    lengths_geo = {}  # dict of: pair -> list of geodesic lengths
    lengths_euc = {}  # dict of: pair -> list of Euclidean lengths
    for d in data:
        pair = tuple(sorted(d["cluster_pair"]))
        lengths_geo.setdefault(pair, []).append(d["length_geodesic"])
        lengths_euc.setdefault(pair, []).append(d["length_euclidean"])
    return lengths_geo, lengths_euc

def compute_cov(lengths_dict, expected_count):
    filtered = [v for v in lengths_dict.values() if len(v) == expected_count]
    if not filtered:
        return None
    arr = np.array(filtered)
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    covs = std / (mean + 1e-8)
    return float(covs.mean())

def main(seed, pairfile, max_decoders=3):
    pair_tag = Path(pairfile).stem.replace("selected_pairs_", "")
    geo_results = {}
    euc_results = {}

    for d in range(1, max_decoders + 1):
        path = f"src/artifacts/spline_ensemble_optimized_seed{seed}_p{pair_tag}_d{d}.pt"
        if not Path(path).exists():
            print(f"[WARN] Missing file for d={d}: {path}")
            continue

        geo_dict, euc_dict = load_lengths(path)
        cov_geo = compute_cov(geo_dict, expected_count=d)
        cov_euc = compute_cov(euc_dict, expected_count=d)

        if cov_geo is None or cov_euc is None:
            print(f"[WARN] Skipping d={d}: no valid pairs")
            continue

        geo_results[d] = cov_geo
        euc_results[d] = cov_euc
        print(f"[INFO] d={d} | CoV geo: {cov_geo:.4f} | CoV euc: {cov_euc:.4f}")

    # Save results to file
    result_file = f"cov_joint_results_seed{seed}_p{pair_tag}.json"
    with open(result_file, "w") as f:
        json.dump({
            "geodesic": geo_results,
            "euclidean": euc_results
        }, f, indent=2)
    print(f"[INFO] Saved results to {result_file}")

    # Plot CoV vs. number of decoders
    decoders = sorted(geo_results.keys())
    geo_y = [geo_results[d] for d in decoders]
    euc_y = [euc_results[d] for d in decoders]

    plt.figure(figsize=(8, 6))
    plt.plot(decoders, euc_y, label="Euclidean distance", marker='o')
    plt.plot(decoders, geo_y, label="Geodesic distance", marker='o')
    plt.xlabel("Number of decoders")
    plt.ylabel("Coefficient of Variation")
    plt.title(f"CoV vs. number of decoders (Seed {seed}, {pair_tag})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"cov_vs_decoders_seed{seed}_p{pair_tag}.png")
    print(f"[INFO] Saved plot to cov_vs_decoders_seed{seed}_p{pair_tag}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairfile", type=str, required=True)
    parser.add_argument("--max_decoders", type=int, default=3)
    args = parser.parse_args()
    main(args.seed, args.pairfile, args.max_decoders)
