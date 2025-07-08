import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_lengths(path):
    data = torch.load(path, map_location="cpu")
    geo = {}
    euc = {}
    for d in data:
        pair = tuple(sorted(d["cluster_pair"]))
        geo.setdefault(pair, []).append(d["length_geodesic"])
        euc.setdefault(pair, []).append(d["length_euclidean"])
    return geo, euc

def compute_cov(lengths_dict):
    arr = np.array(lengths_dict)
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    covs = std / (mean + 1e-8)
    return float(covs.mean())

def main(pairfile, encoder_seed, reruns, max_decoders):
    pair_tag = Path(pairfile).stem.replace("selected_pairs_", "")
    geo_results = {}
    euc_results = {}

    for d in range(1, max_decoders + 1):
        print(f"\n[INFO] Evaluating ensemble size: {d} decoder(s)")
        all_geo = {}
        all_euc = {}

        valid_pairs = None

        for rerun in reruns:
            for decoder_idx in range(d):  # Use first `d` decoders
                fname = f"src/artifacts/spline_ensemble_optimized_seed{encoder_seed}_p{pair_tag}_rerun{rerun}_dec{decoder_idx}.pt"
                if not Path(fname).exists():
                    print(f"[WARN] Missing: {fname}")
                    continue
                geo_dict, euc_dict = load_lengths(fname)

                if valid_pairs is None:
                    valid_pairs = set(geo_dict.keys())
                else:
                    valid_pairs &= set(geo_dict.keys())

                for pair in geo_dict:
                    all_geo.setdefault(pair, []).extend(geo_dict[pair])
                for pair in euc_dict:
                    all_euc.setdefault(pair, []).extend(euc_dict[pair])

        if not valid_pairs:
            print(f"[WARN] No common pairs for d={d}")
            continue

        expected = len(reruns) * d
        print(f"[DEBUG] d={d} | Valid pairs: {len(valid_pairs)} | Expected per pair: {expected}")

        filtered_geo = [all_geo[p] for p in valid_pairs if len(all_geo[p]) == expected]
        filtered_euc = [all_euc[p] for p in valid_pairs if len(all_euc[p]) == expected]

        if not filtered_geo or not filtered_euc:
            print(f"[WARN] Skipping d={d} due to incomplete data.")
            continue

        print("[SANITY] Example Euclidean lengths for a pair:")
        for k, v in all_euc.items():
            if len(v) == expected:
                print(f"{k}: {v}")
                break

        cov_geo = compute_cov(filtered_geo)
        cov_euc = compute_cov(filtered_euc)

        geo_results[d] = cov_geo
        euc_results[d] = cov_euc
        print(f"[OK] d={d} | CoV geo: {cov_geo:.4f} | CoV euc: {cov_euc:.4f}")

    # Save results
    out_json = f"cov_results_seed{encoder_seed}_reruns{'-'.join(map(str, reruns))}_p{pair_tag}.json"
    with open(out_json, "w") as f:
        json.dump({"geodesic": geo_results, "euclidean": euc_results}, f, indent=2)
    print(f"[INFO] Saved: {out_json}")

    # Plot
    decoders = sorted(geo_results)
    geo_vals = [geo_results[d] for d in decoders]
    euc_vals = [euc_results[d] for d in decoders]

    plt.figure(figsize=(8, 6))
    plt.plot(decoders, euc_vals, label="Euclidean distance", marker='o')
    plt.plot(decoders, geo_vals, label="Geodesic distance", marker='o')
    plt.xlabel("Number of Decoders")
    plt.ylabel("Coefficient of Variation")
    plt.title(f"CoV vs. Number of Decoders (Seed={encoder_seed}, Reruns={len(reruns)}, {pair_tag})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_plot = f"cov_plot_seed{encoder_seed}_reruns{'-'.join(map(str, reruns))}_p{pair_tag}.png"
    plt.savefig(out_plot)
    print(f"[INFO] Saved plot: {out_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairfile", type=str, required=True)
    parser.add_argument("--encoder_seed", type=int, required=True)
    parser.add_argument("--reruns", nargs='+', type=int, required=True)
    parser.add_argument("--max_decoders", type=int, default=3)
    args = parser.parse_args()
    main(args.pairfile, args.encoder_seed, args.reruns, args.max_decoders)
