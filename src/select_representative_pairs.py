import json
import os
import numpy as np
from itertools import combinations

def select_representatives(latents, labels, max_labels=133):
    unique_labels = np.unique(labels)
    selected_labels = unique_labels[:max_labels]

    representatives = []
    for lbl in selected_labels:
        inds = np.where(labels == lbl)[0]
        cluster = latents[inds]
        center = cluster.mean(axis=0)
        closest_idx = inds[np.argmin(np.linalg.norm(cluster - center, axis=1))]

        representatives.append({
            "index": int(closest_idx),
            "label": str(lbl)
        })

    return representatives

def save_pairs(representatives, path="src/artifacts/selected_pairs.json"):
    indices = [r["index"] for r in representatives]
    pairs = list(combinations(indices, 2))

    data = {
        "representatives": representatives,
        "pairs": pairs
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(pairs)} pairs from {len(representatives)} representatives to {path}")

def load_pairs(path="src/artifacts/selected_pairs.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data["representatives"], data["pairs"]

if __name__ == "__main__":
    latent_path = "src/artifacts/latents_VAE_ld2_ep100_bs64_lr1e-03.npy"
    label_path = "data/tasic-ttypes.npy"

    latents = np.load(latent_path)
    labels = np.load(label_path)

    reps = select_representatives(latents, labels, max_labels=5)
    save_pairs(reps)