import json
import numpy as np
import torch
from itertools import combinations
from pathlib import Path
import argparse
import os

from src.train import (
    EVAE,
    GaussianPrior, GaussianEncoder, GaussianDecoder,
    make_encoder_net, make_decoder_net
)
# from src.vae import 

def extract_latents(model, data, device):
    model.eval()
    with torch.no_grad():
        latents = model.encoder(data.to(device)).base_dist.loc
    return latents.cpu().numpy()

def select_representatives(latents, labels, max_labels=10):
    unique_labels = np.unique(labels)
    selected_labels = unique_labels[:max_labels]
    if len(selected_labels) < max_labels:
        print(f"Warning: Only {len(selected_labels)} unique labels found, expected {max_labels}.")

    representatives = []
    for lbl in selected_labels:
        inds = np.where(labels == lbl)[0]
        cluster = latents[inds]
        center = cluster.mean(axis=0)
        closest_idx = inds[np.argmin(np.linalg.norm(cluster - center, axis=1))]
        representatives.append({"index": int(closest_idx), "label": str(lbl)})
    return representatives

def save_pairs(representatives, path):
    indices = [r["index"] for r in representatives]
    pairs = list(combinations(indices, 2))
    data = {"representatives": representatives, "pairs": pairs}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(pairs)} pairs from {len(representatives)} representatives to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["vae", "evae"], required=True, help="Type of model: vae or evae")
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--num-decoders", type=int, default=10, help="Only used for EVAE")
    parser.add_argument("--max-labels", type=int, default=10)
    parser.add_argument("--data-path", type=str, default="data/tasic-pca50.npy")
    parser.add_argument("--label-path", type=str, default="data/tasic-ttypes.npy")
    parser.add_argument("--vae-latent-path", type=str, default="src/artifacts/latents_VAE_ld2_ep100_bs64_lr1e-03_seed12.npy", help="Only used for single VAE")
    parser.add_argument("--model-path", type=str, default="experiment/model_seed12.pt")
    parser.add_argument("--output-path", type=str, default="experiment/pairs/selected_pairs_10.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = np.load(args.label_path)
    data = torch.from_numpy(np.load(args.data_path).astype(np.float32)).to(device)

    if args.model_type == "vae":
        print("[INFO] Using precomputed VAE latents from .npy file")
        latents = np.load(args.vae_latent_path)

    elif args.model_type == "evae":
        print("[INFO] Using EVAE model to extract latents via encoder")
        prior = GaussianPrior(args.latent_dim)
        encoder = GaussianEncoder(make_encoder_net(data.shape[1], args.latent_dim))
        decoder = GaussianDecoder(make_decoder_net(args.latent_dim, data.shape[1]))
        model = EVAE(prior, encoder, decoder, num_decoders=args.num_decoders).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        latents = extract_latents(model, data, device)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Representative selection and saving
    reps = select_representatives(latents, labels, max_labels=args.max_labels)
    save_pairs(reps, Path(args.output_path))
