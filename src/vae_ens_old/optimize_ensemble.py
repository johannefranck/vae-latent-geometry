import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
from src.vae import EVAE
from src.single_decoder.optimize_energy import construct_nullspace_basis
from src.single_decoder.optimize_energy_batched import GeodesicSplineBatch

@torch.no_grad()
def compute_geodesic_lengths(spline, decoder, t_vals):
    z = spline(t_vals)
    x = decoder(z.view(-1, z.shape[-1])).mean
    x = x.view(t_vals.shape[0], z.shape[1], -1)
    diffs = x[1:] - x[:-1]
    return torch.norm(diffs, dim=2).sum(dim=0).cpu()

def compute_energy_single(spline, decoder, t_vals):
    z = spline(t_vals)
    x = decoder(z.view(-1, z.shape[-1])).mean
    x = x.view(t_vals.shape[0], z.shape[1], -1)
    diffs = x[1:] - x[:-1]
    return (diffs ** 2).sum(dim=2).sum(dim=0)

def optimize_all(seed, pairfile, num_decoders, reruns, batch_size=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_tag = Path(pairfile).stem.replace("selected_pairs_", "")
    t_vals = torch.linspace(0, 1, 2000, device=device)

    for rerun in reruns:
        # Load model
        model = EVAE(input_dim=50, latent_dim=2, num_decoders=num_decoders).to(device)
        model_path = f"src/artifacts/ensemble/EVAE_dec{num_decoders}_rerun{rerun}_best.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        for decoder_idx in range(num_decoders):
            spline_path = f"src/artifacts/ensemble/spline_ensemble_seed{seed}_p{pair_tag}_rerun{rerun}_dec{decoder_idx}.pt"
            output_path = f"src/artifacts/ensemble/spline_ensemble_optimized_seed{seed}_p{pair_tag}_rerun{rerun}_dec{decoder_idx}.pt"

            print(f"\n[INFO] Optimizing splines for rerun={rerun}, decoder={decoder_idx}")

            spline_data = torch.load(spline_path, map_location=device)["spline_data"]
            if not spline_data:
                print(f"[WARN] Empty spline data for rerun={rerun}, decoder={decoder_idx}")
                continue

            n_poly = spline_data[0]["n_poly"]
            basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)
            decoder = model.decoders[decoder_idx]

            all_outputs = []
            for start in range(0, len(spline_data), batch_size):
                end = min(start + batch_size, len(spline_data))
                chunk = spline_data[start:end]

                a_batch = torch.stack([d["a"] for d in chunk]).to(device)
                b_batch = torch.stack([d["b"] for d in chunk]).to(device)
                omega_batch = torch.stack([d["omega_init"] for d in chunk]).to(device)
                cluster_pairs = [(d["a_label"], d["b_label"]) for d in chunk]

                model_spline = GeodesicSplineBatch(a_batch, b_batch, basis, omega_batch, n_poly).to(device)
                optimizer = optim.Adam([model_spline.omega], lr=1e-3)

                for step in range(1000):
                    optimizer.zero_grad()
                    energy = compute_energy_single(model_spline, decoder, t_vals)
                    endpoint_error = (model_spline(t_vals[-1:]) - b_batch[None]) ** 2
                    loss = energy + 1000 * endpoint_error.sum(dim=(0, 2))
                    loss.sum().backward()
                    optimizer.step()
                    if step % 100 == 0:
                        print(f"  Step {step} | Energy: {energy.mean().item():.4f}")

                lengths = compute_geodesic_lengths(model_spline, decoder, t_vals)
                for i in range(len(chunk)):
                    all_outputs.append({
                        "a": a_batch[i].cpu(),
                        "b": b_batch[i].cpu(),
                        "cluster_pair": cluster_pairs[i],
                        "decoder_idx": decoder_idx,
                        "n_poly": n_poly,
                        "basis": basis.cpu(),
                        "omega_init": omega_batch[i].cpu(),
                        "omega_optimized": model_spline.omega.data[i].cpu(),
                        "length_geodesic": lengths[i].item(),
                        "length_euclidean": torch.norm(a_batch[i] - b_batch[i]).item()
                    })

                torch.cuda.empty_cache()
                del model_spline

            torch.save(all_outputs, output_path)
            print(f"[OK] Saved: {output_path} | {len(all_outputs)} splines")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_seed", type=int, required=True)
    parser.add_argument("--pairfile", type=str, required=True)
    parser.add_argument("--num_decoders", type=int, required=True)
    parser.add_argument("--reruns", type=int, nargs="+", required=True)
    args = parser.parse_args()

    optimize_all(args.global_seed, args.pairfile, args.num_decoders, args.reruns)
