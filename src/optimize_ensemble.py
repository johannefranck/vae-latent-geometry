import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
from src.vae import VAE, EVAE
from src.single_decoder.optimize_energy import construct_nullspace_basis
from src.single_decoder.optimize_energy_batched import GeodesicSplineBatch

@torch.no_grad()
def compute_geodesic_lengths(spline, decoders, t_vals):
    z = spline(t_vals)  # (T, B, latent)
    outputs = [dec(z.view(-1, z.shape[-1])).mean for dec in decoders]
    x = torch.stack(outputs).mean(dim=0)  # (T*B, F)
    x = x.view(t_vals.shape[0], z.shape[1], -1)  # (T, B, F)
    diffs = x[1:] - x[:-1]
    return torch.norm(diffs, dim=2).sum(dim=0).cpu()  # (B,)

def compute_energy_ensemble(spline, decoders, t_vals):
    z = spline(t_vals)  # (T, B, latent)
    outputs = [decoder(z.view(-1, z.shape[-1])).mean for decoder in decoders]  # List of (T*B, F)
    x = torch.stack(outputs).mean(dim=0)  # Average: (T*B, F)
    x = x.view(t_vals.shape[0], z.shape[1], -1)  # (T, B, F)
    diffs = x[1:] - x[:-1]  # (T-1, B, F)
    energy = (diffs ** 2).sum(dim=2).sum(dim=0)  # (B,)
    return energy

def main(seed, pairfile, num_decoders, batch_size=5000):
    pair_tag = Path(pairfile).stem.replace("selected_pairs_", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_decoders == 1:
        suffix = f"VAE_ld2_d1_ep100_bs64_lr1e-03_seed{seed}"
        model = VAE(input_dim=50, latent_dim=2).to(device)
        decoders = [model.decoder]  # wrap single decoder in a list for consistency
    else:
        suffix = f"EVAE_ld2_d{num_decoders}_ep100_bs64_lr1e-03_seed{seed}"
        model = EVAE(input_dim=50, latent_dim=2, num_decoders=num_decoders).to(device)
        decoders = model.decoders

    spline_path = f"src/artifacts/spline_ensemble_seed{seed}_p{pair_tag}_d{num_decoders}.pt"
    model_path = f"src/artifacts/{suffix}_best.pth"
    output_path = f"src/artifacts/spline_ensemble_optimized_seed{seed}_p{pair_tag}_d{num_decoders}.pt"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    spline_data = torch.load(spline_path, map_location=device)["spline_data"]
    n_poly = spline_data[0]["n_poly"]
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)
    t_vals = torch.linspace(0, 1, 2000, device=device)

    all_outputs = []
    for start in range(0, len(spline_data), batch_size):
        end = min(start + batch_size, len(spline_data))
        chunk = spline_data[start:end]
        print(f"Optimizing splines {start} to {end - 1}")

        a = torch.stack([d["a"] for d in chunk]).to(device)
        b = torch.stack([d["b"] for d in chunk]).to(device)
        omega = torch.stack([d["omega_init"] for d in chunk]).to(device)
        cluster_pairs = [(d["a_label"], d["b_label"]) for d in chunk]

        model_spline = GeodesicSplineBatch(a, b, basis, omega, n_poly).to(device)
        optimizer = optim.Adam([model_spline.omega], lr=1e-3)

        for step in range(500):
            optimizer.zero_grad()
            energy = compute_energy_ensemble(model_spline, decoders, t_vals)
            endpoint_error = (model_spline(t_vals[-1:]) - b[None]) ** 2
            loss = energy + 1000 * endpoint_error.sum(dim=(0, 2))
            loss.sum().backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"Step {step}")

        lengths = compute_geodesic_lengths(model_spline, decoders, t_vals)

        for i in range(len(chunk)):
            all_outputs.append({
                "a": a[i].cpu(),
                "b": b[i].cpu(),
                "cluster_pair": cluster_pairs[i],
                "n_poly": n_poly,
                "basis": basis.cpu(),
                "omega_init": omega[i].cpu(),
                "omega_optimized": model_spline.omega.data[i].cpu(),
                "length_geodesic": lengths[i].item(),
                "length_euclidean": torch.norm(a[i] - b[i]).item()
            })

        del a, b, omega, model_spline
        torch.cuda.empty_cache()

    torch.save(all_outputs, output_path)
    print(f"[INFO] Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairfile", type=str, required=True)
    parser.add_argument("--num_decoders", type=int, required=True)
    args = parser.parse_args()
    main(args.seed, args.pairfile, args.num_decoders)
