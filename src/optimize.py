import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

from src.train import EVAE, GaussianEncoder, GaussianDecoder, GaussianPrior, make_encoder_net, make_decoder_net
from src.single_decoder.optimize_energy import construct_nullspace_basis


class GeodesicSplineBatch(nn.Module):
    def __init__(self, a, b, basis, omega, n_poly):
        super().__init__()
        self.a = a
        self.b = b
        self.basis = basis
        self.omega = nn.Parameter(omega)
        self.n_poly = n_poly

    def forward(self, t):
        B, K, D = self.omega.shape
        coeffs = torch.einsum("nk,bkd->nbd", self.basis, self.omega)
        coeffs = coeffs.view(self.n_poly, 4, B, D)

        seg_idx = torch.clamp((t * self.n_poly).floor().long(), max=self.n_poly - 1)
        local_t = t * self.n_poly - seg_idx.float()
        powers = torch.stack([local_t ** i for i in range(4)], dim=1).to(t.device)

        coeffs_selected = coeffs[seg_idx]
        poly = torch.einsum("ti,tibd->tbd", powers, coeffs_selected)

        linear = (1 - t[:, None, None]) * self.a[None] + t[:, None, None] * self.b[None]
        return linear + poly


def compute_energy_mc(model, decoders, t_vals, M=2):
    """
    Compute the MC-estimated energy for each spline in the batch.
    Each segment uses independently sampled decoder pairs.
    """
    T, B, D = t_vals.shape[0], model.a.shape[0], model.a.shape[1]
    z = model(t_vals)  # (T, B, D)
    M_dec = len(decoders)

    # decoded_z = torch.stack([d(z).rsample() for d in decoders], dim=0)  # (M_dec, T, B, X)

    total_energy = torch.zeros(B, device=z.device)
    decoded_z = torch.stack([d(z).mean for d in decoders], dim=0)
    energies_per_sample = []

    for _ in range(M):
        idx_t = torch.arange(T - 1)
        idx_b = torch.arange(B)

        d1_idx = torch.randint(0, M_dec, (T-1, B), device=z.device)
        d2_idx = torch.randint(0, M_dec, (T-1, B), device=z.device)

        x1 = decoded_z[d1_idx, idx_t[:, None], idx_b[None, :]]  # (T-1, B, X)
        x2 = decoded_z[d2_idx, idx_t[:, None] + 1, idx_b[None, :]]  # (T-1, B, X)

        dist_sq = ((x2 - x1) ** 2).sum(dim=2)  # (T-1, B)
        energy_sample = dist_sq.sum(dim=0)
        total_energy += dist_sq.sum(dim=0)

        energies_per_sample.append(energy_sample)

    # if M > 1:
    #     e_stack = torch.stack(energies_per_sample)  # (M, B)
    #     std = e_stack.std(dim=0).mean().item()
    #     mean = e_stack.mean(dim=0).mean().item()
    #     print(f"[MC E] Mean: {mean:.2f}, Std: {std:.2f}, CoV: {std / mean:.4f}")

    return total_energy / M




def main(model_path, spline_path, init_type, pair_count, steps=500, batch_size=200, M=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Infer spline_path if not given
    if spline_path is None:
        model_name = Path(model_path).stem
        spline_dir = Path("experiment") / f"splines_init_{model_name}"
        spline_pattern = f"spline_batch_init_{init_type}_{pair_count}.pt"
        spline_path = spline_dir / spline_pattern
        if not spline_path.exists():
            raise FileNotFoundError(f"[ERROR] Expected spline file not found: {spline_path}")
        print(f"[INFO] Automatically using spline: {spline_path}")

    # === Load model ===
    latent_dim = 2
    input_dim = 50
    encoder = GaussianEncoder(make_encoder_net(input_dim, latent_dim))
    decoder = GaussianDecoder(make_decoder_net(latent_dim, input_dim))
    prior = GaussianPrior(latent_dim)
    model = EVAE(prior, encoder, decoder, num_decoders=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    decoders = list(model.decoder)
    print(f"[DEBUG] Loaded model: {model_path}")

    # === Check if decoders are identical ===
    with torch.no_grad():
        z_test = torch.randn(1, latent_dim, device=device)
        outputs = [dec(z_test).mean for dec in decoders]
        identical = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        print("[CHECK] All decoders identical?", identical)
        if identical:
            print("[WARNING] All decoders are returning the same output! Potential weight sharing or load issue.")
        else:
            print("Decoders are (at least slightly) diverse.")

    # Load original data once
    raw_data = np.load("data/tasic-pca50.npy")
    data_tensor = torch.tensor(raw_data, dtype=torch.float32).to(device)

    # === Load splines ===
    spline_blob = torch.load(spline_path, map_location=device)
    spline_data = spline_blob["spline_data"]
    n_poly = spline_data[0]["n_poly"]
    basis = spline_data[0]["basis"].to(device)
    _, K = basis.shape

    print(f"[INFO] Optimizing {len(spline_data)} splines (n_poly={n_poly}, K={K})")
    t_vals = torch.linspace(0, 1, 2000, device=device)

    # === Prepare output path ===
    model_name = Path(model_path).stem  # e.g., model_seed12
    spline_tag = Path(spline_path).stem.replace("spline_batch_init_", "")  # e.g., "entropy_133"
    init_dir = Path(spline_path).parent.name  # e.g., splines_init_model_seed12
    model_id = Path(model_path).stem  # e.g. model_seed12 or model_123
    opt_dir = f"splines_opt_{model_id}"
    save_dir = Path("experiment") / opt_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"spline_batch_opt_{spline_tag}.pt"

    # === Optimize splines in batches ===
    for start in tqdm(range(0, len(spline_data), batch_size), desc="Batched optimization"):
        end = min(start + batch_size, len(spline_data))
        print(f"[BATCH] Batch {start // batch_size + 1}/{(len(spline_data) - 1) // batch_size + 1}: optimizing splines [{start}:{end}]")
        chunk = spline_data[start:end]

        a = torch.stack([d["a"] for d in chunk]).to(device)
        b = torch.stack([d["b"] for d in chunk]).to(device)
        omega = torch.stack([d["omega_init"] for d in chunk]).to(device)

        model_batch = GeodesicSplineBatch(a, b, basis, omega.clone(), n_poly).to(device)
        optimizer = optim.Adam([model_batch.omega], lr=1e-3)

        for step in range(steps):
            optimizer.zero_grad()
            energy = compute_energy_mc(model_batch, decoders, t_vals, M=M)
            endpoint_error = (model_batch(t_vals[-1:]) - b[None]) ** 2
            endpoint_loss = endpoint_error.sum(dim=(0, 2))
            loss = energy + 1000 * endpoint_loss
            loss.sum().backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"[Step {step}] Mean Energy: {energy.mean():.4f}")

        omega_optimized = model_batch.omega.detach().cpu()
        lengths = torch.sqrt(energy).detach().cpu()  # Approximate geodesic lengths
        eucl_dists = []
        with torch.no_grad():
            for d in chunk:
                idx_a = d["a_index"]
                idx_b = d["b_index"]
                zA = model.encoder(data_tensor[idx_a:idx_a+1]).base_dist.loc.squeeze(0)
                zB = model.encoder(data_tensor[idx_b:idx_b+1]).base_dist.loc.squeeze(0)
                eucl_dists.append(torch.norm(zA - zB).item())
        # print("[DEBUG] Latents A/B:", zA.tolist(), zB.tolist())

        # print(lengths)
        # print(eucl_dists)

        for i, d in enumerate(chunk):
            d["omega_optimized"] = omega_optimized[i]
            d["geodesic_length"] = lengths[i].item()
            d["euclidean_distance"] = eucl_dists[i]


    # === Save ===
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "spline_data": spline_data,
        "representatives": spline_blob.get("representatives", None),
        "pairs": spline_blob.get("pairs", None),
        "metadata": {
        "model_name": model_name,
        "init_type": init_type,
        "pair_count": pair_count,
        "mc_samples": M,
        "steps": steps
    }
    }, save_path)

    print(f"[✓] Saved optimized splines to: {save_path}")

    from src.plotting import plot_initial_and_optimized_splines

    # Only load latents once
    data = np.load("data/tasic-pca50.npy")
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        latents = model.encoder(data_tensor).base_dist.loc.cpu().numpy()

    plot_initial_and_optimized_splines(
        spline_path=save_path,
        latents=latents,
        save_path=save_dir / f"spline_plot_both_{spline_tag}.png",
        device=device
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True) # experiment/model_seed12.pt
    parser.add_argument("--spline-path", type=str, default=None) # experiment/splines_opt_model_seedX/spline_batch_opt_entropy_133.pt
    parser.add_argument("--init-type", type=str, default="entropy", choices=["entropy", "euclidean"],
                    help="Choose which spline init to use if --spline-path is not specified.")
    parser.add_argument("--pair-count", type=int, required=True,
                    help="Number of points used to generate the spline pairs (e.g. 10, 133). Required if --spline-path is not specified.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200) # Number of splines to optimize in parallel
    parser.add_argument("--mc-samples", type=int, default=2)
    args = parser.parse_args()


    main(
        model_path=args.model_path,
        spline_path=args.spline_path,
        init_type=args.init_type,
        pair_count=args.pair_count,
        steps=args.steps,
        batch_size=args.batch_size,
        M=args.mc_samples
    )

    
