import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from src.train import GaussianPrior, GaussianEncoder, GaussianDecoder, EVAE, make_decoder_net, make_encoder_net
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True) #eg experiment/model_seed12.pt
    parser.add_argument("--data-path", type=str, default="data/tasic-pca50.npy")
    parser.add_argument("--color-path", type=str, default="data/tasic-colors.npy")
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--num-decoders", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="experiment/plots/latent_plot_uncertainty.png")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resolution", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    input_dim = 50
    encoder_net = make_encoder_net(input_dim, args.latent_dim)
    decoder_net = make_decoder_net(args.latent_dim, input_dim)

    encoder = GaussianEncoder(encoder_net)
    decoder = GaussianDecoder(decoder_net)
    prior = GaussianPrior(args.latent_dim)
    model = EVAE(prior, encoder, decoder, num_decoders=args.num_decoders).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # extract the seed number from model_path
    seed = args.model_path.split("seed")[-1].split(".")[0]
    print(f"Loaded model with seed: {seed}")

    # Load data
    data = np.load(args.data_path).astype(np.float32)
    colors = np.load(args.color_path, allow_pickle=True)
    data_tensor = torch.from_numpy(data).to(device)

    with torch.no_grad():
        q = model.encoder(data_tensor)
        latents = q.base_dist.loc.cpu().numpy()  # (N, 2)

    # --------- Uncertainty background over latent space ---------
    resolution = args.resolution
    padding = 0.5

    z1_min, z1_max = latents[:, 0].min(), latents[:, 0].max()
    z2_min, z2_max = latents[:, 1].min(), latents[:, 1].max()

    # Make ranges square
    z1_center = (z1_min + z1_max) / 2
    z2_center = (z2_min + z2_max) / 2
    range_half = max((z1_max - z1_min), (z2_max - z2_min)) / 2 + padding

    z1_min, z1_max = z1_center - range_half, z1_center + range_half
    z2_min, z2_max = z2_center - range_half, z2_center + range_half

    zs1 = torch.linspace(z1_min, z1_max, resolution)
    zs2 = torch.linspace(z2_min, z2_max, resolution)
    Z1, Z2 = torch.meshgrid(zs1, zs2, indexing="xy")
    grid_latents = torch.stack([Z1.reshape(-1), Z2.reshape(-1)], dim=-1).to(device)

    with torch.no_grad():
        reconstructions = []
        for decoder in model.decoder:
            samples = decoder(grid_latents).mean
            reconstructions.append(samples)

        decoded_stack = torch.stack(reconstructions, dim=0)  # (num_decoders, N, D)
        std_across_decoders = decoded_stack.std(dim=0)       # (N, D)
        uncertainty = std_across_decoders.mean(dim=1)        # (N,)
        uncertainty_2d = uncertainty.view(resolution, resolution).cpu().numpy()

    # --------- Plot ---------
    fig, ax = plt.subplots(figsize=(7, 7))  # Square figure

    # Create an axis divider for aligned colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Background heatmap
    c = ax.pcolormesh(Z1.cpu().numpy(), Z2.cpu().numpy(),
                    uncertainty_2d, cmap="viridis", shading="auto", rasterized=True)

    # Colorbar that matches plot height
    plt.colorbar(c, cax=cax, label="Decoder uncertainty (std)")

    # Overlay latents
    ax.scatter(latents[:, 0], latents[:, 1], c=colors, s=5, alpha=0.8, linewidth=0)

    # Formatting
    ax.set_xlim(z1_min, z1_max)
    ax.set_ylim(z2_min, z2_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_title(f"Latent space ensmble VAE (seed {seed})")

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    print(f"Saved latent plot with uncertainty to: {args.save_path}")

if __name__ == "__main__":
    main()
