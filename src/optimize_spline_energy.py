import torch
import pickle
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from src.vae import VAE
from src.spline_pytorch import TorchCubicSpline, resample_path

# ---- Energy Function ----
def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)  # (T, 2)
    dz = (z[1:] - z[:-1]) * t_vals.shape[0]  # finite difference
    dz = dz.unsqueeze(1)  # (T-1, 1, 2)

    G_all = []
    for zi in z[:-1]:
        zi = zi.detach().requires_grad_(True)
        x = decoder(zi.unsqueeze(0))[0]
        J_rows = []
        for j in range(x.shape[0]):
            grad_j = torch.autograd.grad(x[j], zi, retain_graph=True, create_graph=True)[0]
            J_rows.append(grad_j)
        J = torch.stack(J_rows)  # (x_dim, 2)
        G = J.T @ J  # (2, 2)
        G_all.append(G.unsqueeze(0))

    G_all = torch.cat(G_all, dim=0)  # (T-1, 2, 2)
    energy = torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2)).mean()
    return energy

# ---- Optimization Loop ----
def optimize_splines(splines, decoder, steps=10, lr=1e-1):
    t_vals = torch.linspace(0, 1, 100)
    optimized = []

    for i, spline in enumerate(splines[:2]):
        mask = torch.ones_like(spline.ctrl_pts, dtype=torch.bool)
        mask[0] = False
        mask[-1] = False
        spline.train()
        optimizer = Adam([spline.ctrl_pts], lr=lr)

        print(f"\n--- Optimizing spline {i+1}/{len(splines[:2])} ---")
        for step in range(steps):
            optimizer.zero_grad()
            energy = compute_energy(spline, decoder, t_vals)
            energy.backward()
            with torch.no_grad():
                spline.ctrl_pts.grad[~mask] = 0.0
            optimizer.step()

            print(f"Step {step+1:3d} | Energy: {energy.item():.6f}")

        spline.eval()
        optimized.append(spline)
        print(f"Spline {i+1} optimized. Final energy: {energy.item():.4f}")

    return optimized

# ---- Plotting ----
def plot_optimized_paths(latents, labels, grid, splines, filename):
    import seaborn as sns
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.5, legend=False)
    plt.scatter(grid[:, 0], grid[:, 1], s=5, alpha=0.2, color="lightblue")

    t_vals = torch.linspace(0, 1, 200)
    for spline in splines:
        with torch.no_grad():
            pts = spline(t_vals).cpu().numpy()
        plt.plot(pts[:, 0], pts[:, 1], '-', color='red', alpha=0.8)

    plt.title("Optimized Geodesic Splines")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ---- Main ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained decoder
    decoder = VAE(input_dim=50, latent_dim=2).decoder
    decoder.load_state_dict(torch.load("src/artifacts/decoder_ld2_ep600_bs64_lr1e-03.pth", map_location=device))
    decoder.eval().to(device)

    # Load splines
    with open("src/artifacts/spline_inits_torch.pkl", "rb") as f:
        splines = pickle.load(f)
    for spline in splines[:2]:
        spline.to(device)

    # Optimize
    optimized_splines = optimize_splines(splines, decoder, steps=10, lr=1e-1)

    # Save
    with open("src/artifacts/spline_optimized_torch.pkl", "wb") as f:
        pickle.dump(optimized_splines, f)

    # Optional plot
    latents = np.load("src/artifacts/latents_ld2_ep600_bs64_lr1e-03.npy")
    labels = np.load("data/tasic-ttypes.npy")
    grid = np.load("src/artifacts/grid.npy")
    plot_optimized_paths(latents, labels, grid, optimized_splines, "src/plots/latents_optimized_splines.png")

if __name__ == "__main__":
    main()
