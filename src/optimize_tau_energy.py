# optimize_tau_energy.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import random
import os
import numpy as np
from glob import glob
from pathlib import Path

from src.vae import VAE
from src.catmull_init_spline import CatmullRom





SEED = 12
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)




# Load trained decoder
def load_trained_decoder(path, input_dim=50, latent_dim=2, device="cpu"):
    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.decoder

# Load spline from previous init
def load_spline(path, device="cpu"):
    obj = torch.load(path, map_location=device)
    knots = obj["knots"].to(device)
    spline = CatmullRom(knots).to(device)
    spline.tau.data = obj["tau"].to(device)
    return spline

# help for jacobian
def compute_jacobian(decoder, z):
    z = z.detach().clone().requires_grad_(True)
    x = decoder(z).mean.view(-1)  # (output_dim,)
    J_rows = []
    for i in range(x.shape[0]):
        grad_i = torch.autograd.grad(x[i], z, retain_graph=True, create_graph=True)[0]  # shape: (1, 2)
        J_rows.append(grad_i)
    J = torch.cat(J_rows, dim=0)  # (output_dim, 2)
    return J

# Energy = Sum of squared pixel distances between decoded outputs along the spline
def compute_energy(spline, decoder, t_vals=None):
    if t_vals is None:
        t_vals = torch.linspace(0, 1, 64, device=spline.p.device)
    z = spline(t_vals)                              # (N, 2)
    dz = (z[1:] - z[:-1]) * t_vals.shape[0]         # dz/dt scaled
    dz = dz.unsqueeze(1)                            # (N-1, 1, 2)

    G_all = []
    for zi in z[:-1]:
        # zi = zi.detach().unsqueeze(0).requires_grad_(True)
        # x = decoder(zi).mean                        # (1, output_dim)
        J = compute_jacobian(decoder, zi.unsqueeze(0))  # (output_dim, 2)
        # print(f"Jacobian norm: {J.norm().item():.3e}")
        G = J.T @ J                                 # (2, 2)
        G_all.append(G.unsqueeze(0))
    G_all = torch.cat(G_all, dim=0)                 # (N-1, 2, 2)

    energy = torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2))  # (N-1, 1, 1)
    return energy.mean()


# 4. Optimize tau using gradient descent
def optimize_tau(spline, decoder, steps=1000, lr=1e-2):
    optimizer = optim.Adam([spline.tau], lr=lr)

    energy = compute_energy(spline, decoder, t_vals)
    best_energy = energy.item()
    print(f"Initial Energy: {best_energy:.4f}")

    for step in range(steps):
        tau_backup = spline.tau.data.clone()

        optimizer.zero_grad()
        energy = compute_energy(spline, decoder, t_vals)
        energy.backward()
        optimizer.step()

        new_energy = energy.detach().item()

        if new_energy > best_energy:
            spline.tau.data.copy_(tau_backup)
        else:
            best_energy = new_energy

        if step % 50 == 0:
            change = "↓" if new_energy <= best_energy else "↑"
            print(f"Step {step:4d}: Energy = {new_energy:.2f} {change} | τ grad norm = {spline.tau.grad.norm():.4f}")

    return spline



# 5. Plot optimized spline
def plot_spline(spline, out_path="src/plots/spline_tau_optimized.png"):
    with torch.no_grad():
        t = torch.linspace(0, 1, 400, device=spline.p.device)
        curve = spline(t).cpu().numpy()
        knots = spline.p.cpu().numpy()
        plt.figure(figsize=(4,4))
        plt.plot(curve[:,0], curve[:,1], 'r-', label="Optimized Spline", lw=2)
        plt.scatter(knots[:,0], knots[:,1], c='b', s=14, label='Knots')
        plt.axis("equal")
        plt.tight_layout()
        plt.legend()
        Path(out_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_path, dpi=300)
        print("Saved:", out_path)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_vals = torch.linspace(0, 1, 64, device=device)

    decoder_path = "src/artifacts/vae_best_avae.pth"
    decoder = load_trained_decoder(decoder_path, input_dim=50, latent_dim=2, device=device)

    spline_files = sorted(glob("src/artifacts/spline_inits/*.pt"))

    plt.figure(figsize=(6,6))

    for path in spline_files[:1]:
        data = torch.load(path, map_location=device)
        spline = CatmullRom(data["knots"].to(device)).to(device)
        spline.tau.data = data["tau"].to(device)

        # Save initial spline for plotting
        with torch.no_grad():
            t = torch.linspace(0, 1, 400, device=device)
            init_curve = spline(t).cpu().numpy()
            plt.plot(init_curve[:, 0], init_curve[:, 1], 'gray', alpha=0.4, linestyle="--", label="Initial Spline")

        # Optimize spline
        spline = optimize_tau(spline, decoder, steps=10, lr=1e-2)

        # Save optimized tau
        outname = Path(path).name.replace("spline_init", "spline_optimized")
        torch.save({'knots': spline.p, 'tau': spline.tau.detach()}, f"src/artifacts/{outname}")

        # Save optimized curve for plotting
        with torch.no_grad():
            optimized_curve = spline(t).cpu().numpy()
            plt.plot(optimized_curve[:, 0], optimized_curve[:, 1], 'r', lw=1, label="Optimized Spline")

    plt.axis("equal")
    plt.title("All Initial and Optimized Splines")
    plt.tight_layout()
    out_path = "src/plots/all_splines_overlay.png"
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=300)
    print("Saved plot:", out_path)
