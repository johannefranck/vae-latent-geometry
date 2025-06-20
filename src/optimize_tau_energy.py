# optimize_tau_energy.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np

from src.advanced_vae import AdvancedVAE
from src.catmull_init_spline import CatmullRom



import random
import os

SEED = 12
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)




# 1. Load trained decoder
def load_trained_decoder(path, input_dim=50, latent_dim=2, device="cpu"):
    model = AdvancedVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.decoder

# 2. Load spline from previous init
def load_spline(path, device="cpu"):
    obj = torch.load(path, map_location=device)
    knots = obj["knots"].to(device)
    spline = CatmullRom(knots).to(device)
    spline.tau.data = obj["tau"].to(device)
    return spline


def compute_jacobian(decoder, z):
    z = z.detach().clone().requires_grad_(True)
    x = decoder(z).mean.view(-1)  # shape: (output_dim,)
    J_rows = []
    for i in range(x.shape[0]):
        grad_i = torch.autograd.grad(x[i], z, retain_graph=True, create_graph=True)[0]  # shape: (1, 2)
        J_rows.append(grad_i)
    J = torch.cat(J_rows, dim=0)  # shape: (output_dim, 2)
    return J





# 3. Energy = Sum of squared pixel distances between decoded outputs along the spline
def compute_energy(spline, decoder, t_vals=None):
    if t_vals is None:
        t_vals = torch.linspace(0, 1, 64, device=spline.p.device)
    z = spline(t_vals)                              # (N, 2)
    dz = (z[1:] - z[:-1]) * t_vals.shape[0]         # ∂z/∂t scaled
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


# def compute_energy_batched(spline, decoder, t_vals=None):
#     if t_vals is None:
#         t_vals = torch.linspace(0, 1, 64, device=spline.p.device)

#     z = spline(t_vals)                                # (N, 2)
#     dz = (z[1:] - z[:-1]) * t_vals.shape[0]           # (N-1, 2)
#     dz = dz.unsqueeze(1)                              # (N-1, 1, 2)
#     z_in = z[:-1].detach().clone().requires_grad_(True)  # (N-1, 2)

#     # Compute Jacobians per sample (looped, avoids cross-sample gradients)
#     J_list = []
#     for i in range(z_in.shape[0]):
#         zi = z_in[i].unsqueeze(0)                     # (1, 2)
#         zi.requires_grad_(True)
#         xi = decoder(zi).mean.view(-1)                # (D,)
#         grads = []
#         for j in range(xi.shape[0]):
#             grad_j = torch.autograd.grad(xi[j], zi, retain_graph=True, create_graph=True)[0]  # (1, 2)
#             grads.append(grad_j)
#         Ji = torch.cat(grads, dim=0).unsqueeze(0)     # (1, D, 2)
#         J_list.append(Ji)

#     J = torch.cat(J_list, dim=0)                      # (N-1, D, 2)
#     J = J.transpose(1, 2)                             # (N-1, 2, D)
#     G_all = torch.bmm(J, J.transpose(1, 2))           # (N-1, 2, 2)

#     energy = torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2))  # (N-1, 1, 1)
#     return energy.mean()








# 4. Main optimization loop
# def optimize_tau(spline, decoder, steps=1000, lr=1e-2):
#     optimizer = optim.Adam([spline.tau], lr=lr)
#     for step in range(steps):
#         optimizer.zero_grad()
#         E = compute_energy(spline, decoder, t_vals)
#         E.backward()
#         optimizer.step()
#         if step % 50 == 0:
#             print(f"tau grad norm: {spline.tau.grad.norm().item():.4f}")
#             print(f"Step {step}: Energy = {E.item():.4f}")
#     return spline
# def optimize_tau(spline, decoder, steps=1000, lr=1e-2):
#     optimizer = optim.Adam([spline.tau], lr=lr)
#     prev_energy = compute_energy(spline, decoder, t_vals).item()
#     print(f"Initial Energy: {prev_energy:.4f}")

#     for step in range(steps):
#         tau_backup = spline.tau.data.clone()  # backup current τ

#         optimizer.zero_grad()
#         E = compute_energy(spline, decoder, t_vals)
#         E.backward()
#         optimizer.step()

#         new_energy = compute_energy(spline, decoder, t_vals).item()

#         if new_energy > prev_energy:
#             spline.tau.data = tau_backup  # revert τ
#             for group in optimizer.param_groups:
#                 for p in group['params']:
#                     if p.grad is not None:
#                         p.grad.zero_()
#             if step % 50 == 0:
#                 print(f"Step {step}: Rejected ↑ Energy = {new_energy:.2f} > {prev_energy:.2f}")
#         else:
#             prev_energy = new_energy
#             if step % 50 == 0:
#                 print(f"Step {step}: Accepted ↓ Energy = {new_energy:.2f} | τ grad norm: {spline.tau.grad.norm().item():.4f}")

#     return spline


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

# 6. Entrypoint
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     t_vals = torch.linspace(0, 1, 64, device=device)
#     decoder_path = "src/artifacts/vae_best_avae.pth"
#     spline_path  = "src/artifacts/spline_init_syrota.pt"

#     decoder = load_trained_decoder(decoder_path, input_dim=50, latent_dim=2, device=device)
#     spline = load_spline(spline_path, device)

#     spline = optimize_tau(spline, decoder, steps=100, lr=1e-2)
#     torch.save({'knots': spline.p, 'tau': spline.tau.detach()}, "src/artifacts/spline_optimized_tau.pt")
#     print("Saved optimized τ to src/artifacts/spline_optimized_tau.pt")

#     plot_spline(spline)

if __name__ == "__main__":
    from glob import glob

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
            plt.plot(init_curve[:, 0], init_curve[:, 1], 'gray', alpha=0.4, linestyle="--")

        # Optimize spline
        spline = optimize_tau(spline, decoder, steps=50, lr=1e-2)

        # Save optimized tau
        outname = Path(path).name.replace("spline_init", "spline_optimized")
        torch.save({'knots': spline.p, 'tau': spline.tau.detach()}, f"src/artifacts/{outname}")

        # Save optimized curve for plotting
        with torch.no_grad():
            optimized_curve = spline(t).cpu().numpy()
            plt.plot(optimized_curve[:, 0], optimized_curve[:, 1], 'r', lw=1)

    plt.axis("equal")
    plt.title("All Initial and Optimized Splines")
    plt.tight_layout()
    out_path = "src/plots/all_splines_overlay.png"
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=300)
    print("Saved plot:", out_path)
