import torch
import matplotlib.pyplot as plt
import numpy as np
from src.vae_new import VAE
from src.optimize_energy import GeodesicSpline  # reuse class

# Load spline data
ckpt = torch.load("src/artifacts/spline_ab.pt")
omega = ckpt["omega"]
a = ckpt["a"]
b = ckpt["b"]
basis = ckpt["basis"]
n_poly = ckpt["n_poly"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Rebuild spline
spline = GeodesicSpline((a.to(device), b.to(device)), basis.to(device), n_poly).to(device)
spline.omega.data.copy_(omega.to(device))

# Sample t
t = torch.linspace(0, 1, 2000, device=device, requires_grad=True)
z = spline(t)

# Derivatives
dz = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True)[0]
ddz = torch.autograd.grad(dz, t, grad_outputs=torch.ones_like(dz), create_graph=True)[0]
print("Spline shape:", z.shape)
print("Velocity shape:", dz.shape)
print("Acceleration shape:", ddz.shape)

# Detach for plotting
z_np = z.detach().cpu().numpy()
dz_np = dz.detach().cpu().numpy()
ddz_np = ddz.detach().cpu().numpy()

# Plot
fig, axs = plt.subplots(3, 1, figsize=(6, 12))

axs[0].plot(z_np[:, 0], z_np[:, 1], 'r-', lw=2, label='Geodesic Spline')
axs[0].scatter([a[0], b[0]], [a[1], b[1]], c='black', label='Endpoints')
axs[0].set_title("Spline in Latent Space")
axs[0].axis("equal")
axs[0].legend()

axs[1].plot(dz_np[:, 0], label="dx/dt")
axs[1].plot(dz_np[:, 1], label="dy/dt")
axs[1].set_title("Velocity (First Derivative)")
axs[1].legend()

axs[2].plot(ddz_np[:, 0], label="d²x/dt²")
axs[2].plot(ddz_np[:, 1], label="d²y/dt²")
axs[2].set_title("Acceleration (Second Derivative)")
axs[2].legend()

plt.tight_layout()
plt.savefig("src/plots/spline_loaded.png", dpi=300)
print("Saved plot to src/plots/spline_loaded.png")
