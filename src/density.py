import torch
import numpy as np
from pathlib import Path
from src.vae_good import VAE
from src.plotting import plot_latent_density_with_splines
from src.optimize_energy import GeodesicSpline


# ---- CONFIG ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_DIR = Path("src/artifacts")
PLOT_DIR = Path("src/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = PLOT_DIR / "density_with_splines.png"

# ---- LOAD LATENTS + LABELS ----
latents = np.load(ARTIFACT_DIR / "latents_VAE_ld2_ep100_bs64_lr1e-03.npy")
labels = np.load("data/tasic-ttypes.npy")

# ---- LOAD VAE + DECODER ----
vae = VAE(input_dim=50, latent_dim=2).to(device)
vae.load_state_dict(torch.load(ARTIFACT_DIR / "vae_best_avae.pth", map_location=device))
vae.eval()
decoder = vae.decoder

# ---- LOAD SPLINE OBJECT ----
spline_data = torch.load(ARTIFACT_DIR / "spline_pair.pt", map_location=device)
a = spline_data["a"].to(device)
b = spline_data["b"].to(device)
n_poly = spline_data["n_poly"]
basis = spline_data["basis"].to(device)
omega_init = spline_data["omega_init"].to(device)
omega_opt = spline_data["omega_optimized"].to(device)
pair = (a, b)

spline_init = GeodesicSpline(pair, basis, n_poly).to(device)
spline_init.omega.data.copy_(omega_init)

spline_opt = GeodesicSpline(pair, basis, n_poly).to(device)
spline_opt.omega.data.copy_(omega_opt)

# ---- PLOT ----
plot_latent_density_with_splines(
    latents=latents,
    labels=labels,
    splines=[spline_init, spline_opt],
    filename=str(PLOT_PATH)
)

print(f"Saved: {PLOT_PATH}")
