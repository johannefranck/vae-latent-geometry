import torch
import numpy as np
from pathlib import Path
from src.vae import EVAE

def load_model(num_decoders, rerun, input_dim=50, latent_dim=2, model_root="models_v102", device="cpu"):
    model = EVAE(input_dim=input_dim, latent_dim=latent_dim, num_decoders=num_decoders).to(device)
    model_path = Path(model_root) / f"dec{num_decoders}" / f"model_rerun{rerun}.pt"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, state_dict  # return both

# Load dataset
data = np.load("data/tasic-pca50.npy")
data_tensor = torch.tensor(data, dtype=torch.float32)

# Pick one pair of points
idx_a, idx_b = 3, 42
xA = data_tensor[idx_a].unsqueeze(0)
xB = data_tensor[idx_b].unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xA = xA.to(device)
xB = xB.to(device)

# Check encoder weights consistency
print("=== Encoder Weight Check Across Decoder Counts ===")
key = "encoder.encoder_net.0.weight"
for rerun in range(3):
    w_ref = None
    print(f"\nRerun {rerun}")
    for d in [1, 2, 3]:
        _, state = load_model(d, rerun, input_dim=50, latent_dim=2, model_root="models_v101", device=device)
        w = state[key]
        print(f"dec={d} | norm={w.norm():.6f}", end="")
        if w_ref is not None:
            print(f" | Δ={torch.norm(w - w_ref):.6e}")
        else:
            w_ref = w.clone()
            print(" | ref")

# Then do the CoV stability check as usual
print("\n=== Euclidean Distance CoV Check ===")
for d in [1, 2, 3]:
    diffs = []
    for rerun in range(3):
        model, _ = load_model(d, rerun, input_dim=50, latent_dim=2, model_root="models_v101", device=device)
        with torch.no_grad():
            zA = model.encoder(xA).mean.squeeze(0)
            zB = model.encoder(xB).mean.squeeze(0)
        dist = (zA - zB).norm().item()
        diffs.append(dist)
    diffs = np.array(diffs)
    print(f"[decoders={d}] mean={diffs.mean():.4f} | std={diffs.std():.4f} | CoV={diffs.std()/diffs.mean():.4f} | values={diffs}")
