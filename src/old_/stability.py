import torch
import numpy as np
from pathlib import Path
from src.vae import EVAE
from src.select_representative_pairs import load_pairs

# def load_model(num_decoders, rerun, input_dim=50, latent_dim=2, model_root="models_v102", device="cpu"):
#     model = EVAE(input_dim=input_dim, latent_dim=latent_dim, num_decoders=num_decoders).to(device)
#     model_path = Path(model_root) / f"dec{num_decoders}" / f"model_rerun{rerun}.pt"
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model, state_dict  # return both

# # Load dataset
# data = np.load("data/tasic-pca50.npy")
# data_tensor = torch.tensor(data, dtype=torch.float32)

# # Pick one pair of points
# idx_a, idx_b = 3, 42
# xA = data_tensor[idx_a].unsqueeze(0)
# xB = data_tensor[idx_b].unsqueeze(0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# xA = xA.to(device)
# xB = xB.to(device)

# # Check encoder weights consistency
# print("=== Encoder Weight Check Across Decoder Counts ===")
# key = "encoder.encoder_net.0.weight"
# for rerun in range(3):
#     w_ref = None
#     print(f"\nRerun {rerun}")
#     for d in [1, 2, 3]:
#         _, state = load_model(d, rerun, input_dim=50, latent_dim=2, model_root="models_v101", device=device)
#         w = state[key]
#         print(f"dec={d} | norm={w.norm():.6f}", end="")
#         if w_ref is not None:
#             print(f" | Δ={torch.norm(w - w_ref):.6e}")
#         else:
#             w_ref = w.clone()
#             print(" | ref")

# # Then do the CoV stability check 
# print("\n=== Euclidean Distance CoV Check ===")
# for d in [1, 2, 3]:
#     diffs = []
#     for rerun in range(3):
#         model, _ = load_model(d, rerun, input_dim=50, latent_dim=2, model_root="models_v101", device=device)
#         with torch.no_grad():
#             zA = model.encoder(xA).mean.squeeze(0)
#             zB = model.encoder(xB).mean.squeeze(0)
#         dist = (zA - zB).norm().item()
#         diffs.append(dist)
#     diffs = np.array(diffs)
#     print(f"[decoders={d}] mean={diffs.mean():.4f} | std={diffs.std():.4f} | CoV={diffs.std()/diffs.mean():.4f} | values={diffs}")


# import torch
# from pathlib import Path

# MODEL_ROOT = "models_v103"
# DECODER_COUNTS = [1, 2, 3, 4, 5, 6]
# RERUNS = range(10)

# def load_encoder_weights(path, device="cpu"):
#     state = torch.load(path, map_location=device)
#     return {k: v for k, v in state.items() if "encoder" in k}

# def check_consistency():
#     for rerun in RERUNS:
#         print(f"\n=== RERUN {rerun} ===")
#         base_path = Path(MODEL_ROOT) / f"dec1" / f"model_rerun{rerun}.pt"
#         if not base_path.exists():
#             print(f"Missing base model for rerun {rerun}")
#             continue

#         ref_enc = load_encoder_weights(base_path)

#         for d in DECODER_COUNTS[1:]:
#             test_path = Path(MODEL_ROOT) / f"dec{d}" / f"model_rerun{rerun}.pt"
#             if not test_path.exists():
#                 print(f"  [SKIP] dec{d}/rerun{rerun} missing")
#                 continue

#             test_enc = load_encoder_weights(test_path)
#             mismatches = [k for k in ref_enc if not torch.equal(ref_enc[k], test_enc.get(k))]
#             if mismatches:
#                 print(f"  [X] Mismatch in dec{d}/rerun{rerun}: {mismatches}")
#             else:
#                 print(f"  [✓] dec{d}/rerun{rerun} matches")

# check_consistency()

pairs1 = load_pairs("src/artifacts/selected_pairs_50.json")[1]
pairs2 = load_pairs("src/artifacts/selected_pairs_50.json")[1]
assert pairs1 == pairs2 