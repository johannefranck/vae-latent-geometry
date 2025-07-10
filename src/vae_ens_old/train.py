import os
import sys
import random
import torch
import yaml
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from src.vae import VAE, EVAE


def main(config):
    # ----- CONFIG -----
    SEED = config["training"]["seed"]
    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["training"]["batch_size"]
    LR = float(config["training"]["lr"])
    INPUT_DIM = config["vae"]["input_dim"]
    LATENT_DIM = config["vae"]["latent_dim"]
    NUM_DECODERS = config["vae"].get("num_decoders", 1)
    USE_ENSEMBLE = NUM_DECODERS > 1

    DATA_DIR = "data/"
    PLOT_DIR = os.path.join("src", "plots")
    ARTIFACT_DIR = os.path.join("src", "artifacts")

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ----- SEEDING -----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # ----- DEVICE -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- LOAD DATA -----
    data = np.load(os.path.join(DATA_DIR, "tasic-pca50.npy"))
    labels = np.load(os.path.join(DATA_DIR, "tasic-ttypes.npy"))
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # ----- SPLIT DATA -----
    val_ratio = 0.1
    n_val = int(len(data_tensor) * val_ratio)
    n_train = len(data_tensor) - n_val
    train_dataset, val_dataset = random_split(
        data_tensor, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training: {n_train}, Validation: {n_val}")
    print(f"Training {'EVAE' if USE_ENSEMBLE else 'VAE'} with {NUM_DECODERS} decoder(s)")

    # ----- INIT MODEL -----
    model = (
        EVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, num_decoders=NUM_DECODERS).to(device)
        if USE_ENSEMBLE else
        VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    suffix = f"{'EVAE' if USE_ENSEMBLE else 'VAE'}_ld{LATENT_DIM}_d{NUM_DECODERS}_ep{EPOCHS}_bs{BATCH_SIZE}_lr{LR:.0e}_seed{SEED}"
    best_model_path = os.path.join(ARTIFACT_DIR, f"{suffix}_best.pth")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    # ----- TRAIN LOOP -----
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            beta = min(1.0, epoch / 30)
            loss = -model.elbo(x, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                loss = -model.elbo(x, beta=1.0)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        scheduler.step()

    # ----- LATENT EXTRACTION -----
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in DataLoader(data_tensor, batch_size=256):
            q = model.encoder(batch.to(device))
            latents.append(q.mean.cpu())
    latents = torch.cat(latents, dim=0).numpy()

    # ----- SAVE -----
    np.save(os.path.join(ARTIFACT_DIR, f"latents_{suffix}.npy"), latents)
    # np.save(os.path.join(ARTIFACT_DIR, f"train_losses_{suffix}.npy"), train_losses)
    # np.save(os.path.join(ARTIFACT_DIR, f"val_losses_{suffix}.npy"), val_losses)


    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"loss_{suffix}.png"))
    plt.close()

    print(f"Training done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    main(config)
