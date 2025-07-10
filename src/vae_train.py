import os
import random
import torch
import yaml
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from src.vae import VAE

# single vae training
def main():
    # ----- CONFIG -----
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    SEED = 12
    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["training"]["batch_size"]
    LR = config["training"]["lr"]
    INPUT_DIM = config["vae"]["input_dim"]
    LATENT_DIM = config["vae"]["latent_dim"]
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
    colors = np.load(os.path.join(DATA_DIR, "tasic-colors.npy"))

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # ----- SPLIT DATA -----
    val_ratio = 0.1
    n_val = int(len(data_tensor) * val_ratio)
    n_train = len(data_tensor) - n_val
    train_dataset, val_dataset = random_split(data_tensor, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training: {n_train}, Validation: {n_val}")
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

    # ----- INIT MODEL -----
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    filename_suffix = f"VAE_ld{LATENT_DIM}_ep{EPOCHS}_bs{BATCH_SIZE}_lr{LR:.0e}_seed{SEED}"
    best_model_path = os.path.join(ARTIFACT_DIR, f"vae_best_seed{SEED}.pth")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    # ----- TRAIN LOOP -----
    for epoch in range(EPOCHS):
        vae.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            beta = min(1.0, epoch / 30)
            loss = -vae.elbo(x, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # ----- VALIDATION -----
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                loss = -vae.elbo(x, beta=1.0)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        scheduler.step()

    # ----- LATENT EXTRACTION -----
    vae.eval()
    latents = []
    with torch.no_grad():
        for batch in DataLoader(data_tensor, batch_size=256):
            q = vae.encoder(batch.to(device))
            latents.append(q.mean.cpu())
    latents = torch.cat(latents, dim=0).numpy()

    # ----- SAVE -----
    np.save(os.path.join(ARTIFACT_DIR, f"latents_{filename_suffix}.npy"), latents)
    np.save(os.path.join(ARTIFACT_DIR, f"train_losses_{filename_suffix}.npy"), train_losses)
    np.save(os.path.join(ARTIFACT_DIR, f"val_losses_{filename_suffix}.npy"), val_losses)
    torch.save(vae.state_dict(), os.path.join(ARTIFACT_DIR, f"vae_{filename_suffix}.pth"))
    torch.save(vae.decoder.state_dict(), os.path.join(ARTIFACT_DIR, f"decoder_{filename_suffix}.pth"))

    # ----- PLOT -----
    plot_latent_space(
        latents=latents,
        labels=labels,
        title="Latent space colored by cell type",
        save_path=os.path.join(PLOT_DIR, f"latent_{filename_suffix}.png")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"loss_{filename_suffix}.png"))
    plt.close()

    print(f"Training done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
