import os
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from src.vae import VAE

def main():
    # Config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["training"]["batch_size"]
    LR = config["training"]["lr"]
    LATENT_DIM = config["vae"]["latent_dim"]
    INPUT_DIM = config["vae"]["input_dim"]

    # Seeding
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Data
    data = np.load("data/tasic-pca50.npy")
    print("PCA mean:", data.mean(axis=0).mean())
    print("PCA std:", data.std(axis=0).mean())
    # Normalize PCA scores to mean 0, std 1
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    print("PCA mean:", data.mean(axis=0).mean())
    print("PCA std:", data.std(axis=0).mean())

    tensor_data = torch.tensor(data, dtype=torch.float32)
    val_ratio = 0.1
    n_val = int(len(data) * val_ratio)
    n_train = len(data) - n_val
    train_ds, val_ds = random_split(tensor_data, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LR)

    # Train
    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(EPOCHS):
        vae.train()
        train_loss = 0
        beta = min(1.0, epoch / 30)
        for x in train_loader:
            x = x.to(device)
            loss = -vae.elbo(x, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                val_loss += vae(x).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        elbo, recon, kl = vae.elbo(x, beta=beta, return_parts=True)

        print(f"Epoch {epoch+1} | Train: {train_loss:.2f} | Val: {val_loss:.2f}")
        print(f"Epoch {epoch+1} | Recon: {-recon.item():.2f} | KL: {kl.item():.2f} | ELBO: {-elbo.item():.2f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(vae.state_dict(), "src/artifacts/vae_best.pth")

    # Plot loss
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.title("VAE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("src/plots/loss.png")

if __name__ == "__main__":
    main()
