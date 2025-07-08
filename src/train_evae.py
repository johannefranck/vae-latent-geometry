import os
import sys
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from src.vae import VAE, EVAE
import torch.optim as optim

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def load_data(path, batch_size, seed):
    data = np.load(os.path.join(path, "tasic-pca50.npy"))
    data_tensor = torch.tensor(data, dtype=torch.float32)

    val_ratio = 0.1
    n_val = int(len(data_tensor) * val_ratio)
    n_train = len(data_tensor) - n_val

    train_dataset, val_dataset = random_split(data_tensor, [n_train, n_val],
                                              generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return data_tensor, train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs, lr, device, save_prefix):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            beta = min(1.0, epoch / 30)
            loss = -model.elbo(x, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
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
            torch.save(model.state_dict(), f"{save_prefix}_best.pth")

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        scheduler.step()

    # Save loss curves
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curve - {save_prefix}")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png")
    plt.close()

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seeds = config["training"]["seeds"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    ensemble_sizes = config["training"]["ensemble_sizes"]
    input_dim = config["vae"]["input_dim"]
    latent_dim = config["vae"]["latent_dim"]

    data_dir = "data/"
    artifact_dir = os.path.join("src", "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        print(f"\nTraining models for seed {seed}")
        set_seed(seed)
        data_tensor, train_loader, val_loader = load_data(data_dir, batch_size, seed)

        for ensemble_size in ensemble_sizes:
            print(f" - Training EVAE with {ensemble_size} decoder(s)")
            model = EVAE(input_dim=input_dim, latent_dim=latent_dim,
                         num_decoders=ensemble_size).to(device)

            suffix = f"EVAE_ld{latent_dim}_dec{ensemble_size}_ep{epochs}_bs{batch_size}_lr{lr}_seed{seed}"
            save_prefix = os.path.join(artifact_dir, suffix)
            train_model(model, train_loader, val_loader, epochs, lr, device, save_prefix)

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/config_train_evae.yaml"
    main(config_file)
