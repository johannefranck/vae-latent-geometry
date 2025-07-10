import os
import sys
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from src.vae import EVAE
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

    train_dataset, val_dataset = random_split(
        data_tensor, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return data_tensor, train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs, lr, device, save_path_prefix):
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
            torch.save(model.state_dict(), f"{save_path_prefix}_best.pth")

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        scheduler.step()

    # Save loss curve
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curve - {save_path_prefix}")
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_loss.png")
    plt.close()

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    reruns = config["training"]["reruns"]
    ensemble_sizes = config["training"]["ensemble_sizes"]
    global_seed = config["training"]["global_seed"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    input_dim = config["vae"]["input_dim"]
    latent_dim = config["vae"]["latent_dim"]

    data_dir = "data/"
    artifact_dir = os.path.join("src", "artifacts", "ensemble")
    os.makedirs(artifact_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[INFO] Using global_seed={global_seed}")
    set_seed(global_seed)
    _, train_loader, val_loader = load_data(data_dir, batch_size, global_seed)

    for num_decoders in ensemble_sizes:
        for rerun in reruns:
            print(f"\n[INFO] Training EVAE: decoders={num_decoders}, rerun={rerun}")
            model = EVAE(input_dim=input_dim, latent_dim=latent_dim,
                         num_decoders=num_decoders).to(device)

            suffix = f"EVAE_dec{num_decoders}_rerun{rerun}"
            save_prefix = os.path.join(artifact_dir, suffix)

            train_model(model, train_loader, val_loader, epochs, lr, device, save_prefix)

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/config_train_evae.yaml"
    main(config_file)
