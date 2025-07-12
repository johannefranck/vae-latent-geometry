import os
import torch
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from src.vae import EVAE  # Your own implementation
import torch.optim as optim

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def load_raw_data(data_path="data/"):
    data = np.load(os.path.join(data_path, "tasic-pca50.npy"))
    return torch.tensor(data, dtype=torch.float32)

def get_data_splits(data_tensor, batch_size, seed):
    val_ratio = 0.1
    n_val = int(len(data_tensor) * val_ratio)
    n_train = len(data_tensor) - n_val
    train_dataset, val_dataset = random_split(data_tensor, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def plot_loss(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(model, train_loader, val_loader, device, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    train_losses, val_losses = [], []
    best_model = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
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
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                loss = -model.elbo(x, beta=1.0)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        scheduler.step()

    return best_model, train_losses, val_losses

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", type=int, required=True, help="Rerun index (e.g. 0,1,2...)")
    args = parser.parse_args()
    rerun_id = args.rerun

    config_file = "configs/config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["training"]["batch_size"]
    LR = config["training"]["lr"]
    LATENT_DIM = config["vae"]["latent_dim"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAVE_ROOT = "models_v105"
    N_DECODERS = 10
    OUTPUT_DIR = os.path.join(SAVE_ROOT, f"dec{N_DECODERS}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Set seed unique per rerun ===
    encoder_seed = 1000 + rerun_id
    set_seed(encoder_seed)

    # === Load data ===
    data_tensor = load_raw_data()
    train_loader, val_loader = get_data_splits(data_tensor, batch_size=BATCH_SIZE, seed=12)
    input_dim = data_tensor.shape[1]

    # === Build model ===
    model = EVAE(input_dim=input_dim,
                 latent_dim=LATENT_DIM,
                 num_decoders=N_DECODERS).to(DEVICE)

    # === Train ===
    best_model, train_losses, val_losses = train_model(model, train_loader, val_loader, DEVICE, EPOCHS, LR)

    # === Save ===
    model_path = os.path.join(OUTPUT_DIR, f"model_rerun{rerun_id}.pt")
    loss_path = os.path.join(OUTPUT_DIR, f"loss_rerun{rerun_id}.png")
    torch.save(best_model, model_path)
    plot_loss(train_losses, val_losses, loss_path)

    print(f"[✓] Trained EVAE with {N_DECODERS} decoders | rerun {rerun_id}")

if __name__ == "__main__":
    main()
