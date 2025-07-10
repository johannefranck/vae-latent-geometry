import os
import torch
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from src.vae import EVAE
import torch.optim as optim
from collections import OrderedDict

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def load_raw_data(data_path="data/"):
    data = np.load(os.path.join(data_path, "tasic-pca50.npy"))
    return torch.tensor(data, dtype=torch.float32)

def get_data_splits(data_tensor, batch_size, seed):
    val_ratio = 0.1
    n_val = int(len(data_tensor) * val_ratio)
    n_train = len(data_tensor) - n_val

    train_dataset, val_dataset = random_split(
        data_tensor, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

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
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["training"]["batch_size"]
    LR = config["training"]["lr"]
    LATENT_DIM = config["vae"]["latent_dim"]
    N_RERUNS = 10
    DECODER_COUNTS = [1, 2, 3, 4, 5, 6]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_ROOT = "models_v103"
    ENCODER_SOURCE_DIR = os.path.join(SAVE_ROOT, "encoders")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    print("=== Loading raw data ONCE ===")
    data_tensor = load_raw_data()
    input_dim = data_tensor.shape[1]

    # === Optional: Train encoders ===
    # Uncomment this block to retrain and overwrite encoders
    
    print("=== Step 1: Training encoders ===")
    os.makedirs(ENCODER_SOURCE_DIR, exist_ok=True)
    for rerun in range(N_RERUNS):
        seed = 1000 + rerun
        set_seed(seed)
        train_loader, val_loader = get_data_splits(data_tensor, batch_size=BATCH_SIZE, seed=seed)
    
        model = EVAE(input_dim=input_dim, latent_dim=LATENT_DIM, num_decoders=1).to(DEVICE)
        best_model, train_losses, val_losses = train_model(model, train_loader, val_loader, DEVICE, EPOCHS, LR)
    
        encoder_path = os.path.join(ENCODER_SOURCE_DIR, f"encoder_rerun{rerun}.pt")
        torch.save({k: v for k, v in best_model.items() if k.startswith("encoder.")}, encoder_path)
    
        loss_path = os.path.join(ENCODER_SOURCE_DIR, f"loss_rerun{rerun}.png")
        plot_loss(train_losses, val_losses, loss_path)
        print(f"Saved encoder + loss curve for rerun {rerun}")

    print("\n=== Step 2: Training EVAE with decoder ensembles ===")
    test_input = torch.randn(1, input_dim).to(DEVICE)

    for num_decoders in DECODER_COUNTS:
        dec_dir = os.path.join(SAVE_ROOT, f"dec{num_decoders}")
        os.makedirs(dec_dir, exist_ok=True)

        for rerun in range(N_RERUNS):
            seed = 1000 * num_decoders + rerun
            set_seed(seed)
            train_loader, val_loader = get_data_splits(data_tensor, batch_size=BATCH_SIZE, seed=seed)

            # === Load encoder ===
            encoder_path = os.path.join(ENCODER_SOURCE_DIR, f"encoder_rerun{rerun}.pt")
            encoder_state = torch.load(encoder_path)

            # === Build model and inject encoder ===
            model = EVAE(input_dim=input_dim, latent_dim=LATENT_DIM, num_decoders=num_decoders).to(DEVICE)
            model_state = model.state_dict()
            encoder_state_filtered = OrderedDict({k: v for k, v in encoder_state.items()})
            model_state.update(encoder_state_filtered)
            model.load_state_dict(model_state)

            # log encoder mean
            with torch.no_grad():
                z_debug = model.encoder(test_input).mean
                print("...")
                print(f"[DEBUG] encoder_rerun{rerun} mean: {z_debug.mean().item():.4f}")

            # Freeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = False

            # === Train ===
            best_model, train_losses, val_losses = train_model(model, train_loader, val_loader, DEVICE, EPOCHS, LR)

            model_path = os.path.join(dec_dir, f"model_rerun{rerun}.pt")
            loss_path = os.path.join(dec_dir, f"loss_rerun{rerun}.png")
            torch.save(best_model, model_path)
            plot_loss(train_losses, val_losses, loss_path)

            print(f"[✓] Trained EVAE with {num_decoders} decoders | rerun {rerun}")

if __name__ == "__main__":
    main()
