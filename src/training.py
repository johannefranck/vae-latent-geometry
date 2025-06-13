import os
import random
import torch
import yaml
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from src.vae import VAE, loss_fn
from src.plotting import plot_latent_space  


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




    # ----- SEEDING -----
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # ----- DEVICE -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ----- LOAD DATA -----
    data = np.load(os.path.join(DATA_DIR, "tasic-pca50.npy"))
    labels = np.load(os.path.join(DATA_DIR, "tasic-ttypes.npy"))
    colors = np.load(os.path.join(DATA_DIR, "tasic-colors.npy"))

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # Split data into training and validation sets
    VAL_RATIO = 0.1
    n_val = int(len(data_tensor) * VAL_RATIO)
    n_train = len(data_tensor) - n_val
    train_dataset, val_dataset = random_split(data_tensor, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}, Colors: {colors.shape}")


    # ----- INIT MODEL -----
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LR)

    print(f"Config: latent_dim={LATENT_DIM}, epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}")

    train_losses, val_losses = [], []

    # save best model
    best_val_loss = float('inf')
    best_model_path = os.path.join(ARTIFACT_DIR, f"vae_best.pth")

    for epoch in range(EPOCHS):
        # ---- TRAIN ----
        vae.train()
        total_loss = 0
        for batch in train_loader:
            x = batch.to(device)
            x_recon, mu, logvar = vae(x)
            loss = loss_fn(x, x_recon, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader) # average per batch
        train_losses.append(train_loss)

        # ---- VALIDATION ----
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                x_recon, mu, logvar = vae(x)
                val_loss += loss_fn(x, x_recon, mu, logvar).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # ---- LOGGING ----
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), best_model_path)

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    lr_str = f"{LR:.0e}"  # Format like 1e-03
    filename_suffix = f"ld{LATENT_DIM}_ep{EPOCHS}_bs{BATCH_SIZE}_lr{lr_str}"


    # Plot training and validation loss
    loss_plot_path = os.path.join(PLOT_DIR, f"loss_{filename_suffix}.png")
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training and Validation Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()



    # ----- EXTRACT LATENTS -----
    vae.eval()
    latents = []
    with torch.no_grad():
        for batch in DataLoader(data_tensor, batch_size=256):
            batch = batch.to(device)
            _, mu, _ = vae(batch)
            latents.append(mu.cpu())
    latents = torch.cat(latents, dim=0).numpy()


    # ----- PLOT LATENTS -----
    latent_plot_path = os.path.join(PLOT_DIR, f"latent_{filename_suffix}.png")
    plot_latent_space(
        latents,
        labels,
        title="Latent space colored by cell type",
        save_path=latent_plot_path
    )

    # ------- SAVE LATENTS AND LOSSES -------
    np.save(os.path.join(ARTIFACT_DIR, f"train_losses_{filename_suffix}.npy"), train_losses)
    np.save(os.path.join(ARTIFACT_DIR, f"val_losses_{filename_suffix}.npy"), val_losses)
    np.save(os.path.join(ARTIFACT_DIR, f"latents_{filename_suffix}.npy"), latents)


    # ---- SAVE FULL MODEL AND DECODER ----
    torch.save(vae.state_dict(), os.path.join(ARTIFACT_DIR, f"vae_{filename_suffix}.pth"))
    torch.save(vae.decoder.state_dict(), os.path.join(ARTIFACT_DIR, f"decoder_{filename_suffix}.pth"))



if __name__ == "__main__":
    main()
    print("Training complete. Latent space plotted.")
