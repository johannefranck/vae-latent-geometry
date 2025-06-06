import os
import random
import torch
import yaml
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
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
    data_loader = DataLoader(data_tensor, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}, Colors: {colors.shape}")


    # ----- INIT MODEL -----
    vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LR)


    # ----- TRAINING LOOP -----
    print(f"Training config: latent_dim={LATENT_DIM}, epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}")

    losses = []
    for epoch in range(EPOCHS):
        total_loss = 0
        vae.train()
        for batch in data_loader:
            x = batch.to(device)
            x_recon, mu, logvar = vae(x)
            loss = loss_fn(x, x_recon, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    lr_str = f"{LR:.0e}"  # Format like 1e-03
    filename_suffix = f"ld{LATENT_DIM}_ep{EPOCHS}_bs{BATCH_SIZE}_lr{lr_str}"


    # Plot training loss
    loss_plot_path = os.path.join(PLOT_DIR, f"loss_{filename_suffix}.png")
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("VAE Training Loss")
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
    np.save(os.path.join(ARTIFACT_DIR, f"losses_{filename_suffix}.npy"), losses)
    np.save(os.path.join(ARTIFACT_DIR, f"latents_{filename_suffix}.npy"), latents)




if __name__ == "__main__":
    main()
    print("Training complete. Latent space plotted.")
