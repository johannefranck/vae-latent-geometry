import os
import argparse
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ----------------------------
# Components
# ----------------------------

class GaussianPrior(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std", torch.ones(latent_dim))

    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        out = self.encoder_net(x)
        mean, log_std = out.chunk(2, dim=-1)
        std = torch.exp(log_std)
        return td.Independent(td.Normal(mean, std), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net
        # self.log_scale = nn.Parameter(torch.tensor(0.0))  # Learnable log std

    def forward(self, z):
        mean = self.decoder_net(z)
        # scale = torch.exp(self.log_scale)
        # return td.Independent(td.Normal(mean, scale), 1)
        return td.Independent(td.Normal(mean, 5), 1)

class EVAE(nn.Module):
    def __init__(self, prior, encoder, decoder, num_decoders, beta=1.0):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = nn.ModuleList([deepcopy(decoder) for _ in range(num_decoders)])
        self.beta = beta

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        dec = np.random.choice(self.decoder)
        logpxz = dec(z).log_prob(x)
        kl = q.log_prob(z) - self.prior().log_prob(z)
        return torch.mean(logpxz - self.beta * kl)

    def forward(self, x):
        return -self.elbo(x)

# ----------------------------
# Networks
# ----------------------------

def make_encoder_net(input_dim, latent_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.SiLU(),
        nn.LayerNorm(256),
        nn.Linear(256, 128), nn.SiLU(),
        nn.LayerNorm(128),
        nn.Linear(128, 2 * latent_dim)
    )

def make_decoder_net(latent_dim, output_dim):
    return nn.Sequential(
        nn.Linear(latent_dim, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, output_dim)
    )

# ----------------------------
# Training
# ----------------------------

def train_model(model, optimizer, train_loader, val_loader, epochs, device, save_dir, seed):
    train_losses, val_losses = [], []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_loss = []
        for x, in train_loader:
            x = x.to(device)
            loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        train_losses.append(np.mean(epoch_loss))

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([model(x.to(device)).item() for x, in val_loader])
            val_losses.append(val_loss)

        print(f"Epoch {epoch+1:3d} | Train: {train_losses[-1]:.2f} | Val: {val_losses[-1]:.2f}")

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Neg ELBO")
    plt.title("Training Curve")
    plt.savefig(f"{save_dir}/plots/loss_curve_seed{seed}.png")
    plt.close()

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--num-decoders", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="experiment")
    parser.add_argument("--data-path", type=str, default="data/tasic-pca50.npy")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Data
    data = torch.from_numpy(np.load(args.data_path).astype(np.float32))
    # print("Data shape:", data.shape) # 23822, 50
    # print("Per-feature variance:", data.var(dim=0)) # 1737.2992, 1096.1960,  411.3057,  249.8994, etc...
    # print("Total variance (mean over features):", data.var(dim=0).mean())

    n = len(data)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    val_len = int(0.1 * n)
    train_loader = DataLoader(TensorDataset(data[idx[val_len:]]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(data[idx[:val_len]]), batch_size=args.batch_size)

    # Model
    input_dim = data.shape[1]
    encoder = GaussianEncoder(make_encoder_net(input_dim, args.latent_dim))
    decoder = GaussianDecoder(make_decoder_net(args.latent_dim, input_dim))
    prior = GaussianPrior(args.latent_dim)
    model = EVAE(prior, encoder, decoder, num_decoders=args.num_decoders, beta=1.0).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("beta = ", model.beta)
    train_model(model, optimizer, train_loader, val_loader, args.epochs, args.device, args.save_dir, args.seed)

    # Save model + decoder weights
    torch.save(model.state_dict(), f"{args.save_dir}/model_seed{args.seed}.pt")
    # for i, dec in enumerate(model.decoder):
    #     torch.save(dec.state_dict(), f"{args.save_dir}/decoder_{i}_seed{args.seed}.pt")

    print(f"\nSaved model + {args.num_decoders} decoders.")

    # Latent space check
    data_tensor = data.to(args.device)
    with torch.no_grad():
        z = model.encoder(data_tensor).base_dist.loc
        print("Mean of latent z across dataset:", z.mean(dim=0))
        print("Std of latent z across dataset:", z.std(dim=0))

if __name__ == "__main__":
    main()
