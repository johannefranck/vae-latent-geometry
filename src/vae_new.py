import torch
import torch.nn as nn
import torch.distributions as td

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
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        std = torch.exp(log_std).clamp(min=1e-4, max=10.0)
        return td.Independent(td.Normal(mean, std), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        mean = self.decoder_net(z)
        std = torch.ones_like(mean) * 0.1  # fixed std
        return td.Independent(td.Normal(mean, std), 1)

class VAE(nn.Module):
    def __init__(self, input_dim=50, latent_dim=2):
        super().__init__()

        self.encoder = GaussianEncoder(nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2 * latent_dim)
        ))

        self.decoder = GaussianDecoder(nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        ))

        self.prior = GaussianPrior(latent_dim)

    def elbo(self, x, beta=1.0):
        q = self.encoder(x)
        z = q.rsample()
        recon = self.decoder(z)
        logpx = recon.log_prob(x)
        kl = q.log_prob(z) - self.prior().log_prob(z)
        return (logpx - beta * kl).mean()

    def forward(self, x):
        return -self.elbo(x)

    def sample(self, n=1):
        z = self.prior().sample((n,))
        return self.decoder(z).mean
