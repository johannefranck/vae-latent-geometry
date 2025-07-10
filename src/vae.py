import torch
import torch.nn as nn
import torch.distributions as td
from copy import deepcopy
import random

class GaussianPrior(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std", torch.ones(latent_dim))

    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2 * latent_dim)  # output: mean and log_std
        )

    def forward(self, x):
        mean, log_std = self.encoder_net(x).chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(min=-4.0, max=2.0))  # std in [~0.02, ~7.4]
        return td.Independent(td.Normal(mean, std), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2 * output_dim)  # output: mean and log_std
        )

    def forward(self, z):
        mean_log_std = self.decoder_net(z)
        mean, log_std = mean_log_std.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(min=-2.0, max=2.0))
        return td.Independent(td.Normal(mean, std), 1)

class VAE(nn.Module):
    """
    Sinlge VAE.
    """
    def __init__(self, input_dim=50, latent_dim=2):
        super().__init__()
        self.encoder = GaussianEncoder(input_dim, latent_dim)
        self.decoder = GaussianDecoder(latent_dim, input_dim)
        self.prior = GaussianPrior(latent_dim)

    def elbo(self, x, beta=1.0, return_parts=False):
        q = self.encoder(x)
        z = q.rsample()
        p = self.decoder(z)
        recon_logprob = p.log_prob(x)
        kl = q.log_prob(z) - self.prior().log_prob(z)
        elbo = recon_logprob - beta * kl
        if return_parts:
            return elbo.mean(), recon_logprob.mean(), kl.mean()
        return elbo.mean()

    def forward(self, x):
        return -self.elbo(x)

    def sample(self, n=1):
        z = self.prior().sample((n,))
        return self.decoder(z).mean

class EVAE(nn.Module):
    """
    Ensemble VAE with multiple decoders.
    """
    def __init__(self, input_dim=50, latent_dim=2, num_decoders=3):
        super().__init__()
        self.encoder = GaussianEncoder(input_dim, latent_dim)
        self.decoders = nn.ModuleList([
            GaussianDecoder(latent_dim, input_dim)
            for _ in range(num_decoders)
        ])
        self.decoder = self.decoders[0]  # legacy support for single decoder code
        self.prior = GaussianPrior(latent_dim)

    def elbo(self, x, beta=1.0, decoder_idx=None, return_parts=False):
        q = self.encoder(x)
        z = q.rsample()

        if decoder_idx is None:
            decoder = random.choice(self.decoders)
        else:
            decoder = self.decoders[decoder_idx]

        p = decoder(z)
        recon_logprob = p.log_prob(x)
        kl = q.log_prob(z) - self.prior().log_prob(z)
        elbo = recon_logprob - beta * kl

        if return_parts:
            return elbo.mean(), recon_logprob.mean(), kl.mean()
        return elbo.mean()

    def forward(self, x):
        return -self.elbo(x)

    def sample(self, n=1, decoder_idx=None):
        z = self.prior().sample((n,))
        if decoder_idx is None:
            decoder = random.choice(self.decoders)
        else:
            decoder = self.decoders[decoder_idx]
        return decoder(z).mean
