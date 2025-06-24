import torch
import torch.nn as nn
import torch.distributions as td

class GaussianPrior(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(latent_dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(latent_dim), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        std = torch.exp(log_std).clamp(min=1e-4, max=10.0)  # max clamp
        mean = torch.nan_to_num(mean, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0)
        return td.Independent(td.Normal(loc=mean, scale=std), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2 * output_dim)
        )

    def forward(self, z):
        mean_log_std = self.net(z)
        mean, log_std = mean_log_std.chunk(2, dim=-1)
        std = torch.exp(log_std).clamp(min=1e-4, max=1.0)
        mean = torch.nan_to_num(mean, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0)
        return td.Independent(td.Normal(loc=mean, scale=std), 1)


class VAE(nn.Module):
    def __init__(self, input_dim=50, latent_dim=2):
        super().__init__()

        encoder_net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2 * latent_dim)  # outputs both mean and log_std
        )

        self.prior = GaussianPrior(latent_dim)
        self.encoder = GaussianEncoder(encoder_net)
        self.decoder = GaussianDecoder(latent_dim, input_dim)

    def elbo(self, x, beta=1.0):
        q = self.encoder(x)
        z = q.rsample()
        recon_logprob = self.decoder(z).log_prob(x)
        kl = q.log_prob(z) - self.prior().log_prob(z)
        return (recon_logprob - beta * kl).mean()

    def forward(self, x):
        return -self.elbo(x)

    def sample(self, n=1):
        z = self.prior().rsample((n,))
        return self.decoder(z).mean
