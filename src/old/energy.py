import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
from src.vae import VAE

def set_seed(seed=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(12)

class TorchGeodesic(nn.Module):
    def __init__(self, decoder, n_poly, point_pair, device='cpu'):
        super().__init__()
        self.decoder = decoder
        self.n_poly = n_poly
        self.point_pair = point_pair.to(device)
        self.device = device
        self.basis = self.init_basis()
        self.dim = point_pair.shape[-1]
        gen = torch.Generator(device=device).manual_seed(12)
        self.params = nn.Parameter(torch.randn((self.basis.shape[1], self.dim), generator=gen, device=device))

    def forward(self, t):
        line = self._eval_line(t)
        poly = self._eval_poly(t)
        return (line + poly).T

    def calculate_energy(self, t_):
        latent = self.forward(t_)
        decoded = self.decoder(latent).mean
        diff = decoded[:, 1:] - decoded[:, :-1]
        energy = torch.sum(torch.sum(diff ** 2, dim=-1)) * (len(t_) - 1)
        return energy

    def _basis(self):
        np_ = self.n_poly
        tc = torch.linspace(0, 1, np_ + 1, device=self.device)[1:-1]
        boundary = torch.zeros((2, 4 * np_), device=self.device)
        boundary[0, 0] = 1.0
        boundary[1, -4:] = 1.0

        zeroth = torch.zeros((np_ - 1, 4 * np_), device=self.device)
        first = torch.zeros((np_ - 1, 4 * np_), device=self.device)
        second = torch.zeros((np_ - 1, 4 * np_), device=self.device)

        for i in range(np_ - 1):
            si = 4 * i
            h = 1.0 / np_
            t = (i + 1) * h
            fill_0 = torch.tensor([1, t, t ** 2, t ** 3], device=self.device)
            zeroth[i, si:si+4] = fill_0
            zeroth[i, si+4:si+8] = -fill_0
            fill_1 = torch.tensor([0, 1, 2*t, 3*t**2], device=self.device)
            first[i, si:si+4] = fill_1
            first[i, si+4:si+8] = -fill_1
            fill_2 = torch.tensor([0, 0, 2, 6*t], device=self.device)
            second[i, si:si+4] = fill_2
            second[i, si+4:si+8] = -fill_2

        constraints = torch.cat((boundary, zeroth, first, second), dim=0)
        _, S, Vh = torch.linalg.svd(constraints)
        return Vh.T[:, S.size()[0]:]

    def init_basis(self):
        return self._basis()

    def _eval_line(self, t):
        p0, p1 = self.point_pair[0, :], self.point_pair[1, :]
        return ((1 - t)[:, None] * p0 + t[:, None] * p1).T

    def _eval_poly(self, t):
        coefs = self.basis @ self.params
        coefs = coefs.view(self.n_poly, 4, self.dim)
        t_scaled = t * self.n_poly
        idx = torch.floor(t_scaled).clamp(0, self.n_poly - 1).long()
        local_t = t_scaled - idx.float()
        tp = torch.stack([local_t**i for i in range(4)], dim=1)
        seg_coefs = coefs[idx]
        return torch.einsum("ti,tid->td", tp, seg_coefs).T

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_path = "src/artifacts/vae_best.pth"
    input_dim = 50
    latent_dim = 2
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(decoder_path, map_location=device))
    vae.eval()
    decoder = vae.decoder

    n_poly = 4
    point_pair = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32, device=device)
    spline = TorchGeodesic(decoder, n_poly, point_pair, device=device).to(device)

    from src.optimize_energy import optimize_spline
    spline = optimize_spline(spline, decoder, None, steps=1000, lr=1e-2, patience=500)

    t = torch.linspace(0, 1, 2000, device=device, requires_grad=True)
    z = spline(t)
    dz = torch.zeros_like(z)
    for i in range(z.shape[1]):
        dz[:, i] = torch.autograd.grad(z[:, i], t, grad_outputs=torch.ones_like(t), create_graph=True)[0] * spline.n_poly
    ddz = torch.zeros_like(z)
    for i in range(dz.shape[1]):
        ddz[:, i] = torch.autograd.grad(dz[:, i], t, grad_outputs=torch.ones_like(t), create_graph=True)[0] * spline.n_poly

    z_np = z.detach().cpu().numpy()
    dz_np = dz.detach().cpu().numpy()
    ddz_np = ddz.detach().cpu().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(6, 12))
    axs[0].plot(z_np[:, 0], z_np[:, 1], 'r-', lw=2, label='Geodesic Spline')
    axs[0].scatter([point_pair[0,0].item(), point_pair[1,0].item()], [point_pair[0,1].item(), point_pair[1,1].item()], c='black', label='Endpoints')
    axs[0].set_title("Spline in Latent Space")
    axs[0].axis("equal")
    axs[0].legend()

    axs[1].plot(dz_np[:, 0], label="dx/dt")
    axs[1].plot(dz_np[:, 1], label="dy/dt")
    axs[1].set_title("Velocity (First Derivative)")
    axs[1].legend()

    axs[2].plot(ddz_np[:, 0], label="d²x/dt²")
    axs[2].plot(ddz_np[:, 1], label="d²y/dt²")
    axs[2].set_title("Acceleration (Second Derivative)")
    axs[2].legend()

    print("Spline start:", z_np[0])
    print("Spline end:  ", z_np[-1])
    print("Expected a:  ", point_pair[0].cpu().numpy())
    print("Expected b:  ", point_pair[1].cpu().numpy())

    plt.tight_layout()
    plt.savefig("src/plots/spline_loaded_2.png", dpi=300)
    print("Saved plot to src/plots/spline_loaded_2.png")

if __name__ == "__main__":
    main()