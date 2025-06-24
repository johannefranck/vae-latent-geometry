import torch
import torch.nn as nn
import os
# Import the actual VAE class from the user's source
# Ensure your src/vae.py contains a VAE class with a 'decoder_net' attribute
# that is a torch.nn.Module and performs the decoding.
from src.vae import VAE 

# --- TorchGeodesic Class ---
class TorchGeodesic(torch.nn.Module):
    def __init__(self, decoder, n_poly, point_pair, device='cpu'):
        super().__init__()
        self.decoder = decoder
        self.n_poly = n_poly
        self.point_pair = point_pair.to(device)
        self.basis = self.init_basis()  # (num_free_params, 4 * n_poly)
        self.dim = point_pair.shape[-1] # (2, dim)
        self.params =torch.nn.Parameter(1* torch.zeros((self.basis.shape[1], self.dim), device=device)) # (num_free_params, dim)
        self.point_pair = point_pair.to(device)

    def forward(self, t):
        line = self._eval_line(t, self.point_pair)
        poly = self._eval_poly(t)
        return (line + poly).T

    def calculate_energy(self, t_):
        latent = self.forward(t_)
        decoded = self.decoder(latent).mean  # Assuming decoder is a callable that takes latent vectors
        diff = decoded[:, 1:] - decoded[:, :-1]
        #  jnp.dot(x, x)
        squared_diff = torch.sum(diff ** 2, dim=-1)  # Sum over the last dimension (squared norm)
        energy = torch.sum(squared_diff, dim=-1) * (len(t_) - 1)  # Multiply by the number of intervals
        return energy


    def _basis(self):
        np = self.n_poly
        tc = torch.linspace(0, 1, np + 1)[1:-1] # time cutoffs between polynomials
        boundary = torch.zeros((2, 4 * np), device=self.point_pair.device)
        boundary[0, 0] = 1.0
        boundary[1, -4:] = 1.0
        zeroth, first, second = torch.zeros((3, np - 1, 4 * np), device=self.point_pair.device)

        for i in range(np - 1):
            si = 4 * i
            fill_0 = torch.tensor([1.0, tc[i], tc[i] ** 2, tc[i] ** 3], device=self.point_pair.device)
            zeroth[i, si:si + 4] = fill_0
            zeroth[i, si + 4:si + 8] = -fill_0      
            fill_1 = torch.tensor([0.0, 1.0, 2.0 * tc[i], 3.0 * tc[i] ** 2], device=self.point_pair.device)
            first[i, si:si + 4] = fill_1
            first[i, si + 4:si + 8] = -fill_1
            fill_2 = torch.tensor([0.0, 0.0, 2.0, 6.0 * tc[i]], device=self.point_pair.device)
            second[i, si:si + 4] = fill_2
            second[i, si + 4:si + 8] = -fill_2      
        constraints = torch.cat((boundary, zeroth, first, second), dim=0)
        _, S, Vh = torch.linalg.svd(constraints)

        return Vh.T[:, S.size()[0]:]
    
    def init_basis(self):
        return self._basis()
    
    def _eval_line(self, t, point_pair):
        p0, p1 = point_pair[0, :], point_pair[1, :]
        a, b = p1 - p0, p0
        return a[:, None] * t + b[:, None]
    
    def _eval_poly(self, t):
        coefs = self.basis @ self.params
        coefs = coefs.reshape(self.n_poly, 4, self.dim)
        idx = torch.floor(t * self.n_poly).clip(0, self.n_poly - 1).long()
        coefs_idx = coefs[idx]  # t × n_term × d
        tp = t ** torch.arange(4, device=self.point_pair.device)[:, None]
        return torch.einsum("ted,et->td", coefs_idx, tp).T
    

# --- Main Execution Block ---
def main():
    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the trained VAE
    decoder_path = "src/artifacts/vae_best.pth"
    input_dim = 50
    latent_dim = 2

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)

    print(f"Loading VAE weights from: {decoder_path}")
    try:
        vae.load_state_dict(torch.load(decoder_path, map_location=device))
        vae.eval()
        print("VAE model loaded successfully.")
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        return

    # Extract only the decoder from the VAE
    actual_decoder = vae.decoder  # This is your GaussianDecoder instance

    # 3. Initialize TorchGeodesic
    n_poly = 3
    point_pair = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32, device=device)

    print(f"\nInitializing TorchGeodesic with n_poly={n_poly} and point_pair={point_pair.tolist()}")
    geodesic_model = TorchGeodesic(
        decoder=actual_decoder,
        n_poly=n_poly,
        point_pair=point_pair,
        device=device
    ).to(device)

    print(f"Number of free parameters: {geodesic_model.params.shape[0]}")

    # --- 🔧 Optimize the geodesic_model.params ---
    geodesic_model.train()
    t_vals = torch.linspace(0, 1, 2000, device=device)

    optimizer = torch.optim.Adam([geodesic_model.params], lr=1e-2)
    best_loss = float('inf')
    best_params = geodesic_model.params.data.clone()
    patience = 500
    counter = 0

    print("\nOptimizing geodesic parameters...")

    for step in range(1000):
        optimizer.zero_grad()
        loss = geodesic_model.calculate_energy(t_vals)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:4d} | Energy: {loss.item():.6f} | Grad norm: {geodesic_model.params.grad.norm():.4f}")

        # Early stopping
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            best_params = geodesic_model.params.data.clone()
            counter = 0
        else:
            counter += 1

        if counter > patience:
            print("Early stopping triggered.")
            break

    geodesic_model.params.data.copy_(best_params)
    print("Optimization finished.\n")





    # 4. Evaluate spline and energy
    t_eval = torch.linspace(0, 1, 2000, device=device)

    print(f"\nEvaluating geodesic...")
    latent_path = geodesic_model(t_eval)
    print(f"Latent path shape: {latent_path.shape}")

    decoded = actual_decoder(latent_path).mean
    print(f"Decoded output shape: {decoded.shape}")

    energy = geodesic_model.calculate_energy(t_eval)
    print(f"Calculated energy: {energy.item():.4f}")

    print("Done.")
    import matplotlib.pyplot as plt

    # Ensure t_eval requires grad for derivative computation
    t_eval.requires_grad_(True)

    # Get latent path
    z = geodesic_model(t_eval)  # shape (100, 2)

    # Initialize empty containers for derivatives
    dz = torch.zeros_like(z)
    ddz = torch.zeros_like(z)

    # Compute first and second derivatives for each latent dim
    for i in range(z.shape[1]):
        dz_i = torch.autograd.grad(z[:, i], t_eval, grad_outputs=torch.ones_like(t_eval), create_graph=True)[0]
        ddz_i = torch.autograd.grad(dz_i, t_eval, grad_outputs=torch.ones_like(t_eval), create_graph=True)[0]
        dz[:, i] = dz_i
        ddz[:, i] = ddz_i

    # Convert to NumPy
    t_np = t_eval.detach().cpu().numpy()
    dz_np = dz.detach().cpu().numpy()
    ddz_np = ddz.detach().cpu().numpy()

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].plot(t_np, dz_np[:, 0], label="dz₁/dt")
    axs[0].plot(t_np, dz_np[:, 1], label="dz₂/dt")
    axs[0].set_title("First Derivative (Velocity)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t_np, ddz_np[:, 0], label="d²z₁/dt²")
    axs[1].plot(t_np, ddz_np[:, 1], label="d²z₂/dt²")
    axs[1].set_title("Second Derivative (Acceleration)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("spline_smoothness_check.png", dpi=300)
    plt.show()






if __name__ == "__main__":
    main()

