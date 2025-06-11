import torch
import numpy as np
import torch.nn as nn
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.vae import VAE


class CubicSpline(nn.Module):
    def __init__(self, start, end, decoder, num_polys=5, device='cpu'):
        super().__init__()
        self.device = device
        self.start = start.to(device).detach()
        self.end = end.to(device).detach()
        self.num_polys = num_polys
        self.dim = start.shape[-1]
        self.decoder = decoder

        # Compute fixed basis
        self.basis = self._compute_basis(num_polys).to(device)  # shape: (4*num_polys, basis_dim)
        basis_dim = self.basis.shape[1]

        # Learnable parameters
        self.params = nn.Parameter(torch.zeros(basis_dim, self.dim, device=device))

    def _compute_basis(self, n):
        t = torch.linspace(0, 1, n + 1)[1:-1]

        boundary = torch.zeros((2, 4 * n))
        boundary[0, 0] = 1
        boundary[1, -4:] = 1

        def segment_rows(order_fn):
            rows = torch.zeros(n - 1, 4 * n)
            for i in range(n - 1):
                si = 4 * i
                fill = order_fn(t[i])
                rows[i, si:si + 4] = fill
                rows[i, si + 4:si + 8] = -fill
            return rows

        zeroth = segment_rows(lambda t_: torch.tensor([1, t_, t_**2, t_**3]))
        first = segment_rows(lambda t_: torch.tensor([0, 1, 2*t_, 3*t_**2]))
        second = segment_rows(lambda t_: torch.tensor([0, 0, 2, 6*t_]))

        constraints = torch.cat([boundary, zeroth, first, second], dim=0)

        # Null space of constraints
        u, s, v = torch.linalg.svd(constraints, full_matrices=False)
        null_basis = v[s.numel():].T  # shape: (4n, basis_dim)
        return null_basis

    def forward(self, t):
        # Evaluate spline at t (in [0,1])
        coeffs = self.basis @ self.params  # (4n, dim)
        coeffs = coeffs.view(self.num_polys, 4, self.dim)  # (n, 4, d)

        idx = torch.clamp((t * self.num_polys).long(), max=self.num_polys - 1)  # (N,)
        local_t = t * self.num_polys - idx  # (N,)

        t_powers = torch.stack([torch.ones_like(local_t),
                                local_t,
                                local_t ** 2,
                                local_t ** 3], dim=1)  # (N, 4)

        selected_coeffs = coeffs[idx]  # (N, 4, d)
        poly_part = torch.einsum("ni,nid->nd", t_powers, selected_coeffs)  # (N, d)

        linear_part = (1 - t).unsqueeze(1) * self.start + t.unsqueeze(1) * self.end  # (N, d)

        return poly_part + linear_part

    def fit_to_points(self, t, x, steps=500, lr=1e-2):
        opt = torch.optim.Adam([self.params], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = self.forward(t)
            loss = F.mse_loss(pred, x)
            loss.backward()
            opt.step()

    # def energy(self, t):
    #     x = self.forward(t)  # (N, d)
    #     dx = x[1:] - x[:-1]
    #     return torch.sum(dx.norm(dim=1) ** 2) * len(t)

    def energy(self, t):
        z = self.forward(t)                    # latent path: (N, d)
        x = self.decoder(z)                    # decoded: (N, obs_dim)

        dx = x[1:] - x[:-1]                    # finite differences
        return torch.sum(dx.norm(dim=1)**2) * len(t)


    def length(self, t):
        x = self.forward(t)  # (N, d)
        dx = x[1:] - x[:-1]
        return torch.sum(dx.norm(dim=1))





def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_poly = 5

    vae = VAE(input_dim=50, latent_dim=2)  
    vae.load_state_dict(torch.load("src/artifacts/vae_ld2_ep30_bs64_lr1e-03.pth", map_location=device))
    vae.to(device)
    vae.eval()
    decoder = vae.decoder

    with open("src/artifacts/dijkstra_paths.pkl", "rb") as f:
        dijkstra_paths = pickle.load(f)

    grid = np.load("src/artifacts/grid.npy")
    grid = torch.tensor(grid, dtype=torch.float32, device=device)

    splines = []
    for i, coords in enumerate(dijkstra_paths):
        points = grid[coords].detach()
        t = torch.linspace(0, 1, len(points), device=device)

        spline = CubicSpline(points[0], points[-1], decoder=decoder, num_polys=n_poly, device=device)
        # spline.fit_to_points(t, points, steps=10, lr=1e-2) # euclidean seems
        optimizer = torch.optim.Adam([spline.params], lr=1e-2)
        for _ in range(10):  # or 20 steps
            optimizer.zero_grad()
            energy = spline.energy(t)
            energy.backward()
            optimizer.step()

        splines.append(spline)

        print(f"Fitted spline {i+1}/{len(dijkstra_paths)} | Path length: {len(points)}")

        

    # Visualize optimized splines
    plt.figure(figsize=(7, 7))
    grid_np = grid.cpu().numpy()
    plt.scatter(grid_np[:, 0], grid_np[:, 1], s=1, alpha=0.2, label="Grid", color="gray")

    t_dense = torch.linspace(0, 1, 200, device=device)

    for i, spline in enumerate(splines):
        path = spline(t_dense).detach().cpu().numpy()
        plt.plot(path[:, 0], path[:, 1], lw=1.5, label=f"Spline {i}" if i < 10 else None)

    plt.title(f"Optimized Spline Paths ({len(splines)} total)")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("src/plots/spline_paths.pdf")
    plt.close()


    with open("src/artifacts/splines.pkl", "wb") as f:
        pickle.dump(splines, f)

    print(f"\nSaved {len(splines)} splines to src/artifacts/splines.pkl")


if __name__ == "__main__":
    main()
