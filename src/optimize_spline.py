import torch
import numpy as np
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.vae import VAE
from matplotlib import colormaps


# ---- Utility ----

def set_seed(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(12)

def resample_path(path, n_poly):
    path = np.array(path)
    deltas = np.diff(path, axis=0)
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    cumdist = np.concatenate([[0], np.cumsum(dists)])
    t_orig = cumdist / cumdist[-1]
    t_new = np.linspace(0, 1, n_poly + 1)
    x_new = np.interp(t_new, t_orig, path[:, 0])
    y_new = np.interp(t_new, t_orig, path[:, 1])
    return np.stack([x_new, y_new], axis=1)


# ---- PyTorch Cubic Spline Class ----

class TorchCubicSpline(torch.nn.Module):
    def __init__(self, ctrl_pts, knots=None):
        super().__init__()
        self.n_pts = len(ctrl_pts)
        self.register_parameter('ctrl_pts', torch.nn.Parameter(torch.tensor(ctrl_pts, dtype=torch.float32)))

        self.t = torch.linspace(0, 1, self.n_pts)
        if knots is None:
            self.knots = self.t
        else:
            self.knots = torch.tensor(knots, dtype=torch.float32)

    def forward(self, t_vals):
        t_vals = t_vals.clone().detach().float().unsqueeze(-1) if isinstance(t_vals, torch.Tensor) else torch.tensor(t_vals, dtype=torch.float32).unsqueeze(-1)
        x_spline = self.eval_spline(t_vals, self.ctrl_pts[:, 0])
        y_spline = self.eval_spline(t_vals, self.ctrl_pts[:, 1])
        return torch.stack([x_spline, y_spline], dim=1)

    def eval_spline(self, t_vals, y):
        coeffs = self.natural_cubic_spline_coeffs(self.t, y)
        t_vals = t_vals.squeeze()
        out = torch.zeros_like(t_vals)
        for i in range(self.n_pts - 1):
            mask = (t_vals >= self.t[i]) & (t_vals <= self.t[i+1])
            h = self.t[i+1] - self.t[i]
            xi = t_vals[mask] - self.t[i]
            a, b, c, d = coeffs[i]
            out[mask] = a + b*xi + c*xi**2 + d*xi**3
        return out

    def natural_cubic_spline_coeffs(self, x, y):
        n = len(x)
        h = x[1:] - x[:-1]
        alpha = (3 / h[1:]) * (y[2:] - y[1:-1]) - (3 / h[:-1]) * (y[1:-1] - y[:-2])

        A = torch.zeros((n, n), dtype=torch.float32)
        rhs = torch.zeros(n, dtype=torch.float32)

        A[0, 0] = A[-1, -1] = 1
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            rhs[i] = alpha[i-1]

        c = torch.linalg.solve(A, rhs)
        b, d = [], []
        for i in range(n-1):
            b_i = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2*c[i]) / 3
            d_i = (c[i+1] - c[i]) / (3*h[i])
            b.append(b_i)
            d.append(d_i)

        coeffs = []
        for i in range(n-1):
            a = y[i]
            coeffs.append((a, b[i], c[i], d[i]))
        return coeffs
    
    def fixed_endpoint_mask(self):
        """Returns a mask for optimizing only the interior control points."""
        mask = torch.ones_like(self.ctrl_pts, dtype=torch.bool)
        mask[0] = False  # fix first point
        mask[-1] = False  # fix last point
        return mask



# ---- Optimization and Plotting ----

def optimize_spline(spline, decoder,                                                                                                              n_steps=10, lr=1e-2):
    t_dense = torch.linspace(0, 1, 200)
    losses = []
    path_history = []
    mask = spline.fixed_endpoint_mask()
    optimizer = torch.optim.Adam([spline.ctrl_pts], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        z_path = spline(t_dense)                        # (N, 2)
        dz_dt = (z_path[1:] - z_path[:-1]) * 200        # finite difference

        # Compute Jacobians one-by-one
        J_list = []
        for z in z_path:
            z = z.requires_grad_(True)
            Jz = torch.autograd.functional.jacobian(decoder, z)  # (D, 2)
            #Jz = torch.autograd.functional.jacobian(lambda x: decoder(x) * 10.0, z)

            J_list.append(Jz.unsqueeze(0))
        J = torch.cat(J_list, dim=0)                    # (N, D, 2)
        # print(f"Jacobian shape: {J.shape}")
        # J = J.permute(0, 2, 1)                          # (N, 2, D)
        with torch.no_grad():
            norm = torch.norm(J, dim=(1, 2)).mean().item()
            #print(f"Avg Jacobian norm: {norm:.4f}") #<- almost constant => manifold is close to flat in that region

        G = torch.bmm(J.transpose(1, 2), J)             # (N, D, D)
        dz = dz_dt.unsqueeze(1)                         # (N-1, 1, 2)
        G_mid = G[:-1]                                  # (N-1, 2, 2)

        energy_terms = torch.bmm(torch.bmm(dz, G_mid), dz.transpose(1, 2)).squeeze()
        energy = energy_terms.mean()

        energy.backward()
        with torch.no_grad():
            spline.ctrl_pts.grad[~mask] = 0.0
        #print("grad norm:", spline.ctrl_pts.grad.norm().item()) #not zero, gradients are flowing -> good!

        optimizer.step()
        losses.append(energy.item())

        with torch.no_grad():
            path_history.append(spline(t_dense).detach().cpu().numpy())

        print(f"Step {step+1}, Energy: {energy.item():.4f}")

    return losses, path_history

import matplotlib.patches as patches

def plot_metric_ellipses(z_path, G):
    for z, Gz in zip(z_path[::20], G[::20]):  # every 20th point
        eigvals, eigvecs = torch.linalg.eigh(Gz)
        width, height = 0.2 * eigvals.sqrt().tolist()
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipse = patches.Ellipse(
            xy=z.detach().cpu().numpy(),
            width=width,
            height=height,
            angle=angle,  # now keyword
            edgecolor='black',
            facecolor='none',
            lw=1
        )
        plt.gca().add_patch(ellipse)


def plot_spline_optimization(latents, labels, ctrl_pts, path_history, filename):
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.3, legend=False)

    for i, path in enumerate(path_history):
        alpha = 0.2 + 0.8 * (i + 1) / len(path_history)
        plt.plot(path[:, 0], path[:, 1], lw=2, alpha=alpha, label=f"Step {i+1}" if i in [0, len(path_history)-1] else None)

    plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'bo--', markersize=5, label='Control Points')
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.title("Spline Optimization Over Steps")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_single_spline(latents, labels, spline, ctrl_pts, filename):
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.4, legend=False)
    t_dense = torch.linspace(0, 1, 200)
    with torch.no_grad():
        path = spline(t_dense).numpy()
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2)
    plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'bo--', markersize=5)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_single_spline_with_metric(latents, labels, spline, decoder, filename):
    import matplotlib.patches as patches

    t_dense = torch.linspace(0, 1, 200)
    z_path = spline(t_dense)

    # Compute Jacobians and pullback metrics
    J_list = []
    for z in z_path:
        z = z.requires_grad_(True)
        Jz = torch.autograd.functional.jacobian(decoder, z) # shape (D, 2)
        J_list.append(Jz.unsqueeze(0))
    J = torch.cat(J_list, dim=0)        # (N, D, 2)
    J = J.permute(0, 2, 1)              # (N, 2, D)
    G = torch.bmm(J, J.transpose(1, 2)) # (N, 2, 2) # G = J Jᵀ

    # Plot
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.4, legend=False)

    path_np = z_path.detach().cpu().numpy()
    plt.plot(path_np[:, 0], path_np[:, 1], 'g-', linewidth=2)

    # Draw ellipses at regular intervals
    for z, Gz in zip(z_path[::20], G[::20]):
        eigvals, eigvecs = torch.linalg.eigh(Gz)
        width, height = (eigvals.clamp(min=1e-6).sqrt() * 0.2).tolist()  # clamp avoids sqrt of zero
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipse = patches.Ellipse(
            xy=z.detach().cpu().numpy(),
            width=width,
            height=height,
            angle=angle,  # now keyword
            edgecolor='black',
            facecolor='none',
            lw=1
        )
        plt.gca().add_patch(ellipse)

    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.title("Optimized Spline with Pullback Metric Ellipses")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_all_splines(latents, labels, splines_init, splines_opt, filename):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.4, legend=False)
    t_dense = torch.linspace(0, 1, 200)

    for init_spline, opt_spline in zip(splines_init, splines_opt):
        with torch.no_grad():
            init_path = init_spline(t_dense).cpu().numpy()
            opt_path = opt_spline(t_dense).cpu().numpy()

        plt.plot(init_path[:, 0], init_path[:, 1], 'r-', linewidth=1.5, alpha=0.6)  # initial spline
        plt.plot(opt_path[:, 0], opt_path[:, 1], 'g-', linewidth=1.5, alpha=0.8)   # optimized spline

    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.title("Initial (red) vs Optimized (green) Splines")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_metric_grid(decoder, latents, xlim=(-3, 3), ylim=(-4, 2), res=20, scale=3, filename="src/plots/latent_metric_grid.png"):
    import matplotlib.patches as patches

    x_vals = np.linspace(*xlim, res)
    y_vals = np.linspace(*ylim, res)

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if 'latents' in locals():
        plt.scatter(latents[:, 0], latents[:, 1], alpha=0.05, s=1, color='gray')


    for x in x_vals:
        for y in y_vals:
            z = torch.tensor([x, y], dtype=torch.float32).requires_grad_(True)
            #J = torch.autograd.functional.jacobian(decoder, z)  # (D, 2)
            output = decoder(z)  # shape (D,)
            J_rows = []
            for d in range(output.shape[0]):
                grad = torch.autograd.grad(output[d], z, retain_graph=True)[0]  # shape (2,)
                J_rows.append(grad.unsqueeze(0))
            J = torch.cat(J_rows, dim=0)  # shape (D, 2)


            G = J.T @ J  # (2, 2)

            eigvals, eigvecs = torch.linalg.eigh(G)
            eigvals = eigvals.clamp(min=1e-6)
            sqrt_eigvals = (eigvals.clamp(min=1e-6).sqrt() * scale)
            if sqrt_eigvals.shape[0] >= 2:
                width, height = sqrt_eigvals[:2].tolist()
            else:
                continue  # skip if eigenvalue computation fails

            angle = np.degrees(torch.atan2(eigvecs[1, 0], eigvecs[0, 0]))

            ellipse = patches.Ellipse(
                (x, y), width, height, angle=angle,
                edgecolor="black", facecolor="none", lw=0.8
            )
            ax.add_patch(ellipse)
            #print(f"Ellipse at ({x:.2f}, {y:.2f}) size: {width:.2f}×{height:.2f}")


    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.title("Pullback Metric Ellipses Across Latent Grid")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_multiple_splines_with_metric(latents, labels, splines, decoder, filename):
    import matplotlib.patches as patches
    t_dense = torch.linspace(0, 1, 200)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latents[:, 0], y=latents[:, 1], hue=labels, palette="tab20", s=4, alpha=0.4, legend=False)

    for spline in splines:
        z_path = spline(t_dense)
        # Compute pullback metric G
        J_list = []
        for z in z_path:
            z = z.requires_grad_(True)
            Jz = torch.autograd.functional.jacobian(decoder, z)
            J_list.append(Jz.unsqueeze(0))
        J = torch.cat(J_list, dim=0)        # (N, D, 2)
        G = torch.bmm(J.transpose(1, 2), J) # (N, 2, 2)

        path_np = z_path.detach().cpu().numpy()
        plt.plot(path_np[:, 0], path_np[:, 1], linewidth=1.5, color='green')

        for z, Gz in zip(z_path[::20], G[::20]):
            eigvals, eigvecs = torch.linalg.eigh(Gz)
            width, height = (eigvals.clamp(min=1e-6).sqrt() * 0.2).tolist()
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellipse = patches.Ellipse(
                xy=z.detach().cpu().numpy(),
                width=width,
                height=height,
                angle=angle,
                edgecolor='black',
                facecolor='none',
                lw=0.8
            )
            plt.gca().add_patch(ellipse)

    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.title("Multiple Optimized Splines with Pullback Metric Ellipses")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_metric_density(latents, xi, yi, sigma=0.3, epsilon=1e-4):
    grid = np.stack([xi.ravel(), yi.ravel()], axis=-1)  # shape (res*res, 2)
    density = np.zeros(len(grid))
    for z in latents:
        diff = grid - z
        norm_sq = np.sum(diff**2, axis=1)
        density += np.exp(-0.5 * norm_sq / sigma**2)
    density /= (len(latents) * (2 * np.pi * sigma**2))  # optional normalization
    Gx = 1 / (density + epsilon)  # this is the "cost"
    log_metric = np.log1p(Gx)
    return log_metric.reshape(xi.shape)


def plot_latents_with_density(latents, labels, res=300, filename="src/plots/aml_style_metric_background.png"):
    x = latents[:, 0]
    y = latents[:, 1]

    # Compute plot bounds with margin
    margin_x = 0.1 * (x.max() - x.min())
    margin_y = 0.1 * (y.max() - y.min())
    xlim = (x.min() - margin_x, x.max() + margin_x)
    ylim = (y.min() - margin_y, y.max() + margin_y)

    # Prepare grid
    xi, yi = np.mgrid[xlim[0]:xlim[1]:res*1j, ylim[0]:ylim[1]:res*1j]
    log_metric = evaluate_metric_density(latents, xi, yi, sigma=0.3, epsilon=1e-4)

    # Setup plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        log_metric.T,
        origin='lower',
        extent=(*xlim, *ylim),
        cmap=colormaps['copper'],
        aspect='equal',
        alpha=0.8
    )

    # Add scatter on top
    sns.scatterplot(x=x, y=y, hue=labels, palette="tab20", s=4, alpha=0.4, legend=False, ax=ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_title("Latent Space with Metric-Based Background")

    # Colorbar (aligned to axis height using `make_axes_locatable`)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Density-based metric value log(Gₓ)")

    plt.savefig(filename)
    plt.close()



def plot_latent_density_with_splines(latents, labels, splines, res=300, filename="src/plots/latent_density_with_splines.png"):

    x = latents[:, 0]
    y = latents[:, 1]

    # Axis bounds with margin
    margin_x = 0.1 * (x.max() - x.min())
    margin_y = 0.1 * (y.max() - y.min())
    xlim = (x.min() - margin_x, x.max() + margin_x)
    ylim = (y.min() - margin_y, y.max() + margin_y)

    # Grid and metric
    xi, yi = np.mgrid[xlim[0]:xlim[1]:res*1j, ylim[0]:ylim[1]:res*1j]
    grid = np.stack([xi.ravel(), yi.ravel()], axis=-1)
    density = np.zeros(len(grid))
    sigma = 0.3
    epsilon = 1e-4
    for z in latents:
        diff = grid - z
        norm_sq = np.sum(diff**2, axis=1)
        density += np.exp(-0.5 * norm_sq / sigma**2)
    density /= (len(latents) * (2 * np.pi * sigma**2))
    Gx = 1 / (density + epsilon)
    log_metric = np.log1p(Gx).reshape(xi.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        log_metric.T,
        origin='lower',
        extent=(*xlim, *ylim),
        cmap=colormaps['copper'],
        aspect='equal',
        alpha=0.8
    )

    sns.scatterplot(x=x, y=y, hue=labels, palette="tab20", s=4, alpha=0.4, legend=False, ax=ax)

    t_vals = torch.linspace(0, 1, 200)
    for spline in splines:
        z_path = spline(t_vals).detach().cpu().numpy()
        ax.plot(z_path[:, 0], z_path[:, 1], '#30D5C8', linewidth=1.5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_title("Latent Space with Metric-Based Background and Splines")

    # Colorbar (aligned to axis height using `make_axes_locatable`)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Density-based metric value log(Gₓ)")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()





# ---- Main ----

with open("src/artifacts/dijkstra_paths.pkl", "rb") as f:
    dijkstra_paths = pickle.load(f)


def main():
    latents = np.load("src/artifacts/latents_ld2_ep600_bs64_lr1e-03.npy")
    labels = np.load("data/tasic-ttypes.npy")
    decoder = VAE().decoder
    decoder.eval()

    splines_init = []
    splines_opt = []

    for i, path in enumerate(dijkstra_paths):#[:10]):  
        ctrl_pts = resample_path(path, n_poly=20)
        init_spline = TorchCubicSpline(ctrl_pts)
        splines_init.append(init_spline)

        spline = TorchCubicSpline(ctrl_pts.copy())
        optimize_spline(spline, decoder, n_steps=10, lr=0.001)
        splines_opt.append(spline)

    #plot_single_spline_with_metric(latents, labels, spline, decoder, "src/plots/spline_with_metric.png")

    #plot_all_splines(latents, labels, splines_init, splines_opt, "src/plots/all_splines_comparison.png")

    plot_metric_grid(decoder, latents, xlim=(-3, 3), ylim=(-4, 2), res=25, scale=0.1, filename="src/plots/latent_metric_grid.png")
    plot_latents_with_density(latents, labels, res=300, filename="src/plots/latent_density_background.png")

    # plot_multiple_splines_with_metric(latents, labels, splines_opt, decoder, filename="src/plots/all_splines_with_metric.png")
    plot_latent_density_with_splines(latents, labels, splines_opt, filename="src/plots/latent_density_with_splines.png")


if __name__ == "__main__":
    main()
