import torch
import numpy as np
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps



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



def plot_latent_space(latents, labels=None, title="Latent space of VAE", save_path=None):
    plt.figure(figsize=(6, 6))
    
    if labels is not None:
        sns.scatterplot(
            x=latents[:, 0], y=latents[:, 1],
            hue=labels, s=4, alpha=0.7, palette="tab20", legend=False
        )
    else:
        plt.scatter(latents[:, 0], latents[:, 1], s=2, alpha=0.6, color="steelblue")

    plt.title(title)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()





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


def plot_latent_density_with_splines(latents, labels, splines, res=300, seed=12, filename="src/plots/latent_density_with_splines.png"):
    x = latents[:, 0]
    y = latents[:, 1]

    # Axis bounds with square margin
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    max_span = max(x_span, y_span)
    x_center = (x.max() + x.min()) / 2
    y_center = (y.max() + y.min()) / 2
    span_margin = 0.1 * max_span
    half_span = max_span / 2 + span_margin
    xlim = (x_center - half_span, x_center + half_span)
    ylim = (y_center - half_span, y_center + half_span)

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

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        log_metric.T,
        origin='lower',
        extent=(*xlim, *ylim),
        cmap=colormaps['copper'],
        alpha=0.8
    )

    sns.scatterplot(x=x, y=y, hue=labels, palette="tab20", s=4, alpha=0.4, legend=False, ax=ax)

    # Plot splines
    from matplotlib import cm
    colors = cm.get_cmap("tab10", len(splines))

    for i, spline_pair in enumerate(splines):
        color = colors(i)
        if isinstance(spline_pair, (list, tuple)):
            spline_init, spline_opt = spline_pair
            t_vals = torch.linspace(0, 1, 200, device=spline_init.omega.device)
            z_init = spline_init(t_vals).detach().cpu().numpy()
            z_opt = spline_opt(t_vals).detach().cpu().numpy()
            ax.plot(z_init[:, 0], z_init[:, 1], '--', linewidth=1.2, alpha=0.6, color=color)
            ax.plot(z_opt[:, 0], z_opt[:, 1], '-', linewidth=2.0, color=color)
        else:
            spline = spline_pair
            t_vals = torch.linspace(0, 1, 200, device=spline.omega.device)
            z_path = spline(t_vals).detach().cpu().numpy()
            ax.plot(z_path[:, 0], z_path[:, 1], '-', linewidth=1.5, color=color)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")  # Maintain square shape

    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_title(f"Geodesics in Latent Space (seed {seed})")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Density-based metric value log(Gₓ)")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



