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



