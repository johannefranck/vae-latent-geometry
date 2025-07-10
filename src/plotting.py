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





# for ensembles
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List

from src.vae import EVAE
from src.select_representative_pairs import load_pairs
from src.geodesics import (
    GeodesicSplineBatch,
    construct_nullspace_basis,
)


def plot_cov_results(cov_geo, cov_euc, pairs=10):
    geo_means = {int(k): np.mean(v) for k, v in cov_geo.items()}
    euc_means = {int(k): np.mean(v) for k, v in cov_euc.items()}

    decoders = sorted(geo_means.keys())
    geo_y = [geo_means[d] for d in decoders]
    euc_y = [euc_means[d] for d in decoders]

    plt.figure(figsize=(8, 6))
    plt.plot(decoders, geo_y, label="Geodesic CoV", marker="o")
    plt.plot(decoders, euc_y, label="Euclidean CoV", marker="s")
    plt.xlabel("Number of decoders")
    plt.ylabel("Coefficient of Variation")
    plt.xticks(decoders)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"models_p{pairs}/cov_results/CoV_plot.png")
    print(f"Saved plot to models_p{pairs}/cov_results/CoV_plot.png")

def plot_latents_from_reruns(
    model_root="models_v103",
    data_path="data/tasic-pca50.npy",
    label_path="data/tasic-ttypes.npy",
    reruns=range(10),
    num_decoders=6,
    latent_dim=2,
    save_path="models_p10/results/latent_encodings_by_rerun.png"
):
    import seaborn as sns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.load(data_path), dtype=torch.float32).to(device)
    labels = np.load(label_path)

    fig, axes = plt.subplots(2, (len(reruns) + 1) // 2, figsize=(16, 8), squeeze=False)
    axes = axes.flatten()

    for i, rerun in enumerate(reruns):
        model_path = Path(model_root) / f"dec{num_decoders}" / f"model_rerun{rerun}.pt"
        model = EVAE(input_dim=50, latent_dim=latent_dim, num_decoders=num_decoders).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            latents = model.encoder(data).mean.cpu().numpy()

        ax = axes[i]
        sns.scatterplot(
            x=latents[:, 0], y=latents[:, 1], hue=labels,
            palette="tab20", s=4, alpha=0.5, legend=False, ax=ax
        )
        ax.set_title(f"Encoder Latents (rerun {rerun})")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")

    for j in range(len(reruns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved latent projection plots to: {save_path}")




def plot_latent_geodesics_from_saved_splines(
    model_path, spline_path, data_path, pairfile, save_path, num_decoders, max_pairs=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model ===
    model = EVAE(input_dim=50, latent_dim=2, num_decoders=num_decoders).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    encoder = model.encoder

    # === Load data and latent ===
    data = torch.tensor(np.load(data_path), dtype=torch.float32).to(device)
    all_z = encoder(data).mean.detach().cpu().numpy()

    # === Load selected pairs ===
    _, pairs = load_pairs(pairfile)
    pair_set = set(tuple(p) for p in pairs[:max_pairs])

    # === Load splines ===
    with open(spline_path, "r") as f:
        spline_data = json.load(f)

    t_vals = torch.linspace(0, 1, 1000, device=device)
    n_poly = 4
    basis, _ = construct_nullspace_basis(n_poly=n_poly, device=device)
    basis = basis.to(device)

    plt.figure(figsize=(8, 8))
    plt.scatter(all_z[:, 0], all_z[:, 1], s=2, alpha=0.4)

    shown = 0
    for entry in spline_data:
        pair = (entry["idx_a"], entry["idx_b"])
        if pair not in pair_set:
            continue

        a = torch.tensor(entry["a"], dtype=torch.float32, device=device).unsqueeze(0)
        b = torch.tensor(entry["b"], dtype=torch.float32, device=device).unsqueeze(0)
        omega = torch.tensor(entry["omega"], dtype=torch.float32, device=device).unsqueeze(0)

        spline = GeodesicSplineBatch(a, b, basis, omega, n_poly=n_poly)
        zs = spline(t_vals).squeeze(1).detach().cpu().numpy()

        plt.plot(zs[:, 0], zs[:, 1], lw=2)
        plt.scatter([a[0, 0].item(), b[0, 0].item()], [a[0, 1].item(), b[0, 1].item()],
                    c="red", s=20)
        shown += 1
        if shown >= max_pairs:
            break

    plt.title("Latent space with saved geodesics")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved latent space plot with geodesics to: {save_path}")



def build_distance_matrices(
    spline_json_path: str,
    rerun: int,
    num_decoders: int,
    cluster_map_path: str,
    plot_path_geo: str,
    plot_path_euc: str,
    json_out_path: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    Path(plot_path_geo).parent.mkdir(parents=True, exist_ok=True)
    Path(plot_path_euc).parent.mkdir(parents=True, exist_ok=True)
    Path(json_out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(spline_json_path, "r") as f:
        spline_data = json.load(f)

    with open(cluster_map_path, "r") as f:
        cluster_data = json.load(f)

    representatives = cluster_data["representatives"]
    id_to_label = {rep["index"]: rep["label"] for rep in representatives}
    sorted_indices = sorted(id_to_label.keys())
    cluster_ids = [id_to_label[idx] for idx in sorted_indices]
    index_to_matrix_idx = {idx: i for i, idx in enumerate(sorted_indices)}

    N = len(cluster_ids)
    D_geo = np.full((N, N), np.nan)
    D_euc = np.full((N, N), np.nan)

    for entry in spline_data:
        if entry["rerun"] == rerun and entry["num_decoders"] == num_decoders:
            i = index_to_matrix_idx.get(entry["idx_a"])
            j = index_to_matrix_idx.get(entry["idx_b"])
            geo_dist = entry.get("geo_distance") or entry.get("length_geodesic")
            a = np.array(entry["a"])
            b = np.array(entry["b"])
            euc_dist = np.linalg.norm(a - b)
            if i is not None and j is not None:
                if geo_dist is not None:
                    D_geo[i, j] = geo_dist
                    D_geo[j, i] = geo_dist
                D_euc[i, j] = euc_dist
                D_euc[j, i] = euc_dist

    np.fill_diagonal(D_geo, 0.0)
    np.fill_diagonal(D_euc, 0.0)

    # vmax = max(np.nanmax(D_geo), np.nanmax(D_euc))

    def plot_matrix(D, title, path):
        plt.figure(figsize=(12, 12))
        sns.heatmap(
            D,
            cmap="copper",
            square=True,
            annot=False,
            xticklabels=cluster_ids,
            yticklabels=cluster_ids,
            cbar=False,
            vmin=0,
            # vmax=vmax  # wrong! johanne stupido
        )
        plt.xticks(rotation=90, fontsize=3)
        plt.yticks(rotation=0, fontsize=3)
        plt.title(title)
        plt.xlabel("Cluster ID")
        plt.ylabel("Cluster ID")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()


    plot_matrix(D_geo, f"Geodesic Distance Matrix - rerun {rerun}, dec {num_decoders}", plot_path_geo)
    plot_matrix(D_euc, f"Euclidean Distance Matrix - rerun {rerun}, dec {num_decoders}", plot_path_euc)

    with open(json_out_path, "w") as f:
        json.dump({
            "rerun": rerun,
            "num_decoders": num_decoders,
            "cluster_ids": cluster_ids,
            "geodesic_distance_matrix": D_geo.tolist(),
            "euclidean_distance_matrix": D_euc.tolist()
        }, f, indent=2)

    return D_geo, D_euc, cluster_ids
