import matplotlib.pyplot as plt
import seaborn as sns

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
