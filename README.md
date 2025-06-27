# vae-latent-geometry

## Identifying and Measuring Identifiable Distances of Geodesics at Large Scale

Aim of producing geodesics in regard to this project by Syrota et al.: https://github.com/mustass/identifiable-latent-metric-space

**Aim**: Implementing algorithms at large scale

**Data**: Obtained from here (pre-processed): https://github.com/berenslab/rna-seq-tsne/tree/master/data/tasic-preprocessed



Geodesics: Single decoder (initial pipeline):
* dir src/single_decoder/ (batched!? not yet)
    * init_spline.py => spline_batch_seedX.pt (CHANGE TO TAKE ONLY SEED! model params are the same)
    * optimize_energy.py => spline_batch_optimized_seedX.pt 
    * density.py => geodesic_distances_seedX.json
Call src/single_nbatched.sh

Output visuals src/plots/single_decoder/

![vae12](src/plots/density_with_splines_seed12.png "vae latent space seed 12") ![vae123](src/plots/density_with_splines_seed123.png "vae latent space seed 123")
![geo-dist12](src/plots/geodesic_distance_seed12.png "vae latent space seed 12") ![geo-dist123](src/plots/geodesic_distance_seed123.png "vae latent space seed 123")

The numerical similarity should be more similar.

Geodesics: Ensemble (structure):
* dir src/ (should be batched!)
    * select_representative_pairs.py => selected_pairs.json. Samples one point from n_max different classes (bound 133 unique).
    * 

Call src/....sh