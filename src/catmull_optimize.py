# catmull_optimize.py

from pathlib import Path
import argparse, pickle, numpy as np, matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

def resample_arc(path: np.ndarray, n: int) -> np.ndarray:
    """Even-arc-length samples of a 2-D poly-line (shape → [n,2])."""
    d   = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s   = np.insert(np.cumsum(d), 0, 0.0);   s /= s[-1]
    u   = np.linspace(0.0, 1.0, n)
    return np.vstack([np.interp(u, s, path[:, k]) for k in range(path.shape[1])]).T

class CatmullRom(nn.Module):
    """Centripetal Catmull–Rom with *learnable* inner tangents τ."""
    def __init__(self, knots: torch.Tensor):
        super().__init__()
        self.register_buffer("p", knots)
        # Initialize tau directly from knots as per Catmull-Rom standard
        # The first and last tau are zero because of fixed endpoints
        # self.tau will have shape (K-2, 2) if knots is (K, 2)
        initial_tau = 0.5 * (knots[2:] - knots[:-2])
        self.tau = nn.Parameter(initial_tau)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        K = self.p.size(0)
        s  = t * (K - 1)
        k  = torch.clamp(s.floor(), max=K - 2).long()
        u  = s - k.float()

        P0, P1 = self.p[k], self.p[k + 1]
        m0 = torch.zeros_like(P0);   m1 = torch.zeros_like(P1)
        # Apply learnable self.tau to the correct segments
        m0[k > 0]       = self.tau[(k - 1)[k > 0]]
        m1[k < K - 2]   = self.tau[k[k < K - 2]]

        u2, u3 = u*u, u*u*u
        h00 =  2*u3 - 3*u2 + 1
        h10 =      u3 - 2*u2 + u
        h01 = -2*u3 + 3*u2
        h11 =      u3 -     u2

        return (h00[:,None]*P0 + h10[:,None]*m0 +
                h01[:,None]*P1 + h11[:,None]*m1)

# The construct_basis and syrota_energy functions remain the same as they operate
# on the latent curve 'z', not directly on 'tau' or 'omega'.

def construct_basis(n_poly: int) -> torch.Tensor:
    """Null-space basis (App.C in the paper). Shape (4n , 4n-2)."""
    tc   = torch.linspace(0, 1, n_poly+1, dtype=torch.float32)[1:-1]
    rows = []

    # boundary S(0)=0 , S(1)=0
    e0, e1 = torch.zeros(4*n_poly), torch.zeros(4*n_poly)
    e0[0] = 1;   e1[-4:] = 1
    rows += [e0, e1]

    # C⁰/C¹/C² continuity at internal knots
    for i, t in enumerate(tc):
        s = 4*i
        p  = torch.tensor([1, t, t**2, t**3])
        d1 = torch.tensor([0, 1, 2*t, 3*t**2])
        d2 = torch.tensor([0, 0, 2, 6*t])
        for v in (p, d1, d2):
            r = torch.zeros(4*n_poly)
            r[s:s+4]     =  v
            r[s+4:s+8]   = -v
            rows.append(r)

    C = torch.stack(rows);   _, _, Vh = torch.linalg.svd(C)
    return Vh.T[:, C.size(0):].contiguous()

def syrota_energy(z: torch.Tensor,
                  decoder: nn.Module,
                  scale: float = 1.0) -> torch.Tensor:
    """
    Discrete Syrota energy: E = mean_t (Δzᵀ · G(t) · Δz)
    G(t) = J(t)ᵀ · J(t) where J = ∂ decoder_mean / ∂z.
    Gradients propagate back to the spline parameters.
    """
    dz   = (z[1:] - z[:-1]) * scale  # (T-1, 2)
    z_in = z[:-1] # (T-1, 2)

    x_mean = decoder(z_in).mean # (T-1, x_dim)

    # Compute batched Jacobian J: (T-1, x_dim, 2)
    # J[k, i, j] = d(x_mean[k, i]) / d(z_in[k, j])
    jacobian_slices = []
    for j in range(x_mean.size(1)):
        # Compute gradient of each x_mean feature with respect to z_in
        grad_j = torch.autograd.grad(outputs=x_mean[:, j].sum(),
                                     inputs=z_in,
                                     retain_graph=True,
                                     create_graph=True,
                                     allow_unused=True
                                    )[0]
        jacobian_slices.append(grad_j)
    J = torch.stack(jacobian_slices, dim=1) # Stacks to form (T-1, x_dim, 2)

    # Metric tensor G = J^T J: (T-1, 2, 2)
    G = torch.bmm(J.transpose(1, 2), J)

    dz = dz.unsqueeze(1) # (T-1, 1, 2)
    return torch.bmm(torch.bmm(dz, G), dz.transpose(1, 2)).mean()

def optimise_one(idx: int, n_seg: int,
                 steps: int, lr: float,
                 device: str = "cpu") -> None:

    # ---- load initial data ------------------------------------------------
    bundle = torch.load("src/artifacts/spline_init_syrota.pt", map_location=device)
    knots  = bundle["knots_list"][idx].to(device)

    # Instantiate CatmullRom spline and directly optimize its tau parameter
    cr = CatmullRom(knots).to(device)

    t_dense = torch.linspace(0, 1, 256, device=device)

    # Initial spline path using the CatmullRom's initial tau
    with torch.no_grad():
        z_init = cr(t_dense).cpu().numpy()

    # ---- load decoder -----------------------------------------------------
    from src.advanced_vae import AdvancedVAE
    vae = AdvancedVAE(input_dim=50, latent_dim=2).to(device)
    vae.load_state_dict(torch.load("src/artifacts/vae_best_avae.pth", map_location=device))
    vae.eval()
    decoder = vae.decoder

    # ---- Adam: Optimize cr.tau directly -----------------------------------
    # The CatmullRom model (cr) holds the 'tau' parameter.
    opt = Adam(cr.parameters(), lr=lr) # Optimize parameters of the CatmullRom module
    for s in range(1, steps+1):
        opt.zero_grad()
        # Pass the spline output directly to syrota_energy
        E = syrota_energy(cr(t_dense), decoder, scale=t_dense.numel())

        # Debugging: Print energy and gradient norm before step
        if s % 10 == 0 or s == 1: # Print more frequently to see progression
            print(f"step {s:4d} | energy {E.item():.6f}")
            # Perform backward pass only to get gradients for printing
            E.backward(retain_graph=True) # Retain graph as backward is called again for opt.step()
            if cr.tau.grad is not None:
                grad_norm = cr.tau.grad.norm().item()
                print(f"         | grad norm: {grad_norm:.6e}")
            else:
                print(f"         | grad: None (Issue with graph?)")
            opt.zero_grad() # Clear gradients after printing, before actual backward for opt.step()

        E.backward();   opt.step() # Actual backward pass and optimizer step

    # ---- save updated tau (instead of omega) --------------------------------
    # If you need to save the optimized spline, you save the 'tau' parameters
    # or the CatmullRom model state dict. For simplicity, we'll just save the final path.
    # If you need to store optimized 'tau' for future runs, you'd modify the bundle
    # or save a new artifact. For now, the spline itself is used for plotting.

    # ---- figure : initial vs optimised ---------------------------------------
    with torch.no_grad():
        z_final = cr(t_dense).cpu().numpy() # Get final spline path from optimized cr
    Path("src/plots").mkdir(parents=True, exist_ok=True)
    fname = f"src/plots/spline_opt_progress_idx{idx}.png"

    plt.figure(figsize=(4,4))
    plt.scatter(knots[:,0].cpu(), knots[:,1].cpu(), c="k", s=18, label="knots")
    plt.plot(z_init[:,0],  z_init[:,1],  "b--", lw=1.4, label="initial")
    plt.plot(z_final[:,0], z_final[:,1], "r-",  lw=1.8, label="optimised")

    # Dynamic plot limits to ensure both splines are visible
    all_points = np.vstack([z_init, z_final, knots.cpu().numpy()])
    min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
    min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
    padding_x = (max_x - min_x) * 0.1
    padding_y = (max_y - min_y) * 0.1
    plt.xlim(min_x - padding_x, max_x + padding_x)
    plt.ylim(min_y - padding_y, max_y + padding_y)

    plt.axis("equal"); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=300); plt.close()
    print("saved progress plot →", fname)


# ------------------------------------------------------------ main ---------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",      choices=["init", "optim"], default="init")
    p.add_argument("--idx",       type=int, default=0,   help="index into path list")
    p.add_argument("--segments",  type=int, default=8,   help="number of cubic segments")
    p.add_argument("--steps",     type=int, default=200)
    p.add_argument("--lr",        type=float, default=5e-3)
    args = p.parse_args()

    torch.manual_seed(0);   np.random.seed(0)

    if args.mode == "init":
        paths = pickle.loads(Path("src/artifacts/dijkstra_paths_avae.pkl").read_bytes())
        path  = paths[args.idx]
        knots = torch.tensor(resample_arc(path, args.segments+1), dtype=torch.float32)

        cr  = CatmullRom(knots)
        # The Syrota basis and omega calculation in 'init' mode are for a different
        # use case (projecting Catmull-Rom onto Syrota's *basis* for a specific representation).
        # If your goal is to simply optimize the Catmull-Rom using Syrota's energy,
        # then the `omega` part might not be strictly necessary for the `optim` mode.
        # However, if you want to initialize `omega` for an *omega-based* optimization,
        # keep this. For now, we're shifting to direct 'tau' optimization.
        B   = construct_basis(args.segments)
        t4n = torch.linspace(0,1,4*args.segments)
        with torch.no_grad():
            S = cr(t4n)
        line  = (1-t4n)[:,None]*knots[0] + t4n[:,None]*knots[-1]
        omega = torch.linalg.lstsq(B, S-line).solution # For initial setup if omega is needed later

        Path("src/artifacts").mkdir(parents=True, exist_ok=True)
        # Saving 'omega_list' here is still useful if you want to use the
        # "Syrota basis" representation for something else.
        # But for 'optim' mode, we'll directly optimize 'cr.tau'.
        torch.save({"knots_list":[knots],"omega_list":[omega]},
                   "src/artifacts/spline_init_syrota.pt")
        print("initial ω stored  –  shape", omega.shape)

        # tiny demo picture for 'init' mode
        Path("src/plots").mkdir(parents=True, exist_ok=True)
        uu = torch.linspace(0,1,400)
        with torch.no_grad(): curve = cr(uu).numpy()
        plt.figure(figsize=(4,4))
        plt.plot(path[:,0], path[:,1], "g--", lw=1, label="raw path")
        plt.scatter(knots[:,0], knots[:,1], c="b", s=12, label="knots")
        plt.plot(curve[:,0], curve[:,1], "r-", lw=2, label="Catmull–Rom")
        plt.axis("equal"); plt.legend(); plt.tight_layout()
        plt.savefig("src/plots/syrota_spline_demo.png", dpi=300); plt.close()

    # ------------------------- mode optim --------------------------------
    else:
        optimise_one(idx=args.idx,
                     n_seg=args.segments,
                     steps=args.steps,
                     lr=args.lr)

if __name__ == "__main__":
    main()