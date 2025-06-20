# catmull_init_spline.py  –  *smooth* initial curve
# --------------------------------------------------------------------------
# Produces src/plots/syrota_spline_demo.png and prints the parameter vector τ
#
# * Centripetal Catmull-Rom (0.5 power on chord length) ⇒ no overshoot
# * τ  = internal tangent vectors, shape (n_knots-2, 2)
# * Tangents are what you would optimise later instead of ω
# --------------------------------------------------------------------------
import pickle, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
torch.manual_seed(0);  np.random.seed(0)

# ---------- helper ---------------------------------------------------------
def resample_arc(path: np.ndarray, n: int) -> np.ndarray:
    d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.insert(np.cumsum(d), 0, 0.0);  s /= s[-1]
    u = np.linspace(0., 1., n)
    return np.vstack([np.interp(u, s, path[:,k]) for k in range(path.shape[1])]).T

def centripetal_t(p0,p1):           # chord-length-to-power-½
    return (np.linalg.norm(p1-p0)**0.5)

# ---------- spline class ---------------------------------------------------
class CatmullRom(torch.nn.Module):
    """
    knots :  (K,2)  float32        fixed
    tau   :  (K-2,2) float32       learnable tangents inside the curve
    """
    def __init__(self, knots: torch.Tensor):
        super().__init__()
        K = knots.size(0)
        self.register_buffer('p', knots)           # (K,2)
        # initialise internal tangents with finite differences
        init_tau = 0.5*(knots[2:] - knots[:-2])
        self.tau = torch.nn.Parameter(init_tau)    # (K-2,2)

    def forward(self, t: torch.Tensor):
        """
        Catmull-Rom evaluation for arbitrary t∈[0,1] vector.
        """
        K = self.p.size(0)
        # map global t → segment index k and local u in [0,1]
        seg_t = t * (K-1)
        k     = torch.clamp(seg_t.floor(), max=K-2).long()   # 0..K-2
        u     = seg_t - k.float()                           # local

        # endpoints & tangents
        P0 = self.p[k]
        P1 = self.p[k+1]

        # tangent at P0
        m0 = torch.zeros_like(P0)
        mask0 = (k > 0)
        m0[mask0] = self.tau[k[mask0]-1]

        # tangent at P1
        m1 = torch.zeros_like(P1)
        mask1 = (k < K-2)
        m1[mask1] = self.tau[k[mask1]]

        # Hermite basis
        u2 = u*u; u3 = u2*u
        h00 =  2*u3 - 3*u2 + 1
        h10 =      u3 - 2*u2 + u
        h01 = -2*u3 + 3*u2
        h11 =      u3 -   u2

        return (h00[:,None]*P0 + h10[:,None]*m0 +
                h01[:,None]*P1 + h11[:,None]*m1)

# ---------- Syrota null-space basis ----------------------------------------
def construct_basis(n_poly: int) -> torch.Tensor:
    """
    Returns B  ∈  ℝ^{4·n_poly × (4·n_poly-2)}   (float32)
    whose columns span the null-space of the boundary + C²-continuity
    constraints (App. C in Syrota et al.).
    """
    tc   = torch.linspace(0, 1, n_poly + 1, dtype=torch.float32)[1:-1]
    rows = []

    # boundary: S(0)=0 , S(1)=0
    e0, e1 = torch.zeros(4*n_poly), torch.zeros(4*n_poly)
    e0[0] = 1;  e1[-4:] = 1
    rows += [e0, e1]

    # C⁰ / C¹ / C² at internal knots
    for i, t in enumerate(tc):
        s = 4*i
        p = torch.tensor([1, t, t**2, t**3])
        d1= torch.tensor([0, 1, 2*t, 3*t**2])
        d2= torch.tensor([0, 0, 2   , 6*t   ])
        for v in (p, d1, d2):
            r = torch.zeros(4*n_poly)
            r[s:s+4]     =  v
            r[s+4:s+8]   = -v
            rows.append(r)

    C = torch.stack(rows).float()
    _, _, Vh = torch.linalg.svd(C)               # full SVD
    return Vh.T[:, C.size(0):].contiguous()      # null-space (4n, 4n-2)



# ---------- main -----------------------------------------------------------
# if __name__ == '__main__':
#     # 1) raw Dijkstra path (latent-dim = 2)
#     path = pickle.loads(Path('src/artifacts/dijkstra_paths_avae.pkl')
#                         .read_bytes())

#     # 2) choose number of segments  (== n_knots-1)
#     n_seg  = 8
#     knots_np = resample_arc(path, n_seg+1)
#     knots    = torch.tensor(knots_np, dtype=torch.float32)

#     # 3) spline object (τ is the parameter vector you will later optimise)
#     cr_spline = CatmullRom(knots)

#     # 4) dense evaluation for plot
#     u = torch.linspace(0,1,400)
#     with torch.no_grad():
#         curve = cr_spline(u).numpy()

#     # ---------- project Catmull-Rom onto Syrota basis --------------------------
#     n_poly = n_seg                                   # same symbol as in optimiser
#     B      = construct_basis(n_poly)                 # (4n, d_free)

#     # sample your Catmull-Rom at the 4·n_poly “basis abscissae”
#     t_basis = torch.linspace(0, 1, 4*n_poly, dtype=torch.float32)
#     with torch.no_grad():
#         S_full = cr_spline(t_basis)                  # (4n, 2)

#     # subtract the straight line (Syrota’s definition of S(t))
#     line = (1 - t_basis)[:, None]*knots[0] + t_basis[:, None]*knots[-1]
#     S_dev = S_full - line                            # (4n, 2)

#     # least-squares for ω  :   B ω  ≈ S_dev
#     omega = torch.linalg.lstsq(B, S_dev).solution    # (d_free, 2)

#     # save knots + ω  (both are needed later)
#     torch.save({
#         'knots': cr_spline.p,
#         'tau': cr_spline.tau.detach(),
#         'omega': omega
#     }, "src/artifacts/spline_init_syrota.pt")
#     print('→ wrote Syrota params   ω-shape:', omega.shape)


#     # 5) save figure
#     plt.figure(figsize=(4,4))
#     plt.plot(path[:,0], path[:,1],'g--',lw=1,label='raw Dijkstra')
#     plt.scatter(knots_np[:,0],knots_np[:,1],c='b',s=14,label='knots')
#     plt.plot(curve[:,0],curve[:,1],'r-',lw=2,label='Catmull-Rom')
#     plt.axis('equal'); plt.legend(); plt.tight_layout()
#     out = Path('src/plots/syrota_spline_demo.png')
#     out.parent.mkdir(parents=True,exist_ok=True)
#     plt.savefig(out,dpi=300); plt.close()
#     print('saved', out)
#     print('parameter τ shape:', cr_spline.tau.shape)        # (n_seg-1, 2)
if __name__ == '__main__':
    # Load multiple paths (assumed list of arrays)
    all_paths = pickle.loads(Path('src/artifacts/dijkstra_paths_avae.pkl').read_bytes())

    n_seg = 8
    out_dir = Path("src/artifacts/spline_inits")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(all_paths):
        knots_np = resample_arc(path, n_seg+1)
        knots = torch.tensor(knots_np, dtype=torch.float32)

        cr_spline = CatmullRom(knots)

        # Sample for projection onto Syrota basis
        t_basis = torch.linspace(0, 1, 4*n_seg, dtype=torch.float32)
        with torch.no_grad():
            S_full = cr_spline(t_basis)

        line = (1 - t_basis)[:, None]*knots[0] + t_basis[:, None]*knots[-1]
        S_dev = S_full - line

        B = construct_basis(n_seg)
        omega = torch.linalg.lstsq(B, S_dev).solution

        # Save to separate file
        torch.save({
            'knots': cr_spline.p,
            'tau': cr_spline.tau.detach(),
            'omega': omega
        }, out_dir / f"spline_init_{idx:02d}.pt")
        print(f"✓ Saved spline_init_{idx:02d}.pt — ω shape: {omega.shape}")

