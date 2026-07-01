"""
exp19 (E0) — Per-room Gaussian-splat init from ONE panorama + metric depth.

Builds 3D Gaussians from a single pano's depth point cloud, exports a 3DGS-format
.ply (to seed a GS optimizer/viewer), and runs a CPU SANITY CHECK:
  - reproject the Gaussians to the SAME view  -> should reproduce the input pano (high PSNR);
  - reproject to a small-baseline view         -> shows disocclusion HOLES (the single-view
                                                  limitation that GS optimization must fill).
The differentiable GS optimization (CUDA rasterizer) is isolated in `optimize_room_gs`.

  python experiments/exp19_gs_room.py                       # sample tour, first pano
  python experiments/exp19_gs_room.py --stem <pano_stem>
  python experiments/exp19_gs_room.py --home 0025 --depth_dir .../depth_meters --stem <stem>
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import providers, gsplat_init as gsi


def get_provider(a):
    if a.home:
        from experiments.exp18_floor_graph_real import make_provider
        return make_provider(a.home, a.depth_dir, a.floor)
    return providers.default_zind()


def load_pano(prov, stem, hw):
    im = cv2.imread(str(prov.pano_dir / f"{stem}.jpg"))
    return cv2.cvtColor(cv2.resize(im, (hw[1], hw[0])), cv2.COLOR_BGR2RGB)


def optimize_room_gs(ply_path, pano_rgb, depth, iters=3000, device="cuda"):
    """HOOK (GPU): refine the init by differentiable rendering against the pano.
    Single view memorizes -> use depth regularization + (later) cross-room doorway
    views (E1+). Requires a Gaussian rasterizer (gsplat / diff-gaussian-rasterization)."""
    raise NotImplementedError("GS optimization: run on a GPU machine with a Gaussian rasterizer")


def main(a):
    prov = get_provider(a)
    H, W = a.res, a.res * 2
    names = [n for n in prov.fl.panos if (prov.depth_dir / f"{n}.npy").exists()]
    stem = a.stem or names[0]
    depth = cv2.resize(prov.depth(stem), (W, H), interpolation=cv2.INTER_NEAREST)
    rgb = load_pano(prov, stem, (H, W))

    g = gsi.gaussian_init_from_pano(depth, rgb, stride=a.stride, max_depth=a.max_depth, scale_mult=a.gscale)
    n = len(g["xyz"])

    same, m_same = gsi.render_equirect(g, H, W)
    ps = gsi.psnr(same, rgb, m_same)
    cov = float(m_same.mean())
    nov, m_nov = gsi.render_equirect(g, H, W, t=np.array([a.baseline, 0.0, 0.0]))
    cov_nov = float(m_nov.mean())
    print(f"room {stem}: {n:,} gaussians")
    print(f"  same-view reproject PSNR {ps:.1f} dB  (coverage {cov*100:.0f}%)")
    print(f"  novel-view (+{a.baseline} m) coverage {cov_nov*100:.0f}%  "
          f"-> {(cov-cov_nov)*100:.0f}% of pixels disoccluded (holes GS must fill)")

    out = config.RESULTS_ROOT / "gs"; out.mkdir(parents=True, exist_ok=True)
    gsi.write_gs_ply(out / f"{stem}_gs_init.ply", g)
    gsi.write_point_ply(out / f"{stem}_points.ply", g)
    print(f"  wrote {out/(stem+'_gs_init.ply')}  (3DGS format)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for x, (im, ttl) in zip(ax, [(rgb, "input pano"),
                                 (same, f"reproject same view  (PSNR {ps:.1f} dB)"),
                                 (nov, f"novel view +{a.baseline} m  (holes = disocclusion)")]):
        x.imshow(im); x.set_title(ttl, fontsize=10); x.axis("off")
    fig.suptitle(f"E0: per-room Gaussian init from one panorama — {stem}  ({n:,} gaussians)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out / f"{stem}_gs_init.png"; fig.savefig(p, dpi=110); print("  saved", p)

    if a.optimize:
        from src import gs_optim
        I = np.eye(4, dtype=np.float32)
        gs_optim.convention_check(g, [rgb], [I], opengl=a.opengl, device=a.device)
        g2 = gs_optim.optimize(g, [rgb], [I], iters=a.iters, opengl=a.opengl, device=a.device)
        gsi.write_gs_ply(out / f"{stem}_gs_opt.ply", g2)
        gsi.write_point_ply(out / f"{stem}_gs_opt_points.ply", g2)     # colored, for Open3D
        print("  wrote optimized", out / f"{stem}_gs_opt.ply", "(+ _points.ply)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=None)
    ap.add_argument("--depth_dir", default=None)
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--res", type=int, default=512, help="pano height (width=2*res)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_depth", type=float, default=12.0)
    ap.add_argument("--baseline", type=float, default=0.5, help="novel-view shift (m)")
    ap.add_argument("--optimize", action="store_true", help="run GPU GS optimization")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--opengl", action="store_true", help="flip camera convention if convention_check PSNR is low")
    ap.add_argument("--gscale", type=float, default=1.5, help="init Gaussian scale multiplier (larger = more opaque walls, less bleed)")
    a = ap.parse_args()
    main(a)
