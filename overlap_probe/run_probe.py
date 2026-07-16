"""
overlap_probe.run_probe — THE experiment.

For each held-out home: build the dense and sparse (one-pano-per-room) scenes, run each
requested model on both, score poses against ZInD GT, and aggregate. The headline result is
a single table/plot: per model, pose error DENSE vs SPARSE. If a model's error stays low on
SPARSE, it solves our setting (and our door pipeline is obsolete). If it blows up on SPARSE
while staying low on DENSE, we have a real, quantified gap to build on.

Validate the harness itself (no external models needed):
  python overlap_probe/run_probe.py --root $ZIND_ROOT --only scripts/depth_homes.txt \
      --models oracle,noisy --limit 5
  -> oracle: ate_norm~0, relrot~0 ;  noisy: small but non-zero, and sparse>=dense.

Real run (after wiring adapters + setting ARGUS_DIR / PANOVGGT_DIR / VGGT_DIR):
  python overlap_probe/run_probe.py --root $ZIND_ROOT --only scripts/depth_homes.txt \
      --models argus,panovggt,vggt_tiled
"""
import os, sys, argparse, csv, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from overlap_probe import common, overlap as ov, metrics as M, adapters as A


def run(args):
    homes = common.iter_homes(args.root, only=args.only, limit=args.limit)
    models = [A.make(n) for n in args.models.split(",")]
    for m in models:
        if not m.available:
            print(f"[skip] model '{m.name}' unavailable "
                  f"(set its *_DIR env / wire adapter). Harness will still run others.")
    models = [m for m in models if m.available]
    if not models:
        sys.exit("no available models — try --models oracle,noisy to validate the harness.")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rows, strat_rows = [], []
    tmp_root = Path(tempfile.mkdtemp(prefix="overlap_probe_"))

    for hi, home in enumerate(homes):
        sc = common.build_scenes(home, min_rooms=args.min_rooms)
        fl = sc["fl"]
        if fl is None:
            continue
        for regime in ("dense", "sparse"):
            scene = sc[regime]
            if scene is None:
                continue
            ov.annotate(scene, fl)
            cc = ov.category_counts(scene)
            for m in models:
                wd = tmp_root / f"{home.name}_{regime}_{m.name}"
                pred = m.predict(scene, wd)
                if not pred.ok:
                    print(f"{home.name}/{regime}/{m.name}: {pred.note}")
                    continue
                mt = M.eval_poses(pred.poses_c2w, scene.gt_c2w)
                row = dict(home=home.name, floor=scene.floor, regime=regime, model=m.name,
                           n=scene.n, rooms=len(set(scene.rooms)),
                           n_same=cc["same"], n_adj=cc["adjacent"], n_far=cc["far"],
                           ate_m=round(mt["ate_m"], 4), ate_norm=round(mt["ate_norm"], 4),
                           relrot_med=round(mt["relrot_med_deg"], 3),
                           sim3_scale=round(mt["sim3_scale"], 4),
                           logscale_abs=round(mt["logscale_abs"], 4))
                rows.append(row)
                st = M.stratify_relrot(mt, scene)
                for cat, (val, k) in st.items():
                    if k:
                        strat_rows.append(dict(home=home.name, regime=regime, model=m.name,
                                               overlap=cat, relrot_med=round(val, 3), n_pairs=k))
        if (hi + 1) % 10 == 0:
            print(f"...{hi+1}/{len(homes)} homes")

    if not rows:
        sys.exit("no scenes evaluated (check --root / --min_rooms / homes have panos).")

    # ---- write CSVs
    with open(out_dir / "probe_scenes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if strat_rows:
        with open(out_dir / "probe_overlap_strata.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(strat_rows[0].keys()))
            w.writeheader(); w.writerows(strat_rows)

    # ---- aggregate + print headline table
    print("\n" + "=" * 78)
    print(f"{'model':12} {'regime':7} {'#sc':>4} {'ATEnorm med':>12} {'relRot med':>11} "
          f"{'|log s| med':>11}")
    print("-" * 78)
    agg = {}
    for m in sorted({r["model"] for r in rows}):
        for regime in ("dense", "sparse"):
            sub = [r for r in rows if r["model"] == m and r["regime"] == regime]
            if not sub:
                continue
            an = float(np.median([r["ate_norm"] for r in sub]))
            rr = float(np.median([r["relrot_med"] for r in sub]))
            ls = float(np.median([r["logscale_abs"] for r in sub]))
            agg[(m, regime)] = (an, rr, ls, len(sub))
            print(f"{m:12} {regime:7} {len(sub):4d} {an:12.3f} {rr:11.2f} {ls:11.3f}")
    print("=" * 78)
    print("READ: for each model, sparse vs dense. A model that SOLVES our setting keeps "
          "ATEnorm/relRot low on SPARSE. A large sparse>>dense gap = the room for improvement.")

    _plots(rows, strat_rows, out_dir)
    print(f"\nsaved CSVs + plots to {out_dir}")


def _plots(rows, strat_rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({r["model"] for r in rows})
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for ci, metric in enumerate(["ate_norm", "relrot_med"]):
        for k, regime in enumerate(("dense", "sparse")):
            vals = [np.median([r[metric] for r in rows
                               if r["model"] == m and r["regime"] == regime] or [np.nan])
                    for m in models]
            ax[ci].bar(x + (k - 0.5) * w, vals, w, label=regime)
        ax[ci].set_xticks(x); ax[ci].set_xticklabels(models, rotation=20)
        ax[ci].set_title("normalized ATE (lower=better)" if ci == 0
                         else "median relative-rotation err (deg)")
        ax[ci].legend()
    fig.suptitle("Feed-forward pose accuracy: DENSE vs SPARSE (one-pano-per-room)")
    fig.tight_layout(); fig.savefig(out_dir / "probe_dense_vs_sparse.png", dpi=120)
    plt.close(fig)

    if strat_rows:
        cats = ["same", "adjacent", "far"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for m in models:
            ys = [np.median([r["relrot_med"] for r in strat_rows
                             if r["model"] == m and r["overlap"] == c] or [np.nan]) for c in cats]
            ax.plot(cats, ys, "o-", label=m)
        ax.set_xlabel("GT overlap category (high -> zero)")
        ax.set_ylabel("median relative-rotation err (deg)")
        ax.set_title("Pose error vs cross-view overlap")
        ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "probe_error_vs_overlap.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ZInD full_dataset root ($ZIND_ROOT)")
    ap.add_argument("--only", default=None, help="file of home ids OR comma list")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min_rooms", type=int, default=3)
    ap.add_argument("--models", default="oracle,noisy",
                    help="comma: oracle,noisy,argus,panovggt,vggt_tiled")
    ap.add_argument("--out", default="results/overlap_probe")
    run(ap.parse_args())
