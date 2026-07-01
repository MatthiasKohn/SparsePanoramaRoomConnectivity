"""
exp23 — Batch held-out evaluation + pitch-ready summary.

Runs the connectivity-AP evaluation (exp12.eval_home) across many HELD-OUT ZInD homes
and aggregates into: mean AP (+/- std) vs random, per-home bars, AP-vs-#rooms, and a CSV.
Connectivity AP needs only panos + the trained encoder (NO depth) -- runnable now.

  python experiments/exp23_heldout_eval.py --root <ZIND_ROOT> \
      --only runs/full_ft/val_homes.txt --ckpt best.pt --max 60
  python experiments/exp23_heldout_eval.py --root ../data/zind/full_dataset --selftest --max 20
"""
import sys, os, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from experiments.exp12_connectivity_graph import eval_home


def main(a):
    homes = sorted({p.parent for p in Path(a.root).glob("**/zind_data.json")})
    if a.only:
        if not Path(a.only).exists():
            raise SystemExit(f"--only file not found: {a.only}\n"
                             "Held-out list missing -> you'd be evaluating on possibly-TRAINING "
                             "homes. Point --only at your run's val_homes.txt, or drop --only to "
                             "deliberately eval every home under --root.")
        keep = set(Path(a.only).read_text().split())
        homes = [h for h in homes if h.name in keep]
        print(f"restricted to {len(homes)} held-out homes from {a.only}")
    homes = homes[:a.max]

    embed = None
    if not a.selftest:
        import torch
        from src import contrastive
        embed = contrastive.load_embedder(a.ckpt, "cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    print(f"{'home':10} {'rooms':>5} {'gt':>3} {'AP':>5} {'F1':>5} {'rand':>5}")
    for h in homes:
        r = eval_home(h, embed, a.selftest, draw=False, mutual=a.mutual)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['home']:10} {r['rooms']:5d} {r['gt']:3d} {r['ap']:5.2f} {r['f1']:5.2f} {r['rand']:5.2f}")
    if not rows:
        print("no evaluable homes"); return

    ap = np.array([x["ap"] for x in rows]); f1 = np.array([x["f1"] for x in rows])
    rnd = np.array([x["rand"] for x in rows]); rooms = np.array([x["rooms"] for x in rows])
    lift = float(np.mean(ap / np.maximum(rnd, 1e-6)))
    print("-" * 62)
    print(f"MEAN AP over {len(rows)} held-out homes = {ap.mean():.3f} +/- {ap.std():.3f}  "
          f"| mean F1 {f1.mean():.3f} | mean random {rnd.mean():.3f} | lift x{lift:.1f}")

    out = config.RESULTS_ROOT / "heldout"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "heldout_ap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    o = np.argsort(-ap)
    ax[0].bar(range(len(ap)), ap[o], color="#1f77b4")
    ax[0].plot(range(len(ap)), rnd[o], "r--", lw=1, label="random")
    ax[0].axhline(ap.mean(), color="k", lw=1, label=f"mean {ap.mean():.2f}")
    ax[0].set_title(f"Held-out connectivity AP per home (n={len(ap)})", fontsize=10)
    ax[0].set_xlabel("home (sorted)"); ax[0].set_ylabel("AP"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1)
    ax[1].scatter(rooms, ap, c="#1f77b4", s=25)
    ax[1].set_title("AP vs #rooms", fontsize=10); ax[1].set_xlabel("#rooms"); ax[1].set_ylabel("AP")
    ax[1].grid(alpha=.3); ax[1].set_ylim(0, 1)
    ax[2].hist(ap, bins=12, range=(0, 1), color="#1f77b4", alpha=.85)
    ax[2].axvline(ap.mean(), color="k", lw=1); ax[2].axvline(rnd.mean(), color="r", ls="--", lw=1)
    ax[2].set_title("AP distribution", fontsize=10); ax[2].set_xlabel("AP")
    fig.suptitle(f"Held-out ZInD connectivity — mean AP {ap.mean():.3f} (random {rnd.mean():.3f}, "
                 f"x{lift:.1f} lift)" + ("  [SELFTEST]" if a.selftest else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = out / ("heldout_summary" + ("_selftest" if a.selftest else "") + ".png")
    fig.savefig(p, dpi=120); print("saved", p, "and", out / "heldout_ap.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--only"); ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutual", action="store_true", help="mutual-NN edge scoring")
    a = ap.parse_args()
    main(a)
