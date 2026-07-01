"""
exp25 — Diagnose the AP-vs-#rooms degradation from a heldout_ap.csv.

Is large-home failure PRECISION (false edges: confusable doors) or RECALL (missed edges:
occluded/closed doors)? Plots AP/P/R vs #rooms, bins by size, and reports which drops.
Runs anywhere (just needs the CSV exp23 wrote).

  python experiments/exp25_diagnose_scaling.py --csv results/heldout/heldout_ap.csv
"""
import sys, os, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config


def main(a):
    rows = list(csv.DictReader(open(a.csv)))
    rooms = np.array([float(r["rooms"]) for r in rows])
    ap = np.array([float(r["ap"]) for r in rows])
    P = np.array([float(r["p"]) for r in rows]); R = np.array([float(r["r"]) for r in rows])
    rnd = np.array([float(r["rand"]) for r in rows])

    def corr(x, y): return float(np.corrcoef(x, y)[0, 1])
    print(f"n={len(rows)}  mean AP {ap.mean():.3f}  P {P.mean():.3f}  R {R.mean():.3f}")
    print(f"corr(#rooms, AP) = {corr(rooms, ap):+.2f}")
    print(f"corr(#rooms, precision) = {corr(rooms, P):+.2f}")
    print(f"corr(#rooms, recall)    = {corr(rooms, R):+.2f}")
    # small vs large split
    med = np.median(rooms); sm, lg = rooms <= med, rooms > med
    print(f"small homes (<= {med:.0f} rooms): AP {ap[sm].mean():.2f}  P {P[sm].mean():.2f}  R {R[sm].mean():.2f}")
    print(f"large homes (>  {med:.0f} rooms): AP {ap[lg].mean():.2f}  P {P[lg].mean():.2f}  R {R[lg].mean():.2f}")
    dP = P[sm].mean() - P[lg].mean(); dR = R[sm].mean() - R[lg].mean()
    verdict = ("PRECISION-limited (false edges from confusable doors) -> attack with harder "
               "negatives / mutual-NN + ratio test" if dP > dR else
               "RECALL-limited (missed doors: occlusion/closed/depth-invisible) -> attack door "
               "detection / add geometric channel")
    print(f"drop small->large: precision -{dP:.2f}, recall -{dR:.2f}  ==> {verdict}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for x, (y, ttl, c) in zip(ax, [(ap, "AP", "#1f77b4"), (P, "precision@bestF1", "#2ca02c"),
                                   (R, "recall@bestF1", "#d62728")]):
        x.scatter(rooms, y, s=18, c=c, alpha=.6)
        # binned mean
        bins = np.arange(2, rooms.max() + 3, 3)
        bi = np.digitize(rooms, bins)
        bx = [rooms[bi == k].mean() for k in range(1, len(bins)) if (bi == k).sum()]
        by = [y[bi == k].mean() for k in range(1, len(bins)) if (bi == k).sum()]
        x.plot(bx, by, "k-o", lw=2, ms=4)
        x.set_xlabel("#rooms"); x.set_ylabel(ttl); x.set_ylim(0, 1); x.grid(alpha=.3)
        x.set_title(f"{ttl} vs #rooms  (corr {corr(rooms, y):+.2f})", fontsize=10)
    fig.suptitle("Where connectivity degrades with home size", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = config.RESULTS_ROOT / "heldout"; out.mkdir(parents=True, exist_ok=True)
    p = out / "diagnose_scaling.png"; fig.savefig(p, dpi=120); print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/heldout/heldout_ap.csv")
    a = ap.parse_args()
    main(a)
