"""Summarize the SALVe-vs-oracle experiment from results.csv.

- Acceptance gate: every `salve_gtcheck` row must have pose_rmse~0 AND door_gap~0. Floors that fail the
  gate are excluded from conclusions (their convention/export didn't round-trip).
- Reports the distribution of the REAL SALVe (`salve_est`) errors vs the oracle across floors, and
  writes a per-floor table + a figure.

    python scripts/analyze_salve_experiment.py --csv results/floor/results.csv --out results/floor
"""
import argparse, csv, math
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--gate_m", type=float, default=0.05, help="acceptance tolerance for _gtcheck (m/deg)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    by = defaultdict(dict)                       # (home,floor) -> {tag: row}
    for r in rows:
        if r.get("tag") in ("salve_gtcheck", "salve_est", "oracle"):
            by[(r["home"], r["floor"])][r["tag"]] = r

    passed, failed, table = [], [], []
    for (h, f), d in sorted(by.items()):
        if not {"salve_gtcheck", "salve_est"} <= set(d):
            continue
        gc = d["salve_gtcheck"]
        gate_ok = fnum(gc["pose_rmse_m"]) <= a.gate_m and (
            math.isnan(fnum(gc["door_gap_m"])) or fnum(gc["door_gap_m"]) <= a.gate_m)
        est = d["salve_est"]; orc = d.get("oracle", {})
        loc = f"{est.get('n_rooms','?')}/{orc.get('n_rooms','?')}"
        rec = dict(home=h, floor=f, gate="ok" if gate_ok else "FAIL",
                   loc_rooms=loc, pose_rmse=fnum(est["pose_rmse_m"]), rot=fnum(est["rot_err_deg"]),
                   door=fnum(est["door_gap_m"]), n_doors=est.get("n_doors", "?"))
        table.append(rec)
        (passed if gate_ok else failed).append(rec)

    print(f"{'building':9s} {'gate':5s} {'loc(rooms)':10s} {'pose_rmse':>9s} {'rot_err':>8s} {'door_gap':>9s} {'n_doors':>7s}")
    for r in table:
        print(f"{r['home']:9s} {r['gate']:5s} {r['loc_rooms']:10s} {r['pose_rmse']:9.3f} {r['rot']:8.2f} "
              f"{r['door']:9.3f} {str(r['n_doors']):>7s}")

    def agg(vals):
        v = np.array([x for x in vals if not math.isnan(x)])
        return (float(np.median(v)), float(np.mean(v)), len(v)) if len(v) else (float('nan'), float('nan'), 0)
    dr = [r for r in passed]
    print(f"\n=== SALVe (layout-only) vs oracle over {len(dr)} gate-passing floors "
          f"({len(failed)} failed gate) ===")
    for key, lbl in [("pose_rmse", "pose RMSE (m)"), ("rot", "rot err (deg)"), ("door", "door_gap (m)")]:
        med, mean, n = agg([r[key] for r in dr])
        print(f"  {lbl:16s}: median {med:.3f}  mean {mean:.3f}  (n={n})   [oracle = 0]")

    # figure: per-floor door_gap and rot_err for gate-passing floors
    dr2 = [r for r in dr if not math.isnan(r["door"])]
    if dr2:
        dr2.sort(key=lambda r: r["door"])
        x = np.arange(len(dr2)); labels = [r["home"] for r in dr2]
        fig, ax1 = plt.subplots(figsize=(max(6, len(dr2) * 0.6), 4.2))
        ax1.bar(x - 0.2, [r["door"] for r in dr2], 0.4, color="#a6161a", label="door_gap (m)")
        ax1.set_ylabel("door_gap (m)", color="#a6161a"); ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, fontsize=7)
        ax1.axhline(0, color="#1f4e79", lw=1, ls=":", label="oracle")
        ax2 = ax1.twinx(); ax2.bar(x + 0.2, [r["rot"] for r in dr2], 0.4, color="#e08214", label="rot err (deg)")
        ax2.set_ylabel("rot err (deg)", color="#e08214")
        ax1.set_title(f"Real SALVe (layout-only) vs GT oracle — {len(dr2)} ZInD test floors")
        fig.tight_layout(); fig.savefig(f"{a.out}/salve_vs_oracle.png", dpi=150)
        print(f"\nwrote {a.out}/salve_vs_oracle.png")
    # per-floor table csv
    with open(f"{a.out}/salve_experiment_summary.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    print(f"wrote {a.out}/salve_experiment_summary.csv")


if __name__ == "__main__":
    main()
