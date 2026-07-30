"""Headline figure for the pose half of the paper: reconstruction quality vs pose error.

Reads results/floor/results.csv, keeps the pose sweep at a fixed depth model, averages seeds,
and plots PSNR + LPIPS against Umeyama pose RMSE. The GT point (rmse 0) is the oracle upper bound.

    python scripts/plot_sensitivity.py \
        --csv /leonardo_work/.../results/floor/results.csv \
        --depth gt_layout --out results/floor/pose_sensitivity.png
"""
import argparse, csv, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path, depth):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["depth_model"] == depth]
    # average seeds per noise level; drop any nan lpips (pre-fix rows)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = float(r["noise_deg"])
        for k in ("pose_rmse_m", "rot_err_deg", "psnr", "ssim", "lpips", "coverage"):
            v = float(r[k])
            if not math.isnan(v):
                agg[key][k].append(v)
    pts = []
    for deg in sorted(agg):
        m = {k: sum(v) / len(v) for k, v in agg[deg].items() if v}
        m["noise_deg"] = deg
        pts.append(m)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--depth", default="gt_layout")
    ap.add_argument("--out", default="pose_sensitivity.png")
    a = ap.parse_args()
    pts = load(a.csv, a.depth)
    if not pts:
        raise SystemExit(f"no rows with depth_model={a.depth} in {a.csv}")

    x = [p["pose_rmse_m"] for p in pts]
    psnr = [p["psnr"] for p in pts]
    lpips = [p["lpips"] for p in pts]
    cov = [p["coverage"] for p in pts]

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    l1, = ax1.plot(x, psnr, "o-", color="#1f4e79", label="PSNR (dB)")
    ax1.set_xlabel("pose error  (Umeyama camera RMSE, m)")
    ax1.set_ylabel("PSNR (dB)", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, lpips, "s--", color="#a6161a", label="LPIPS (lower better)")
    ax2.set_ylabel("LPIPS", color="#a6161a")
    ax2.tick_params(axis="y", labelcolor="#a6161a")

    # annotate the oracle (leftmost) and worst point
    ax1.annotate("oracle", (x[0], psnr[0]), textcoords="offset points", xytext=(6, 8), fontsize=8)
    for xi, yi, d in zip(x, psnr, [p["noise_deg"] for p in pts]):
        ax1.annotate(f"{d:.0f}°", (xi, yi), textcoords="offset points", xytext=(0, -12),
                     fontsize=7, ha="center", color="#555")

    ax1.set_title(f"Reconstruction vs pose error  (depth={a.depth}, coverage {min(cov):.2f}–{max(cov):.2f})")
    ax1.legend(handles=[l1, l2], loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    # also dump the averaged table
    print("\n deg  rmse_m  rot°   PSNR   SSIM  LPIPS  cov")
    for p in pts:
        print(f"{p['noise_deg']:4.0f} {p['pose_rmse_m']:6.3f} {p['rot_err_deg']:5.1f}  "
              f"{p['psnr']:5.2f}  {p['ssim']:.3f} {p['lpips']:.3f} {p['coverage']:.3f}")


if __name__ == "__main__":
    main()
