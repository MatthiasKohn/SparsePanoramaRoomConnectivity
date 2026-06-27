"""
exp12 — Room-connectivity graph from door matching (the project's core goal).

Predict a connectivity edge between two rooms when one's door matches the other's
(trained-encoder cosine). Compare to GT (rooms sharing a door); report threshold-free
Average Precision + best-F1, and draw the graph on the floor plan.

Single home:   python experiments/exp12_connectivity_graph.py --home <HOME_DIR> --ckpt door_encoder.pt
Many homes:    python experiments/exp12_connectivity_graph.py --root <ZIND_ROOT> --ckpt door_encoder.pt --max 30
               (reports per-home AP and the MEAN AP -- use HELD-OUT homes for the real number)
Self-test:     python experiments/exp12_connectivity_graph.py --home <HOME_DIR> --selftest
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from itertools import combinations
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import zind, panoproj, door_dataset, contrastive


def collect_doors(fl, panos_dir, fov=70, crop=(224, 224)):
    items, cache = [], {}
    n_have = 0
    def rgb(stem):
        if stem not in cache:
            pp = panos_dir / f"{stem}.jpg"
            im = cv2.imread(str(pp))
            cache[stem] = cv2.cvtColor(cv2.resize(im, (4096, 2048)), cv2.COLOR_BGR2RGB) if im is not None else None
        return cache[stem]
    for pano, info in fl.panos.items():
        im = rgb(pano)
        if im is None:
            continue
        n_have += 1
        for (d0, d1) in info["doors_global"]:
            mid = (d0 + d1) / 2
            az = door_dataset.door_azimuth(fl, pano, mid)
            items.append((info["room"], pano, mid, panoproj.e2p(im, az, 0, fov, crop)))
    return items, n_have


def eval_home(home, embed, selftest=False, thr=None, tol=0.15, draw=True):
    home = Path(home); jp = home / "zind_data.json"
    if not jp.exists():
        return None
    floors = list(json.load(open(jp))["merger"].keys())
    fl = floor_used = None
    for f in floors:
        try:
            cand = zind.ZindFloor(jp, floor=f)
            if len(cand.panos) >= 2:
                fl, floor_used = cand, f; break
        except Exception:
            pass
    if fl is None:
        return None
    items, n_have = collect_doors(fl, home / "panos")
    if len(items) < 2:
        return None
    if selftest:
        E = np.array([np.eye(50)[int(round(m[0]*3)) % 50] + np.eye(50)[int(round(m[1]*3)) % 50]
                      for _, _, m, _ in items], float)
    else:
        E = np.stack([embed(c) for _, _, _, c in items])
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8

    rooms = sorted({it[0] for it in items})
    ridx = {r: [k for k, it in enumerate(items) if it[0] == r] for r in rooms}
    rmids = {r: [items[k][2] for k in ridx[r]] for r in rooms}
    def share(ra, rb):
        return any(np.linalg.norm(ma - mb) < tol for ma in rmids[ra] for mb in rmids[rb])
    y_true, y_score, pairs = [], [], []
    for ra, rb in combinations(rooms, 2):
        y_score.append(float((E[ridx[ra]] @ E[ridx[rb]].T).max()))
        y_true.append(int(share(ra, rb))); pairs.append((ra, rb))
    y_true = np.array(y_true); y_score = np.array(y_score); P = int(y_true.sum())
    if P == 0 or len(pairs) == 0:
        return None
    order = np.argsort(-y_score); yt = y_true[order]
    tp = np.cumsum(yt); fp = np.cumsum(1 - yt)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9); bi = int(np.argmax(f1s))
    res = dict(home=home.name, floor=floor_used, panos=n_have, rooms=len(rooms),
               pairs=len(pairs), gt=P, ap=ap, p=float(prec[bi]), r=float(rec[bi]),
               f1=float(f1s[bi]), rand=P / len(pairs))
    if draw:
        t = float(y_score[order][bi]) if thr is None else thr
        pred = y_score >= t
        rpos = {r: np.mean([fl.panos[items[k][1]]["pos"] for k in ridx[r]], axis=0) for r in rooms}
        fig, ax = plt.subplots(figsize=(7, 7))
        for (ra, rb), gt, pr in zip(pairs, y_true, pred):
            if not (gt or pr):
                continue
            xa, ya = rpos[ra]; xb, yb = rpos[rb]
            col = "#2ca02c" if (gt and pr) else ("#d62728" if pr else "#bbbbbb")
            ax.plot([xa, xb], [ya, yb], color=col, lw=2 if (gt and pr) else 1.2,
                    ls="-" if pr else "--")
        for r, (x, y) in rpos.items():
            ax.plot(x, y, "o", color="#333", ms=7)
        ax.set_aspect("equal")
        ax.set_title(f"{home.name}: ROOM connectivity  AP={ap:.2f} F1={f1s[bi]:.2f} "
                     f"(P={prec[bi]:.2f} R={rec[bi]:.2f})")
        out = config.RESULTS_ROOT / "connectivity"; out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"graph_{home.name}.png", dpi=120); plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home")
    ap.add_argument("--root", help="evaluate every home under this dir")
    ap.add_argument("--ckpt", default="door_encoder.pt")
    ap.add_argument("--max", type=int, default=9999)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--only", help="file with home names (e.g. val_homes.txt) to restrict to")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.root:
        homes = sorted({p.parent for p in Path(a.root).glob("**/zind_data.json")})[:a.max]
    elif a.home:
        homes = [Path(a.home)]
    else:
        ap.error("give --home or --root")
    if a.only:
        keep = set(Path(a.only).read_text().split())
        homes = [h for h in homes if h.name in keep]
        print(f"restricted to {len(homes)} held-out homes from {a.only}")

    embed = None
    if not a.selftest:
        import torch
        embed = contrastive.load_embedder(a.ckpt, "cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    print(f"{'home':10} {'rooms':>5} {'pairs':>5} {'gt':>3} {'AP':>5} {'F1':>5} {'P':>5} {'R':>5} {'rand':>5}")
    for h in homes:
        r = eval_home(h, embed, a.selftest, a.thr, draw=(len(homes) <= 5))
        if r is None:
            continue
        rows.append(r)
        print(f"{r['home']:10} {r['rooms']:5d} {r['pairs']:5d} {r['gt']:3d} "
              f"{r['ap']:5.2f} {r['f1']:5.2f} {r['p']:5.2f} {r['r']:5.2f} {r['rand']:5.2f}")
    if rows:
        aps = np.array([x["ap"] for x in rows]); rnd = np.mean([x["rand"] for x in rows])
        print("-" * 60)
        print(f"MEAN AP over {len(rows)} homes = {aps.mean():.3f}  (std {aps.std():.3f}, "
              f"mean random {rnd:.3f})")


if __name__ == "__main__":
    main()
