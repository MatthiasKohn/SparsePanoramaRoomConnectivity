"""
exp28 — GT-FREE connectivity: semantic door DETECTOR replaces GT door locations.

Closes the paper's biggest honesty gap (NextStage.md G1): every connectivity number so
far cropped doors at GT-annotated azimuths. Here the front end is
sparsepano/doors/door_semantic.SemanticDoorDetector (SegFormer ring views, RGB only, no depth,
no GT) -> door azimuths -> e2p crops -> trained embedding -> room-connectivity AP,
scored against GT connectivity EXACTLY as exp12 (GT used only for labels).

Also reports detector quality (P/R of door azimuths @ --tol_deg vs GT) so the
AP(GT doors) vs AP(detected doors) gap decomposes into detection vs matching.

Detections are cached per home (json) so re-runs with different scoring are free.

  # headline (GPU):
  python -m pipelines.connectivity --root <ZIND_ROOT> \
      --only runs/hardneg/val_homes.txt --ckpt runs/hardneg/best.pt \
      --scoring assign --max 200 --tag hardneg_det

  # oracle-door control on the same homes (equivalent to exp23; run for the paired table):
  python -m pipelines.connectivity ... --doors gt --tag hardneg_gt

  # CPU plumbing self-test (fake segmenter, fake embedder):
  python -m pipelines.connectivity --home <HOME_DIR> --selftest
"""
import sys, os, argparse, csv, json
from pathlib import Path
from itertools import combinations
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.datasets import zind
from sparsepano.geometry import panoproj
from sparsepano.doors import door_dataset


def circ_diff(a, b):
    return abs(((a - b + 180) % 360) - 180)


def load_floor(home):
    """First floor with >=2 panos (same policy as exp12.eval_home)."""
    jp = Path(home) / "zind_data.json"
    if not jp.exists():
        return None, None
    for f in json.load(open(jp))["merger"].keys():
        try:
            cand = zind.ZindFloor(jp, floor=f)
            if len(cand.panos) >= 2:
                return cand, f
        except Exception:
            pass
    return None, None


def pano_rgb(home, stem, cache):
    if stem not in cache:
        im = cv2.imread(str(Path(home) / "panos" / f"{stem}.jpg"))
        cache[stem] = cv2.cvtColor(cv2.resize(im, (4096, 2048)), cv2.COLOR_BGR2RGB) \
            if im is not None else None
    return cache[stem]


def detect_home(home, fl, detector_factory, cache_dir, include_windows=False):
    """Run (or load cached) semantic door detection for every pano of the floor.
    Returns {stem: [azimuth_deg, ...]}."""
    cpath = Path(cache_dir) / f"{Path(home).name}_det.json" if cache_dir else None
    if cpath is not None and cpath.exists():
        return json.load(open(cpath))
    det = detector_factory()
    out, cache = {}, {}
    for stem in fl.panos:
        im = pano_rgb(home, stem, cache)
        if im is None:
            continue
        doors = det.detect(im)
        keep = [d for d in doors if d.category == "door" or
                (include_windows and d.category == "window")]
        out[stem] = [float(d.azimuth_deg) for d in keep]
    if cpath is not None:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(cpath, "w"))
    return out


def gt_door_az(fl, stem):
    """GT door azimuths (deg) of one pano, via the calibrated ZInD convention."""
    return [door_dataset.door_azimuth(fl, stem, (d0 + d1) / 2)
            for d0, d1 in fl.panos[stem]["doors_global"]]


def detector_pr(fl, det_az, tol_deg):
    """Micro precision/recall of detected azimuths vs GT azimuths over the floor."""
    tp = fp = fn = 0
    for stem, dets in det_az.items():
        gts = gt_door_az(fl, stem)
        used = set()
        for a in dets:
            hit = None
            for k, g in enumerate(gts):
                if k not in used and circ_diff(a, g) < tol_deg:
                    hit = k; break
            if hit is None:
                fp += 1
            else:
                used.add(hit); tp += 1
        fn += len(gts) - len(used)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, tp, fp, fn


def score_connectivity(items, E, fl, scoring="assign", tol=0.15):
    """items: (room, stem, az_deg) per door crop; E: (N,D) embeddings.
    Same room-pair AP machinery as exp12.eval_home (GT only for labels)."""
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    rooms = sorted({it[0] for it in items})
    ridx = {r: [k for k, it in enumerate(items) if it[0] == r] for r in rooms}
    # GT room adjacency from GT door midpoints (labels only)
    rmids = {r: [] for r in rooms}
    for stem, info in fl.panos.items():
        for d0, d1 in info["doors_global"]:
            rmids[info["room"]].append((d0 + d1) / 2)
    def share(ra, rb):
        return any(np.linalg.norm(ma - mb) < tol for ma in rmids[ra] for mb in rmids[rb])

    S = E @ E.T
    room_of = [it[0] for it in items]
    N = len(items)
    match_cos = None
    if scoring == "assign":
        cand = [(i, j) for i in range(N) for j in range(i + 1, N) if room_of[i] != room_of[j]]
        cand.sort(key=lambda ij: -S[ij[0], ij[1]])
        used, match_cos = set(), {}
        for i, j in cand:
            if i in used or j in used:
                continue
            used.add(i); used.add(j)
            key = tuple(sorted((room_of[i], room_of[j])))
            match_cos[key] = max(match_cos.get(key, -1.0), float(S[i, j]))
    y_true, y_score = [], []
    for ra, rb in combinations(rooms, 2):
        ia, ib = ridx[ra], ridx[rb]
        base = float(S[np.ix_(ia, ib)].max()) if ia and ib else -1.0
        if scoring == "assign":
            key = tuple(sorted((ra, rb)))
            sc = (1.0 + match_cos[key]) if match_cos and key in match_cos else base
        else:
            sc = base
        y_true.append(int(share(ra, rb))); y_score.append(sc)
    y_true = np.array(y_true); y_score = np.array(y_score); P = int(y_true.sum())
    if P == 0 or len(y_true) == 0:
        return None
    order = np.argsort(-y_score); yt = y_true[order]
    tp = np.cumsum(yt); fp = np.cumsum(1 - yt)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9); bi = int(np.argmax(f1s))
    return dict(rooms=len(rooms), gt=P, pairs=len(y_true), ap=ap,
                p=float(prec[bi]), r=float(rec[bi]), f1=float(f1s[bi]),
                rand=P / len(y_true))


def eval_home(home, embed, detector_factory, a):
    fl, floor = load_floor(home)
    if fl is None:
        return None
    cache = {}
    if a.doors == "gt":
        det_az = {stem: gt_door_az(fl, stem) for stem in fl.panos}
        pr = (1.0, 1.0, 0, 0, 0)
    else:
        det_az = detect_home(home, fl, detector_factory, a.det_cache, a.include_windows)
        pr = detector_pr(fl, det_az, a.tol_deg)

    items, crops = [], []
    for stem, azs in det_az.items():
        if stem not in fl.panos:
            continue
        im = pano_rgb(home, stem, cache)
        if im is None:
            continue
        for az in azs:
            items.append((fl.panos[stem]["room"], stem, az))
            crops.append(panoproj.e2p(im, az, 0, a.fov, (224, 224)))
    if len(items) < 2:
        return None
    if a.selftest:  # deterministic fake embedding keyed on GT room adjacency-ish position
        E = np.array([np.concatenate([np.eye(8)[hash(it[0]) % 8],
                                      [np.cos(np.radians(it[2])), np.sin(np.radians(it[2]))]])
                      for it in items], float)
    else:
        E = np.stack([embed(c) for c in crops])
    res = score_connectivity(items, E, fl, scoring=a.scoring)
    if res is None:
        return None
    res.update(home=Path(home).name, floor=floor, doors=len(items),
               det_p=pr[0], det_r=pr[1])
    return res


def main(a):
    if a.home:
        homes = [Path(a.home)]
    else:
        homes = sorted({p.parent for p in Path(a.root).glob("**/zind_data.json")})
    if a.only:
        keep = set(Path(a.only).read_text().split())
        homes = [h for h in homes if h.name in keep]
        print(f"restricted to {len(homes)} held-out homes from {a.only}")
    homes = homes[:a.max]

    embed = detector_factory = None
    if not a.selftest:
        import torch
        from sparsepano.doors import contrastive
        dev = a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu")
        embed = contrastive.load_embedder(a.ckpt, dev)
        if a.doors == "detected":
            from sparsepano.doors.door_semantic import SemanticDoorDetector
            detector_factory = lambda: SemanticDoorDetector(
                device=dev, n_views=a.n_views, fov_deg=a.det_fov)
    elif a.doors == "detected":
        # selftest fake segmenter: mark a vertical band as 'door' id 14
        from sparsepano.doors.door_semantic import SemanticDoorDetector
        def fake_seg(img):
            lab = np.zeros(img.shape[:2], np.int64)
            lab[img.shape[0] // 3:, : img.shape[1] // 5] = 14
            return lab
        detector_factory = lambda: SemanticDoorDetector(segmenter=fake_seg)

    rows = []
    print(f"{'home':10} {'rooms':>5} {'doors':>5} {'gt':>3} {'AP':>5} {'F1':>5} "
          f"{'detP':>5} {'detR':>5} {'rand':>5}")
    for h in homes:
        try:
            r = eval_home(h, embed, detector_factory, a)
        except Exception as e:
            print(f"{Path(h).name:10} FAILED: {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"{r['home']:10} {r['rooms']:5d} {r['doors']:5d} {r['gt']:3d} "
              f"{r['ap']:5.2f} {r['f1']:5.2f} {r['det_p']:5.2f} {r['det_r']:5.2f} {r['rand']:5.2f}")
    if not rows:
        print("no evaluable homes"); return

    ap = np.array([x["ap"] for x in rows]); rnd = np.array([x["rand"] for x in rows])
    dp = np.array([x["det_p"] for x in rows]); dr = np.array([x["det_r"] for x in rows])
    print("-" * 70)
    print(f"[{a.doors} doors, scoring={a.scoring}] MEAN AP over {len(rows)} homes = "
          f"{ap.mean():.3f} +/- {ap.std():.3f} | mean random {rnd.mean():.3f}")
    if a.doors == "detected":
        print(f"detector micro P/R @ {a.tol_deg} deg: {dp.mean():.2f} / {dr.mean():.2f}")

    out = config.RESULTS_ROOT / "gtfree"; out.mkdir(parents=True, exist_ok=True)
    tag = ("_" + a.tag) if a.tag else ""
    with open(out / f"gtfree_ap{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    o = np.argsort(-ap)
    ax[0].bar(range(len(ap)), ap[o], color="#1f77b4")
    ax[0].plot(range(len(ap)), rnd[o], "r--", lw=1, label="random")
    ax[0].axhline(ap.mean(), color="k", lw=1, label=f"mean {ap.mean():.2f}")
    ax[0].set_title(f"GT-free connectivity AP per home ({a.doors} doors)", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1)
    ax[1].scatter(dr, ap, c="#1f77b4", s=25)
    ax[1].set_xlabel("detector recall"); ax[1].set_ylabel("AP")
    ax[1].set_title("AP vs detector recall (does detection cap matching?)", fontsize=10)
    ax[1].grid(alpha=.3); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    fig.suptitle(f"exp28 [{a.doors}] mean AP {ap.mean():.3f} (random {rnd.mean():.3f})"
                 + ("  [SELFTEST]" if a.selftest else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out / f"gtfree_summary{tag}{'_selftest' if a.selftest else ''}.png"
    fig.savefig(p, dpi=120); print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--home")
    ap.add_argument("--only"); ap.add_argument("--ckpt", default="weights/best.pt")
    ap.add_argument("--max", type=int, default=9999)
    ap.add_argument("--doors", choices=["detected", "gt"], default="detected")
    ap.add_argument("--scoring", choices=["max", "assign"], default="assign")
    ap.add_argument("--fov", type=float, default=70.0, help="embedding crop fov")
    ap.add_argument("--det_fov", type=float, default=90.0, help="detector ring-view fov")
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--tol_deg", type=float, default=15.0)
    ap.add_argument("--det_cache", default="results/gtfree/det_cache")
    ap.add_argument("--include_windows", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if not (a.root or a.home):
        ap.error("give --root or --home")
    main(a)
