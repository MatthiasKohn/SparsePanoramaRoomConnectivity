"""
exp32 (M2) — learned camera->door DISTANCE head. The core PaperV2 "drop depth" experiment:
regress camera->door distance from the door CROP alone (no monocular depth at test time).

Substrate: data_floors/ (exp30) — door tokens with GT camera->door distance (gt_dist_m) and
crops. Model: DINOv2 crop feature (init from the contrastive best.pt; frozen or fine-tuned) +
small MLP -> 1 scalar = log-distance. Loss: Huber on log-distance (distances span ~0.2-6 m).
Eval on held-out homes: median / MAE of the predicted distance, to compare against exp31's
DAP-depth door-distance error (the bar to beat).

Runs on GPU and needs NO depth (trains on GT geometry) -> can run now on Leonardo's data_floors.

  python experiments/exp32_train_distance_head.py --data data_floors --panos_root ../data/zind/full_dataset \
      --ckpt runs/hardneg/best.pt --out runs/dist --epochs 20 --bs 128 --unfreeze
  python experiments/exp32_train_distance_head.py --data data_floors --panos_root ../data/zind/full_dataset --selftest
"""
import sys, os, argparse, json, glob, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np


def load_samples(data_dir):
    """Flatten data_floors/*.json -> list of door samples with GT distance + crop reference."""
    S = []
    cdir = Path(data_dir) / "crops"
    for jp in glob.glob(str(Path(data_dir) / "*.json")):
        rec = json.load(open(jp))
        home, floor = rec["home"], rec["floor"]
        for d in rec["doors"]:
            crop = d.get("crop")
            S.append(dict(home=home, floor=floor, pano=d["pano"], bearing=d["bearing_deg"],
                          width_m=d["width_m"], gt=d["gt_dist_m"],
                          crop=(str(cdir / crop) if crop else None)))
    return S


def split_by_home(samples, val_frac, seed, val_homes=None):
    homes = sorted({s["home"] for s in samples})
    if val_homes:
        val = set(Path(val_homes).read_text().split())
    else:
        rng = np.random.default_rng(seed); rng.shuffle(homes)
        val = set(homes[:max(1, int(len(homes) * val_frac))])
    tr = [s for s in samples if s["home"] not in val]
    va = [s for s in samples if s["home"] in val]
    return tr, va


def load_image(s, panos_root, fov, size):
    import cv2
    from src import panoproj
    if s["crop"] and os.path.exists(s["crop"]):
        im = cv2.imread(s["crop"]); return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    p = Path(panos_root) / s["home"] / "panos" / f"{s['pano']}.jpg"     # fallback: e2p from pano
    im = cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (4096, 2048)), cv2.COLOR_BGR2RGB)
    return panoproj.e2p(im, s["bearing"], 0, fov, (size, size))


def metrics(pred, gt):
    e = np.abs(np.asarray(pred) - np.asarray(gt))
    return dict(median=float(np.median(e)), mae=float(e.mean()))


def main(a):
    samples = load_samples(a.data)
    tr, va = split_by_home(samples, a.val_frac, a.seed, a.val_homes)
    gt_va = np.array([s["gt"] for s in va])
    print(f"[data] {len(samples)} doors | train {len(tr)} | val {len(va)} "
          f"({len(set(s['home'] for s in va))} val homes)")
    print(f"[data] val GT distance: median {np.median(gt_va):.2f} m  range {gt_va.min():.2f}-{gt_va.max():.2f}")

    if a.selftest:                                # no torch: validate harness with trivial predictors
        med = np.median([s["gt"] for s in tr])
        print(f"[selftest] baseline 'predict train-median ({med:.2f} m)': "
              f"{metrics(np.full(len(va), med), gt_va)}")
        # sanity: at least one image loads
        try:
            img = load_image(va[0], a.panos_root, a.fov, a.img); print(f"[selftest] image load OK {img.shape}")
        except Exception as e:
            print(f"[selftest] image load FAILED: {e}")
        return

    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms as T
    from src import contrastive
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class DoorDist(Dataset):
        def __init__(self, rows, train):
            self.rows = rows
            base = [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            self.tf = T.Compose(([T.ToPILImage(), T.RandomResizedCrop(a.img, (0.7, 1.0)),
                                  T.ColorJitter(0.2, 0.2, 0.2)] if train else
                                 [T.ToPILImage(), T.Resize((a.img, a.img))]) + base)

        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            s = self.rows[i]
            x = self.tf(load_image(s, a.panos_root, a.fov, a.img).astype("uint8"))
            return x, torch.tensor([np.log(max(s["gt"], 0.05))], dtype=torch.float32)

    bb = contrastive.build_encoder(dev, unfreeze=a.unfreeze)      # DINOv2 + (unused) head
    if a.ckpt and os.path.exists(a.ckpt):
        sd = torch.load(a.ckpt, map_location=dev)
        bb.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd, strict=False)
    feat_dim = bb.bb.embed_dim

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.bb = bb.bb
            self.head = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU(), nn.Linear(256, 1))
        def forward(self, x): return self.head(self.bb(x))

    net = Net().to(dev)
    params = ([{"params": net.bb.parameters(), "lr": a.lr * 0.05}] if a.unfreeze else []) + \
             [{"params": net.head.parameters(), "lr": a.lr}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    dl = DataLoader(DoorDist(tr, True), batch_size=a.bs, shuffle=True, drop_last=True,
                    num_workers=a.workers, pin_memory=True)
    vdl = DataLoader(DoorDist(va, False), batch_size=a.bs, num_workers=a.workers)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(a.epochs):
        net.train(); tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = nn.functional.smooth_l1_loss(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        net.eval(); preds = []
        with torch.no_grad():
            for x, _ in vdl:
                preds.append(np.exp(net(x.to(dev)).cpu().numpy()[:, 0]))
        m = metrics(np.concatenate(preds), gt_va)
        print(f"epoch {ep+1}/{a.epochs}  loss {tot/len(dl):.4f}  val median {m['median']:.2f} m  MAE {m['mae']:.2f} m", flush=True)
        if m["median"] < best:
            best = m["median"]; torch.save(net.state_dict(), out / "dist_head.pt")
    print(f"[done] best val median {best:.2f} m  (compare to exp31 DAP baseline ~0.65 m on 0025)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_floors")
    ap.add_argument("--panos_root", default="../data/zind/full_dataset")
    ap.add_argument("--ckpt", default=None); ap.add_argument("--out", default="runs/dist")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--fov", type=float, default=70.0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.15); ap.add_argument("--val_homes", default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    main(a)
