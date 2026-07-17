"""
exp10 — Cluster-grade training of the contrastive cross-view door embedding.

Features for long unattended runs:
  - scene-disjoint train/val split, SAVED to <out>/split.json and the held-out
    home list to <out>/val_homes.txt  (feed to exp12 --only for honest connectivity AP);
  - symmetric InfoNCE, AdamW + cosine LR, optional backbone unfreeze;
  - checkpoint every epoch (model+opt+epoch) to <out>/last.pt and RESUME (--resume);
  - best model (by val top5) -> <out>/best.pt and a bare encoder -> <out>/door_encoder.pt
    (the format exp12 / load_embedder expects);
  - periodic held-out RETRIEVAL eval (top1/top5) + per-epoch log to <out>/train_log.csv.

Example:
  python -m pipelines.train_embedding --data data_doorpairs --out runs/full \
      --epochs 60 --bs 128 --lr 3e-4 --eval_every 2 --workers 8 [--unfreeze] [--resume]
"""
import sys, os, argparse, csv, json, random, time
from pathlib import Path
import numpy as np
from sparsepano.doors import contrastive


def read_rows(data):
    return list(csv.DictReader(open(Path(data) / "pairs.csv")))


def scene_split(rows, val_frac, seed):
    scenes = sorted({r["scene"] for r in rows})
    rng = np.random.default_rng(seed); rng.shuffle(scenes)
    nval = max(1, int(len(scenes) * val_frac)); val = set(scenes[:nval])
    tr = [r for r in rows if r["scene"] not in val]
    va = [r for r in rows if r["scene"] in val]
    return tr, va, sorted(set(scenes) - val), sorted(val)


def home_of(scene):                       # "0001_floor_01" -> "0001"
    return scene.split("_floor")[0]


def evaluate_retrieval(enc, data, val_rows, device, img=224, max_n=2000):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    if not val_rows:
        return 0.0, 0.0
    if len(val_rows) > max_n:
        idx = np.random.default_rng(0).choice(len(val_rows), max_n, replace=False)
        val_rows = [val_rows[i] for i in idx]
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    crops = Path(data) / "crops"

    def emb(names):
        enc.eval(); Z = []
        with torch.no_grad():
            for nm in names:
                x = tf(Image.open(crops / nm).convert("RGB"))[None].to(device)
                Z.append(enc(x)[0].cpu().numpy())
        return np.stack(Z)

    Za = emb([r["crop_a"] for r in val_rows]); Zb = emb([r["crop_b"] for r in val_rows])
    Za /= np.linalg.norm(Za, axis=1, keepdims=True) + 1e-8
    Zb /= np.linalg.norm(Zb, axis=1, keepdims=True) + 1e-8
    S = Za @ Zb.T; rank = (-S).argsort(1); gt = np.arange(len(val_rows))
    top1 = float(np.mean(rank[:, 0] == gt))
    top5 = float(np.mean([gt[i] in rank[i, :5] for i in range(len(gt))]))
    return top1, top5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train_full", action="store_true")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hard_neg", action="store_true", help="same-home hard-negative batching")
    ap.add_argument("--homes_per_batch", type=int, default=8)
    ap.add_argument("--backbone_lr", type=float, default=None, help="LR for DINOv2 when --unfreeze (default = lr*0.05)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = read_rows(a.data)
    tr, va, train_scenes, val_scenes = scene_split(rows, a.val_frac, a.seed)
    json.dump({"train": train_scenes, "val": val_scenes}, open(out / "split.json", "w"), indent=1)
    (out / "val_homes.txt").write_text("\n".join(sorted({home_of(s) for s in val_scenes})))
    print(f"[data] {len(rows)} pairs | train {len(tr)} ({len(train_scenes)} sc) | "
          f"val {len(va)} ({len(val_scenes)} sc) -> held-out homes in {out/'val_homes.txt'}")

    ds = contrastive.DoorPairDataset(a.data, img=a.img, train=True, rows=tr)
    if a.hard_neg:
        from sparsepano.doors.hard_neg import HomeBatchSampler
        sampler = HomeBatchSampler(tr, a.bs, a.homes_per_batch, seed=a.seed)
        dl = DataLoader(ds, batch_sampler=sampler, num_workers=a.workers, pin_memory=True)
        print(f"[hard-neg] same-home batching: {len(sampler)} batches, "
              f"{a.homes_per_batch} homes/batch -> in-batch negatives include same-building doors")
    else:
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, drop_last=True,
                        num_workers=a.workers, pin_memory=True)
    enc = contrastive.build_encoder(dev, backbone=a.backbone, unfreeze=a.unfreeze)
    if a.unfreeze:                                   # discriminative LR: keep the pretrained
        bb_lr = a.backbone_lr if a.backbone_lr is not None else a.lr * 0.05   # backbone changes slowly
        opt = torch.optim.AdamW(
            [{"params": [p for p in enc.bb.parameters() if p.requires_grad], "lr": bb_lr},
             {"params": enc.head.parameters(), "lr": a.lr}], weight_decay=a.wd)
        print(f"[unfreeze] backbone lr {bb_lr:.1e}, head lr {a.lr:.1e} "
              f"(low backbone lr avoids washing out DINOv2)")
    else:
        opt = torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    start_ep, best = 0, -1.0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        enc.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start_ep = ck["epoch"] + 1; best = ck.get("best", -1)
        print(f"[resume] from epoch {start_ep} (best top5 {best:.3f})")

    logp = out / "train_log.csv"
    if not logp.exists():
        logp.write_text("epoch,loss,val_top1,val_top5,lr,secs\n")

    for ep in range(start_ep, a.epochs):
        t0 = time.time(); enc.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            za, zb = enc(x), enc(y)
            loss = contrastive.info_nce(za, zb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        loss_ep = tot / max(len(dl), 1)
        t1, t5 = (evaluate_retrieval(enc, a.data, va, dev, a.img)
                  if (ep % a.eval_every == 0 or ep == a.epochs - 1) else (float("nan"),) * 2)
        secs = time.time() - t0
        print(f"epoch {ep+1}/{a.epochs}  loss {loss_ep:.4f}  val_top1 {t1:.3f}  val_top5 {t5:.3f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {secs:.0f}s", flush=True)
        with open(logp, "a") as f:
            f.write(f"{ep},{loss_ep:.4f},{t1:.4f},{t5:.4f},{sched.get_last_lr()[0]:.2e},{secs:.0f}\n")

        torch.save({"model": enc.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "args": vars(a)}, out / "last.pt")
        torch.save(enc.state_dict(), out / "door_encoder.pt")     # exp12 / load_embedder format
        if t5 == t5 and t5 > best:                                 # not NaN and improved
            best = t5
            torch.save(enc.state_dict(), out / "best.pt")
            print(f"  new best val_top5 {best:.3f} -> {out/'best.pt'}")

    print(f"[done] best val_top5 {best:.3f}. Eval connectivity on held-out homes:\n"
          f"  python -m pipelines.connectivity_graph --root <ZIND_ROOT> "
          f"--only {out/'val_homes.txt'} --ckpt {out/'best.pt'}")


if __name__ == "__main__":
    main()
