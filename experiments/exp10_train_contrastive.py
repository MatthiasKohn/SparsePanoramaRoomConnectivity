"""
exp10 — Train + evaluate the contrastive cross-view door embedding.

Scene-disjoint split (no home in both train and val). Train with symmetric InfoNCE,
then evaluate CROSS-VIEW RETRIEVAL on held-out homes: for each val door, is its
opposite-side crop the nearest neighbour among all val doors? (Top-1 / Top-5).
This directly measures the matching ability that the generic DINOv2 (60% which-side)
lacked. The trained encoder then drops into DoorMatcher/exp08 as `_embed`.

  python experiments/exp10_train_contrastive.py --data data_doorpairs --epochs 40
  python experiments/exp10_train_contrastive.py --data data_doorpairs --eval_only --ckpt door_encoder.pt
"""
import sys, os, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np

from src import contrastive


def split_scenes(root, val_frac=0.2, seed=0):
    rows = list(csv.DictReader(open(Path(root) / "pairs.csv")))
    scenes = sorted({r["scene"] for r in rows})
    rng = np.random.default_rng(seed); rng.shuffle(scenes)
    nval = max(1, int(len(scenes) * val_frac))
    val = set(scenes[:nval])
    return [r for r in rows if r["scene"] not in val], [r for r in rows if r["scene"] in val]


def evaluate_retrieval(enc, root, val_rows, device="cuda", img=224):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    crops = Path(root) / "crops"

    def embed(names):
        Z = []
        enc.eval()
        with torch.no_grad():
            for nm in names:
                x = tf(Image.open(crops / nm).convert("RGB"))[None].to(device)
                Z.append(enc(x)[0].cpu().numpy())
        return np.stack(Z)

    Za = embed([r["crop_a"] for r in val_rows])
    Zb = embed([r["crop_b"] for r in val_rows])
    S = Za @ Zb.T                              # (N,N) cosine
    rank = (-S).argsort(1)
    gt = np.arange(len(val_rows))
    top1 = np.mean(rank[:, 0] == gt)
    top5 = np.mean([gt[i] in rank[i, :5] for i in range(len(gt))])
    print(f"[val retrieval over {len(val_rows)} doors]  top1={top1:.2f}  top5={top5:.2f}")
    return top1, top5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ckpt", default="door_encoder.pt")
    ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows, val_rows = split_scenes(a.data)
    print(f"train pairs={len(train_rows)}  val pairs={len(val_rows)}  (scene-disjoint)")

    enc = contrastive.build_encoder(dev)
    if a.eval_only or os.path.exists(a.ckpt) and a.eval_only:
        enc.load_state_dict(torch.load(a.ckpt, map_location=dev))
        evaluate_retrieval(enc, a.data, val_rows, dev); return

    # baseline (untrained head) for reference, then train, then re-eval
    print("baseline (untrained projection head):")
    evaluate_retrieval(enc, a.data, val_rows, dev)
    enc = contrastive.train(a.data, dev, epochs=a.epochs, bs=a.bs, out=a.ckpt)
    print("after training:")
    evaluate_retrieval(enc, a.data, val_rows, dev)


if __name__ == "__main__":
    main()
