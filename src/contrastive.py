"""
Contrastive cross-view door embedding.

Encoder = (frozen DINOv2 backbone) + (trainable projection head) -> 128-D unit vector.
Trained so the SAME door from two rooms (positive pair) is close and different doors
are far, via symmetric InfoNCE (CLIP-style). Frozen backbone keeps it light on GPU;
unfreeze=True fine-tunes the last blocks if you have headroom.

Only the torch parts touch the GPU; info_nce_np mirrors the loss for CPU/unit tests.
"""
import numpy as np


# ---- loss math (numpy mirror, unit-tested without torch) ----
def info_nce_np(za, zb, tau=0.07):
    za = za / (np.linalg.norm(za, axis=1, keepdims=True) + 1e-8)
    zb = zb / (np.linalg.norm(zb, axis=1, keepdims=True) + 1e-8)
    logits = za @ zb.T / tau
    B = len(za)
    def ce(L):
        L = L - L.max(1, keepdims=True)
        p = np.exp(L) / np.exp(L).sum(1, keepdims=True)
        return -np.mean(np.log(p[np.arange(B), np.arange(B)] + 1e-12))
    return 0.5 * (ce(logits) + ce(logits.T))


# ---- torch model / data / train (run on GPU) ----
def build_encoder(device="cuda", backbone="dinov2_vits14", out_dim=128, unfreeze=False):
    import os, torch, torch.nn as nn
    repo = os.environ.get("DINOV2_REPO")
    if repo and os.path.isdir(repo):                 # offline clusters: local clone, no GitHub
        bb = torch.hub.load(repo, backbone, source="local")
    else:
        bb = torch.hub.load("facebookresearch/dinov2", backbone)
    feat_dim = bb.embed_dim
    for p in bb.parameters():
        p.requires_grad = unfreeze

    class DoorEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = bb
            self.head = nn.Sequential(nn.Linear(feat_dim, 512), nn.GELU(),
                                      nn.Linear(512, out_dim))

        def forward(self, x):
            f = self.bb(x)
            z = self.head(f)
            return z / (z.norm(dim=1, keepdim=True) + 1e-8)

    return DoorEncoder().to(device)


def info_nce(za, zb, tau=0.07):
    import torch, torch.nn.functional as F
    logits = za @ zb.t() / tau
    labels = torch.arange(len(za), device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class DoorPairDataset:
    """Reads pairs.csv (+ crops/) -> (aug(crop_a), aug(crop_b)) tensors. Two
    different augmentations stand in for extra viewpoint variation. NO horizontal
    flip (would swap door left/right, which carries the which-side cue)."""
    def __init__(self, root, img=224, train=True, rows=None):
        import csv
        from pathlib import Path
        self.root = Path(root)
        self.rows = rows if rows is not None else list(csv.DictReader(open(self.root / "pairs.csv")))
        from torchvision import transforms as T
        base = [T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        if train:
            self.tf = T.Compose([T.RandomResizedCrop(img, scale=(0.6, 1.0)),
                                 T.ColorJitter(0.3, 0.3, 0.3, 0.1)] + base)
        else:
            self.tf = T.Compose([T.Resize((img, img))] + base)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        r = self.rows[i]
        a = Image.open(self.root / "crops" / r["crop_a"]).convert("RGB")
        b = Image.open(self.root / "crops" / r["crop_b"]).convert("RGB")
        return self.tf(a), self.tf(b)


def train(root, device="cuda", epochs=40, bs=64, lr=3e-4, out="door_encoder.pt"):
    import torch
    from torch.utils.data import DataLoader
    ds = DoorPairDataset(root, train=True)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4)
    enc = build_encoder(device)
    opt = torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad], lr=lr)
    enc.train()
    for ep in range(epochs):
        tot = 0.0
        for a, b in dl:
            a, b = a.to(device), b.to(device)
            za, zb = enc(a), enc(b)
            loss = info_nce(za, zb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"epoch {ep+1}/{epochs}  loss {tot/max(len(dl),1):.4f}")
    torch.save(enc.state_dict(), out)
    print("saved", out)
    return enc


def load_embedder(ckpt, device="cuda", img=224, backbone="dinov2_vits14"):
    """Return embed(crop_rgb_uint8)->np.vector using a trained DoorEncoder ckpt.
    Drop-in for DoorMatcher(embed=...) and the which-side scorer."""
    import torch
    from torchvision import transforms as T
    enc = build_encoder(device, backbone=backbone)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:        # full training checkpoint
        sd = sd["model"]
    enc.load_state_dict(sd); enc.eval()
    tf = T.Compose([T.ToPILImage(), T.Resize((img, img)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def embed(crop_rgb):
        import numpy as np
        x = tf(np.asarray(crop_rgb, dtype="uint8"))[None].to(device)
        with torch.no_grad():
            return enc(x)[0].cpu().numpy()
    return embed
