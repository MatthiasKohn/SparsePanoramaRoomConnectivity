#!/usr/bin/env python3
"""
argus_infer.py — overlap_probe inference shim for Realsee Argus (ECCV'26).

COPY THIS FILE INTO THE ARGUS CODE REPO as  overlap_probe_infer.py
(repo: https://github.com/realsee-developer/Argus  — NOT the RealSee3D dataset repo).

Writes <out>/pred.npz with key `poses` (N,4,4) camera-to-world, sorted input order.

Preprocessing MATCHES the official demo_gradio.py exactly (this is critical — Argus is trained
on a vertically-CROPPED, low-res ERP; feeding a full-FOV pano gives random poses):
    resize to (W=ARGUS_W, H=W//2) then crop rows [H*crop : H*(1-crop)]  (default W=560, crop=0.15
    -> final input 196x560). pose_encoding_to_extri360 returns world-to-camera OpenCV extrinsics;
    we invert to c2w. reorder_by_learning_ref MUST be False so poses stay in input order.

Env knobs (SLURM script):
    ARGUS_CKPT      path to argus_realsee3d.pt (else HF download)
    ARGUS_W         ERP target WIDTH (height=W//2). default 560 (demo default).
    ARGUS_CROP      vertical crop ratio. default 0.15 (demo default).
    ARGUS_POSE_C2W  set 1 if pose_encoding_to_extri360 already returns camera-to-world.
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from argus.models.argus import Argus
from argus.utils.pose_enc import pose_encoding_to_extri360


def load_erp(paths, target_w, crop_ratio):
    """Replicates demo_gradio.load_and_preprocess_images: resize to (W, W//2), then crop the
    top/bottom crop_ratio of the height. Returns (S,3,Hc,W) float in [0,1], RGB."""
    W = target_w; H = target_w // 2
    y0, y1 = int(H * crop_ratio), int(H * (1 - crop_ratio))
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)   # PIL: (width,height)
        arr = np.asarray(im)[y0:y1, :, :]                                  # vertical crop
        ims.append(torch.from_numpy(arr.copy()).float() / 255.0)
    return torch.stack(ims).permute(0, 3, 1, 2).contiguous()              # (S,3,Hc,W)


def normalize_poses(extr, invert):
    extr = np.asarray(extr, float)
    while extr.ndim > 3:
        extr = extr[0]
    S = extr.shape[0]
    out = np.tile(np.eye(4), (S, 1, 1))
    for i in range(S):
        R = extr[i][:3, :3]; t = extr[i][:3, 3]
        if invert:
            out[i, :3, :3] = R.T; out[i, :3, 3] = -R.T @ t
        else:
            out[i, :3, :3] = R; out[i, :3, 3] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_path", default=os.environ.get("ARGUS_CKPT"))
    ap.add_argument("--width", type=int, default=int(os.environ.get("ARGUS_W", "560")))
    ap.add_argument("--crop", type=float, default=float(os.environ.get("ARGUS_CROP", "0.15")))
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.images, "*")))
    if not paths:
        raise SystemExit(f"no images in {a.images}")
    imgs = load_erp(paths, a.width, a.crop)

    mp = a.model_path
    if mp is None:
        from huggingface_hub import hf_hub_download
        mp = hf_hub_download(repo_id="RealseeTechnology/argus-realsee3d",
                             filename="argus_realsee3d.pt")
    # False keeps output views in INPUT order (True reorders to reference-first, which breaks
    # our poses[i] <-> images[i] correspondence against external GT).
    model = Argus(reorder_by_learning_ref=False, restore_metric_scale=True)
    sd = torch.load(mp, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bb = [k for k in missing if any(t in k for t in
          ("aggregator", "backbone", "patch_embed", "blocks", "dinov2"))]
    print(f"[argus] ckpt tensors={len(sd)} | missing={len(missing)} "
          f"(backbone-ish missing={len(bb)}) | unexpected={len(unexpected)}")

    model.eval().cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(imgs.cuda())
    extr, _conf = pose_encoding_to_extri360(pose_encoding=pred["pose_enc"])
    extr = extr.float().cpu().numpy()
    invert = os.environ.get("ARGUS_POSE_C2W", "") not in ("1", "true", "True")
    poses = normalize_poses(extr, invert=invert)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(a.out, "pred.npz"), poses=poses.astype(np.float64))
    print(f"[argus] wrote poses {poses.shape} for {len(paths)} images "
          f"(input {tuple(imgs.shape)} invert_w2c={invert})")


if __name__ == "__main__":
    main()
