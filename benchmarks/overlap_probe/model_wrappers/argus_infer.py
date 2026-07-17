#!/usr/bin/env python3
"""
argus_infer.py — overlap_probe inference shim for Realsee Argus (ECCV'26).

COPY THIS FILE INTO THE ARGUS CODE REPO as  overlap_probe_infer.py
(repo: https://github.com/realsee-developer/Argus  — NOT the RealSee3D dataset repo).

Writes <out>/pred.npz with key `poses` (N,4,4) camera-to-world, in INPUT (sorted-file) order.

Two things are essential and both come straight from the Argus source:
  1. PREPROCESS exactly like demo_gradio.load_and_preprocess_images: resize to (W, W//2) then
     crop rows [H*crop : H*(1-crop)]  (W=560, crop=0.15 -> 196x560). Full-FOV panos give garbage.
  2. reorder_by_learning_ref=True is the TRAINED config (False -> degenerate poses). With True the
     aggregator reorders views reference-first, so pose_enc comes out in order [r, 0..S-1 excl r]
     (see heads/utils.reorder_by_reference) and predictions["ref_idx"] gives r. We INVERT that
     permutation to restore input order, so poses[i] <-> images[i] for the GT-aligned metrics.
pose_encoding_to_extri360 returns world-to-camera OpenCV extrinsics; we invert to c2w.

Env: ARGUS_CKPT, ARGUS_W (=560), ARGUS_CROP (=0.15), ARGUS_POSE_C2W (=1 to skip w2c->c2w).
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from argus.models.argus import Argus
from argus.utils.pose_enc import pose_encoding_to_extri360


def load_erp(paths, target_w, crop_ratio):
    W = target_w; H = target_w // 2
    y0, y1 = int(H * crop_ratio), int(H * (1 - crop_ratio))
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
        arr = np.asarray(im)[y0:y1, :, :]
        ims.append(torch.from_numpy(arr.copy()).float() / 255.0)
    return torch.stack(ims).permute(0, 3, 1, 2).contiguous()   # (S,3,Hc,W)


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
    model = Argus(reorder_by_learning_ref=True, restore_metric_scale=True)
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
    while extr.ndim > 3:                      # [B,S,4,4] -> [S,4,4]
        extr = extr[0]
    S = extr.shape[0]

    # Undo the reference-first reordering to restore INPUT order.
    ref_note = "none"
    if "ref_idx" in pred and S > 1:
        r = int(np.asarray(pred["ref_idx"].detach().cpu()).flatten()[0])
        perm = [r] + [i for i in range(S) if i != r]   # == reorder_by_reference order
        inv = np.argsort(perm)                          # reordered-position -> input slot
        extr = extr[inv]
        ref_note = f"r={r}"

    invert = os.environ.get("ARGUS_POSE_C2W", "") not in ("1", "true", "True")
    poses = normalize_poses(extr, invert=invert)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(a.out, "pred.npz"), poses=poses.astype(np.float64))
    print(f"[argus] wrote poses {poses.shape} for {len(paths)} images "
          f"(input {tuple(imgs.shape)} invert_w2c={invert} unreorder={ref_note})")


if __name__ == "__main__":
    main()
