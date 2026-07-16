#!/usr/bin/env python3
"""
argus_infer.py — overlap_probe inference shim for Realsee Argus (ECCV'26).

COPY THIS FILE INTO THE ARGUS CODE REPO as  overlap_probe_infer.py
(repo: https://github.com/realsee-developer/Argus  — NOT the RealSee3D dataset repo).

It reads a folder of equirectangular panoramas and writes  <out>/pred.npz  with key
`poses` of shape (N,4,4): camera-to-world, in the SAME (sorted) order as the input files —
exactly the contract overlap_probe/adapters.py expects.

Built from the repo's documented API:
    from argus.models.argus import Argus
    from argus.utils.pose_enc import pose_encoding_to_extri360
    pred = model(images[S,3,H,W in 0..1]) ; extr,conf = pose_encoding_to_extri360(pred["pose_enc"])
`extr` is a world-to-camera extrinsic (VGGT convention); we invert it to camera-to-world.

Env knobs (set by the SLURM script):
    ARGUS_CKPT      path to argus_realsee3d.pt   (else tries HF download; needs `hf auth login`)
    ARGUS_H         ERP input height (width = 2*H). default 518 (patch-14 friendly, 2:1).
    ARGUS_POSE_C2W  set to 1 if pose_encoding_to_extri360 already returns camera-to-world
                    (then we skip the inversion). Leave unset for the default w2c assumption.
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from argus.models.argus import Argus
from argus.utils.pose_enc import pose_encoding_to_extri360


def load_erp(paths, H, W):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
        ims.append(torch.from_numpy(np.asarray(im)).float() / 255.0)
    return torch.stack(ims).permute(0, 3, 1, 2).contiguous()   # [S,3,H,W]


def normalize_poses(extr, invert):
    """extr: (...,3,4) or (...,4,4). Return (S,4,4). invert=True treats input as w2c->c2w."""
    extr = np.asarray(extr, float)
    while extr.ndim > 3:            # squeeze any leading batch dims
        extr = extr[0]
    S = extr.shape[0]
    out = np.tile(np.eye(4), (S, 1, 1))
    for i in range(S):
        R = extr[i][:3, :3]; t = extr[i][:3, 3]
        if invert:                 # world-to-camera -> camera-to-world
            out[i, :3, :3] = R.T; out[i, :3, 3] = -R.T @ t
        else:                      # already camera-to-world
            out[i, :3, :3] = R; out[i, :3, 3] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_path", default=os.environ.get("ARGUS_CKPT"))
    ap.add_argument("--height", type=int, default=int(os.environ.get("ARGUS_H", "518")))
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.images, "*")))
    if not paths:
        raise SystemExit(f"no images in {a.images}")
    H = a.height - (a.height % 14); W = 2 * H
    imgs = load_erp(paths, H, W)

    mp = a.model_path
    if mp is None:
        from huggingface_hub import hf_hub_download
        mp = hf_hub_download(repo_id="RealseeTechnology/argus-realsee3d",
                             filename="argus_realsee3d.pt")
    model = Argus(reorder_by_learning_ref=True, restore_metric_scale=True)
    sd = torch.load(mp, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
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
          f"(H={H} W={W} invert_w2c={invert})")


if __name__ == "__main__":
    main()
