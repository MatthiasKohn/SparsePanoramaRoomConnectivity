#!/usr/bin/env python3
"""
panovggt_infer.py — overlap_probe inference shim for PanoVGGT (CVPR'26).

COPY THIS FILE INTO THE PanoVGGT REPO as  overlap_probe_infer.py
(repo: https://github.com/YijingGuo-June/PanoVGGT).

Reads a folder of equirectangular panoramas and writes  <out>/pred.npz  with key `poses`
of shape (N,4,4): camera-to-world, in sorted input order — the contract adapters.py expects.

Built from the repo's own app.py: PanoVGGT emits `preds["camera_poses"]` as 4x4 matrices whose
translation is the camera CENTRE in world coordinates (app.py uses camera_poses[...,:3,3] as the
camera position), i.e. these are already camera-to-world. We just extract and save them.

Env knobs (set by the SLURM script):
    PANOVGGT_CONFIG   default training/config/default.yaml
    PANOVGGT_CKPT     default checkpoints/model.pt
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from panovggt.models.panovggt_model import PanoVGGTModel
from panovggt.utils.basic import load_images_as_tensor


def load_model(config_path, checkpoint_path, device):
    cfg = OmegaConf.load(config_path); OmegaConf.resolve(cfg); mc = cfg.model
    model = PanoVGGTModel(
        img_size=cfg.img_size, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim,
        enable_camera=mc.enable_camera, enable_depth=mc.enable_depth,
        enable_point=mc.enable_point,
        aggregator=OmegaConf.to_container(mc.aggregator, resolve=True))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt:
            ckpt = ckpt[key]; break
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Report whether the checkpoint actually covered the DINOv2 backbone. If the compute node
    # was offline and could not fetch dinov2 pretrained weights, a backbone left in 'missing'
    # here means it is RANDOM -> poses would be garbage. Cache dinov2_vitl14_pretrain.pth into
    # $TORCH_HOME/hub/checkpoints to avoid that.
    bb = [k for k in missing if any(t in k for t in
          ("aggregator", "backbone", "patch_embed", "blocks", "dinov2", "frame_", "global_"))]
    print(f"[panovggt] ckpt tensors={len(sd)} | missing={len(missing)} "
          f"(backbone-ish missing={len(bb)}) | unexpected={len(unexpected)}")
    if bb:
        print("  WARNING backbone keys MISSING from ckpt (random unless dinov2 cached):", bb[:5])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.environ.get("PANOVGGT_CONFIG",
                                                       "training/config/default.yaml"))
    ap.add_argument("--checkpoint", default=os.environ.get("PANOVGGT_CKPT",
                                                           "checkpoints/model.pt"))
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.images, "*")))
    if not paths:
        raise SystemExit(f"no images in {a.images}")

    device = "cuda"
    model = load_model(a.config, a.checkpoint, device).eval().to(device)
    imgs = load_images_as_tensor(a.images, interval=1).to(device)   # [S,3,H,W], sorted
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model(imgs[None])                                   # [1,S,...]

    if "camera_poses" not in preds:
        raise SystemExit("PanoVGGT preds has no 'camera_poses' — enable_camera must be true "
                         "in the config.")
    poses = preds["camera_poses"].float().cpu().numpy()
    poses = np.squeeze(poses)                                       # -> [S,4,4]
    if poses.shape[-2:] != (4, 4):
        raise SystemExit(f"unexpected camera_poses shape {poses.shape}")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(a.out, "pred.npz"), poses=poses.astype(np.float64))
    print(f"[panovggt] wrote poses {poses.shape} for {len(paths)} images")


if __name__ == "__main__":
    main()
