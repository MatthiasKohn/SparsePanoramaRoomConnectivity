"""Completion provider — the generative hole-filling building block (for the walkthrough).

  none : leave holes.
  cv2  : classical texture inpainting (offline, deterministic -> stable in video). Good for small holes.
  sd   : SDXL diffusion inpainting (plausible new content). Weights cached via scripts/setup_sd_inpaint.sh.

NOTE: this only fills UNOBSERVED (low-alpha) pixels; it does not fix soft/see-through observed
surfaces. Realistic *observed*-surface completion is the 3D-consistent research step, not this.
"""
import numpy as np
import cv2

SD_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"   # public, ungated


def load_sd_inpainter(device):
    """SDXL inpainting pipeline; local_files_only forces the offline cache (no failing net call)."""
    import torch
    from diffusers import AutoPipelineForInpainting
    dt = torch.float16 if device == "cuda" else torch.float32
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(SD_INPAINT_MODEL, torch_dtype=dt,
                                                         variant="fp16", local_files_only=True)
    except Exception:
        pipe = AutoPipelineForInpainting.from_pretrained(SD_INPAINT_MODEL, torch_dtype=dt,
                                                         local_files_only=True)
    pipe = pipe.to(device); pipe.set_progress_bar_config(disable=True)
    return pipe


def inpaint_holes(rgb, alpha, thr, backend="cv2", sd=None,
                  prompt="a realistic empty interior room, wall and floor"):
    hole = (alpha < thr).astype(np.uint8)
    if hole.sum() < 20:
        return rgb
    hole = cv2.dilate(hole, np.ones((3, 3), np.uint8), 1)
    if backend == "sd" and sd is not None:
        from PIL import Image
        H, W = rgb.shape[:2]
        img = Image.fromarray(rgb).resize((512, 512))
        msk = Image.fromarray(hole * 255).resize((512, 512))
        gen = sd(prompt=prompt, image=img, mask_image=msk, num_inference_steps=15,
                 guidance_scale=7.0, strength=0.99).images[0]
        return np.array(gen.convert("RGB").resize((W, H)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, hole * 255, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
