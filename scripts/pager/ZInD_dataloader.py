"""ZInD (Zillow Indoor Dataset) — RGB panoramas only, NO ground-truth depth. Inference-only
adapter for PaGeR's inference.py.

INSTALL: copy this file into  <PaGeR>/dataloaders/ZInD_dataloader.py , then two 1-line edits to
<PaGeR>/inference.py:  (a) add  from dataloaders.ZInD_dataloader import ZInD  ;
(b) add "ZInD" to the --dataset choices list.

RESTRICT which homes to process (IMPORTANT — the full ZInD is ~67k panos / ~a day of GPU):
    export PAGER_ZIND_HOMES=scripts/depth_homes.txt   # a file of home ids (one per line) ...
    export PAGER_ZIND_HOMES=0053,0149                 # ... or a comma/space list. Unset = ALL homes.

RUN (indoor scale head; ZInD is all indoor):
    python inference.py --config configs/inference.yaml --checkpoint pager \
        --dataset ZInD --scene_mode indoor --data_path <ZIND_full_dataset_root> \
        --results_path <out> --generate_eval
-> writes <out>/{depth,normals}/preds/<home>__<stem>.npz  (id encodes home+stem).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from dataloaders._base import PanoDataset


def _home_filter():
    sel = os.environ.get("PAGER_ZIND_HOMES", "").strip()
    if not sel:
        return None
    if Path(sel).exists():
        return set(Path(sel).read_text().split())
    return set(sel.replace(",", " ").split())


class ZInD(PanoDataset):
    HEIGHT, WIDTH = 1024, 2048          # ERP working res (base resizes every pano to this)
    SUBDIR = ""                         # --data_path already points at the full_dataset root
    MIN_DEPTH, MAX_DEPTH = 1e-2, 20.0   # indoor depth range
    POLE_CROP_FRAC = 0.0

    def _scan(self):
        keep = _home_filter()
        out = []
        for p in sorted(self.data_path.glob("*/panos/*.jpg")):
            home = p.parent.parent.name
            if keep is not None and home not in keep:
                continue
            out.append({"id": f"{home}__{p.stem}", "rgb": p, "depth": None})
        print(f"[ZInD dataloader] {len(out)} panos"
              + (f" from {len(keep)} homes ({os.environ.get('PAGER_ZIND_HOMES')})" if keep else " (ALL homes)"))
        return out

    def _load_depth(self, entry):
        return np.zeros((self.HEIGHT, self.WIDTH), np.float32)   # no GT; dummy for __getitem__
