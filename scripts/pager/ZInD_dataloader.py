"""ZInD (Zillow Indoor Dataset) — RGB panoramas only, NO ground-truth depth. Inference-only
adapter for PaGeR's inference.py.

INSTALL: copy this file into  <PaGeR>/dataloaders/ZInD_dataloader.py , then make two 1-line edits
to <PaGeR>/inference.py:
    (a) add:  from dataloaders.ZInD_dataloader import ZInD
    (b) add "ZInD" to the --dataset  choices=[...]  list.
RUN (indoor scale head; ZInD is all indoor):
    python inference.py --config configs/inference.yaml --checkpoint pager \
        --dataset ZInD --scene_mode indoor --data_path <ZIND_full_dataset_root> \
        --results_path <out> --generate_eval
-> writes metric-depth + normals .npz per pano as <out>/{depth,normals}/preds/<home>__<stem>.npz
   (id encodes home+stem so scripts/pager/pager_to_pipeline.py can map them back).
"""
from __future__ import annotations

import numpy as np

from dataloaders._base import PanoDataset


class ZInD(PanoDataset):
    HEIGHT, WIDTH = 1024, 2048          # ERP working res (base resizes every pano to this, LANCZOS)
    SUBDIR = ""                         # --data_path already points at the full_dataset root
    MIN_DEPTH, MAX_DEPTH = 1e-2, 20.0   # indoor depth range
    POLE_CROP_FRAC = 0.0

    def _scan(self):
        # <root>/<home>/panos/<stem>.jpg  ->  id "<home>__<stem>"
        return [
            {"id": f"{p.parent.parent.name}__{p.stem}", "rgb": p, "depth": None}
            for p in sorted(self.data_path.glob("*/panos/*.jpg"))
        ]

    def _load_depth(self, entry):
        # ZInD has no GT depth. Return a dummy map so PanoDataset.__getitem__ runs; inference only
        # consumes rgb_cubemap + id, so this is never used for the prediction itself.
        return np.zeros((self.HEIGHT, self.WIDTH), np.float32)
