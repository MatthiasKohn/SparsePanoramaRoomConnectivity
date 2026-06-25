"""
Semantic door detector (SegFormer / ADE20K) over panorama perspective views.

Door vs window are kept as SEPARATE categories, and a door must REACH THE FLOOR
(its segment extends into the lower part of the crop) -- a cheap, robust prior
that rejects windows / cabinets / high wall patches misread as doors.

Only `_segment()` touches the GPU/model; everything else is plain numpy/cv2 and is
unit-tested with an injected fake segmenter.

    pip install "transformers>=4.40" torch pillow
    python experiments/exp06_semantic_doors.py
"""
from typing import Callable, List, Optional
import numpy as np
import cv2

from src import panoproj
from src.doors import Door


class SemanticDoorDetector:
    def __init__(self, pano_id="", model_name="nvidia/segformer-b3-finetuned-ade-512-512",
                 device="cuda", n_views=8, fov_deg=90.0, view_hw=(640, 640),
                 door_keywords=("door",), window_keywords=("window", "windowpane"),
                 min_area_frac=0.012, floor_frac=0.55, merge_deg=14.0,
                 segmenter: Optional[Callable] = None):
        self.pano_id = pano_id
        self.model_name = model_name
        self.device = device
        self.n_views = n_views
        self.fov = fov_deg
        self.view_hw = view_hw
        self.door_kw = tuple(k.lower() for k in door_keywords)
        self.win_kw = tuple(k.lower() for k in window_keywords)
        self.min_area_frac = min_area_frac
        self.floor_frac = floor_frac          # door bottom must be below this * height
        self.merge_deg = merge_deg
        self._segmenter = segmenter
        self._model = self._proc = None
        self._door_ids = self._win_ids = None

    # ---- the only GPU/model part ----
    def _load(self):
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        self._proc = SegformerImageProcessor.from_pretrained(self.model_name)
        self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_name).to(self.device).eval()
        id2label = self._model.config.id2label
        self._door_ids = [i for i, l in id2label.items() if any(k in l.lower() for k in self.door_kw)]
        self._win_ids = [i for i, l in id2label.items() if any(k in l.lower() for k in self.win_kw)]
        print(f"[semantic] door ids={[(i, id2label[i]) for i in self._door_ids]} "
              f"window ids={[(i, id2label[i]) for i in self._win_ids]}")

    def _segment(self, image_rgb):
        if self._segmenter is not None:
            return self._segmenter(image_rgb)
        import torch
        if self._model is None:
            self._load()
        inp = self._proc(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._model(**inp).logits
        up = torch.nn.functional.interpolate(logits, size=image_rgb.shape[:2],
                                             mode="bilinear", align_corners=False)
        return up.argmax(1)[0].cpu().numpy()

    def _ids(self):
        if self._door_ids is not None:
            return self._door_ids, self._win_ids
        return [14], [8]                       # ADE defaults: door, windowpane

    # ---- model-free plumbing (unit-tested) ----
    def _components(self, lab, ids, Wc, Hc, yaw, category, require_floor):
        mask = np.isin(lab, ids).astype(np.uint8)
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
        min_area = self.min_area_frac * Hc * Wc
        out = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            x0 = stats[i, cv2.CC_STAT_LEFT]; w = stats[i, cv2.CC_STAT_WIDTH]
            y0 = stats[i, cv2.CC_STAT_TOP]; h = stats[i, cv2.CC_STAT_HEIGHT]
            if require_floor and (y0 + h) < self.floor_frac * Hc:
                continue                       # door must reach toward the floor
            az = panoproj.crop_x_to_azimuth(cent[i][0], yaw, self.fov, Wc)
            e0 = panoproj.crop_x_to_azimuth(x0, yaw, self.fov, Wc)
            e1 = panoproj.crop_x_to_azimuth(x0 + w, yaw, self.fov, Wc)
            ext = abs(((e1 - e0 + 180) % 360) - 180)
            out.append(Door(self.pano_id, float(az), float(ext),
                            float(stats[i, cv2.CC_STAT_AREA] / (Hc * Wc)),
                            category=category, source="semantic"))
        return out

    def detect(self, equi_rgb) -> List[Door]:
        door_ids, win_ids = self._ids()
        Hc, Wc = self.view_hw
        raw = []
        for yaw, view in panoproj.ring_views(equi_rgb, self.n_views, self.fov, self.view_hw):
            lab = self._segment(view)
            raw += self._components(lab, door_ids, Wc, Hc, yaw, "door", require_floor=True)
            raw += self._components(lab, win_ids, Wc, Hc, yaw, "window", require_floor=False)
        return self._merge(raw)

    def _merge(self, doors: List[Door]) -> List[Door]:
        out: List[Door] = []
        for d in sorted(doors, key=lambda x: -x.score):
            if all(not (o.category == d.category and
                        abs(((d.azimuth_deg - o.azimuth_deg + 180) % 360) - 180) < self.merge_deg)
                   for o in out):
                out.append(d)
        return out
