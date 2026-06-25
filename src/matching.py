"""
Cross-pano door matching — turns per-pano door detections into connectivity edges.

A door seen from two adjacent rooms is the SAME physical doorway. We match doors
across two panoramas by appearance of the door region (v1), then keep mutual
nearest neighbours above a similarity threshold. The embedding model (DINOv2 by
default) is isolated in `_embed`; everything else is plain numpy and unit-tested.

Design note (the hard part, for the contrastive upgrade later): opposite sides of
a door look different, so the durable signal is "what A sees THROUGH the door ~
what B sees DIRECTLY there". v1 embeds the door crop as a baseline; a learned
cross-view embedding is the planned replacement.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

from src import panoproj
from src.doors import Door


@dataclass
class DoorMatch:
    a_idx: int
    b_idx: int
    score: float
    az_a: float
    az_b: float


def door_crop(equi_rgb, door: Door, fov_deg=70.0, hw=(448, 448), pitch_deg=0.0):
    """Perspective patch centred on a door's azimuth (input for the embedder)."""
    return panoproj.e2p(equi_rgb, door.azimuth_deg, pitch_deg, fov_deg, hw)


class DoorMatcher:
    def __init__(self, embed: Optional[Callable] = None,
                 model_name="dinov2_vits14", device="cuda",
                 fov_deg=70.0, crop_hw=(448, 448), min_sim=0.55):
        self.embed = embed                 # injectable for testing
        self.model_name = model_name
        self.device = device
        self.fov = fov_deg
        self.crop_hw = crop_hw
        self.min_sim = min_sim
        self._model = None

    # ---- only GPU/model part ----
    def _embed(self, img_rgb):
        if self.embed is not None:
            v = np.asarray(self.embed(img_rgb), float)
            return v / (np.linalg.norm(v) + 1e-8)
        import torch
        if self._model is None:
            self._model = torch.hub.load("facebookresearch/dinov2", self.model_name).to(self.device).eval()
        import torch.nn.functional as F
        x = torch.from_numpy(img_rgb).float().permute(2, 0, 1)[None] / 255.0
        x = F.interpolate(x, size=(self.crop_hw[0] // 14 * 14, self.crop_hw[1] // 14 * 14))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(self.device)
        with torch.no_grad():
            v = self._model(x)[0].cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-8)

    def embed_doors(self, equi_rgb, doors: List[Door]):
        return np.stack([self._embed(door_crop(equi_rgb, d, self.fov, self.crop_hw))
                         for d in doors]) if doors else np.zeros((0, 1))

    def match(self, equiA, doorsA, equiB, doorsB) -> List[DoorMatch]:
        if not doorsA or not doorsB:
            return []
        EA, EB = self.embed_doors(equiA, doorsA), self.embed_doors(equiB, doorsB)
        S = EA @ EB.T                                   # cosine (rows normalized)
        matches = []
        for i in range(len(doorsA)):
            j = int(np.argmax(S[i]))
            if S[i, j] >= self.min_sim and int(np.argmax(S[:, j])) == i:   # mutual NN
                matches.append(DoorMatch(i, j, float(S[i, j]),
                                         doorsA[i].azimuth_deg, doorsB[j].azimuth_deg))
        return sorted(matches, key=lambda m: -m.score)
