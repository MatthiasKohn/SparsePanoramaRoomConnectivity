"""
Door primitive + evidence fusion — backbone of the door-centric pipeline.

Reframed after the kitchen mislabel: open/closed is an APPEARANCE property and is
NOT inferred from depth. Depth only tells us whether usable see-through GEOMETRY
exists at a door (a pose anchor). So a Door carries:
  - category   : 'door' | 'window' | 'opening'   (what it is)
  - seethrough : does depth show far geometry here? (can it anchor pose geometrically)
We never call a door 'closed' just because depth is flat there.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Door:
    pano: str
    azimuth_deg: float
    az_extent_deg: float = 0.0
    score: float = 1.0
    category: str = "door"          # 'door' | 'window' | 'opening'
    seethrough: bool = False        # depth shows usable geometry through it
    source: str = ""                # 'semantic' | 'depth' | 'fused'
    meta: dict = field(default_factory=dict)


def _circ_diff(a, b):
    return abs(((a - b + 180) % 360) - 180)


def fuse(semantic: List[Door], depth_apertures: List[Door],
         match_deg: float = 20.0) -> List[Door]:
    """Combine semantic doors (existence/location, from RGB) with depth
    see-through apertures (geometry availability). Each semantic door gets a
    seethrough flag; apertures with no semantic door are reported as 'opening'
    (a leaf-less opening, or a depth false-positive to check). No open/closed."""
    out, used = [], set()
    for s in semantic:
        hit = None
        for i, a in enumerate(depth_apertures):
            if i in used:
                continue
            if _circ_diff(s.azimuth_deg, a.azimuth_deg) < match_deg:
                hit = (i, a); break
        if hit:
            used.add(hit[0])
            out.append(Door(s.pano, s.azimuth_deg, max(s.az_extent_deg, hit[1].az_extent_deg),
                            0.5 * (s.score + hit[1].score), category=s.category,
                            seethrough=True, source="fused"))
        else:
            out.append(Door(s.pano, s.azimuth_deg, s.az_extent_deg, s.score,
                            category=s.category, seethrough=False, source="semantic"))
    for i, a in enumerate(depth_apertures):
        if i not in used:
            out.append(Door(a.pano, a.azimuth_deg, a.az_extent_deg, a.score,
                            category="opening", seethrough=True, source="depth"))
    return out
