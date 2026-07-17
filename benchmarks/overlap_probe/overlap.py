"""
overlap_probe.overlap — a GT covisibility proxy for each pano pair.

We can't cheaply compute exact surface covisibility without clean per-pano geometry, so we
use a robust, depth-free proxy grounded in ZInD's own structure. It is intentionally simple
and defensible; it is the STRATIFYING VARIABLE, not a claimed contribution.

  "same"     both panos in the SAME room            -> high overlap (they see the same walls)
  "adjacent" different rooms that SHARE A DOOR       -> low overlap (only the doorway sliver)
  "far"      different rooms, no shared door         -> ~zero overlap

Also returns a continuous scalar in [0,1] (higher = more overlap): 1.0 same-room, else a
smooth function of inter-camera distance capped by the adjacency category. Use whichever the
analysis wants; the categorical one maps directly onto the scientific question.
"""
import numpy as np
from itertools import combinations


def annotate(scene, fl):
    """Fill scene.overlap[(i,j)] = (category, scalar) for i<j."""
    cat = {}
    for i, j in combinations(range(scene.n), 2):
        si, sj = scene.stems[i], scene.stems[j]
        if fl.same_room(si, sj):
            c, base = "same", 1.0
        elif fl.shared_door(si, sj) is not None:
            c, base = "adjacent", 0.5
        else:
            c, base = "far", 0.0
        d = np.linalg.norm((np.asarray(fl.panos[si]["pos"], float)
                            - np.asarray(fl.panos[sj]["pos"], float))
                           * fl.meters_per_coord)
        # distance-modulated scalar, capped by category so 'far' stays ~0
        scal = base * float(np.exp(-d / 4.0)) if c != "far" else 0.0
        cat[(i, j)] = (c, scal)
    scene.overlap = cat
    return scene


def category_counts(scene):
    out = {"same": 0, "adjacent": 0, "far": 0}
    for (c, _) in scene.overlap.values():
        out[c] += 1
    return out
