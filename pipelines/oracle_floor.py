"""DEPRECATED — replaced by pipelines/floor.py (the swappable-block substitution pipeline).

The oracle is now just one config of the general pipeline:
    python -m pipelines.floor --home <...> --floor <...> --pose_model gt --depth_model gt_layout --visuals

All the logic (GT-layout depth, PaGeR fusion, reflection-fixed poses, calibrated render, 3DGS
optimization, room-aware rendering, walkthrough, held-out metrics) moved into pipelines/floor.py +
sparsepano/providers/. See docs/Substitution_Plan.md.
"""
import sys

if __name__ == "__main__":
    sys.exit("pipelines/oracle_floor.py is deprecated — use `python -m pipelines.floor ... "
             "--pose_model gt --depth_model gt_layout --visuals` (see docs/Substitution_Plan.md).")
