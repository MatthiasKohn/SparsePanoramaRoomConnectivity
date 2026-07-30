"""Pose provider — the camera-pose building block.

get_poses(fl, meters, model, ...) -> {pano_stem: 4x4 camera-to-world}

Models:
  gt         : ZInD floor_plan_transformation (the oracle). Includes the reflection fix so the
               world geometry is correct (phi = -psi - rot_deg; a proper-rotation pose mirrors).
  noise      : GT perturbed by a yaw error (std `noise_deg`) + a horizontal translation error
               (`noise_m`) per camera -> the controlled pose-sensitivity sweep (no external method).
  salve/badgr/covispose : load precomputed poses from `pose_file` (a json/npz the external method
               produced on this floor). Not yet integrated -> raises with a clear message.

The GT/noise poses are self-contained (no GPU, no external deps).
"""
import json
from pathlib import Path
import numpy as np


def pose_c2w_gt(info, S):
    """Correct GT camera-to-world (Y-up world). ZInD image azimuth relates to floor-plan azimuth
    by phi = -psi - rot_deg (a reflection), so a plain proper-rotation pose places rooms mirrored;
    this bakes in the reflection. Verified: shared doors agree to 0.000 m from both rooms."""
    rot = np.radians(info["rot_deg"]); C, Sr = np.cos(rot), np.sin(rot)
    R = np.array([[-C, 0.0, -Sr], [0.0, 1.0, 0.0], [-Sr, 0.0, C]])   # Ry(-rot) @ diag(-1,1,1)
    pos = np.asarray(info["pos"], float)
    T = np.eye(4); T[:3, :3] = R
    T[:3, 3] = [pos[0] * S, float(info.get("cam_h_m") or 0.0), pos[1] * S]
    return T


def pose_c2w_render(info, S):
    """PROPER (det=+1) camera at the pano position (forward = Ry(-rot)). gsplat cannot render
    through the reflected GT pose, so renders use this + an output flip (see the renderer)."""
    rot = np.radians(info["rot_deg"]); C, Sr = np.cos(rot), np.sin(rot)
    R = np.array([[C, 0.0, -Sr], [0.0, 1.0, 0.0], [Sr, 0.0, C]])
    pos = np.asarray(info["pos"], float)
    T = np.eye(4); T[:3, :3] = R
    T[:3, 3] = [pos[0] * S, float(info.get("cam_h_m") or 0.0), pos[1] * S]
    return T


def get_poses(fl, meters, model="gt", noise_deg=0.0, noise_m=0.0, seed=0, pose_file=None):
    """Return (build_poses, render_poses, gt_poses), each {stem: 4x4 c2w}.
      build_poses  : reflected GT convention (for placing gaussians in the world).
      render_poses : proper (det=+1) convention for the SAME pose (gsplat renders through these).
      gt_poses     : the true GT (for pose-error metrics).
    Perturbation (noise / external) is applied at the (rot_deg, pos) level so build & render stay
    consistent. pos is in coord units; noise_m metres -> add noise_m/meters in coord units."""
    infos = {s: dict(rot_deg=float(fl.panos[s]["rot_deg"]),
                     pos=np.asarray(fl.panos[s]["pos"], float),
                     cam_h_m=fl.panos[s].get("cam_h_m")) for s in fl.panos}
    if model == "noise":
        rng = np.random.default_rng(seed)
        for s in infos:
            infos[s]["rot_deg"] += noise_deg * rng.standard_normal()
            ang = rng.uniform(0, 2 * np.pi)
            infos[s]["pos"] = infos[s]["pos"] + (noise_m / meters) * np.array([np.cos(ang), np.sin(ang)])
    elif model == "drift":
        # COHERENT drift: rooms don't err independently -- error accumulates along the connectivity
        # graph like a relative-pose estimator chaining room-to-room (SALVe/CovisPose). All panos in
        # a room share that room's accumulated error; adjacent rooms differ by ONE increment, so
        # neighbours stay locally consistent while distant rooms drift apart. Here noise_deg/noise_m
        # are the PER-EDGE increment std (not the absolute error); the absolute RMSE emerges.
        rng = np.random.default_rng(seed)
        rooms = {}
        for s in infos:
            rooms.setdefault(fl.panos[s]["room"], []).append(s)
        rids = list(rooms)
        adjacency = {r: set() for r in rids}
        for i, ra in enumerate(rids):
            for rb in rids[i + 1:]:
                if any(fl.shared_door(a, b, tol=0.25) is not None for a in rooms[ra] for b in rooms[rb]):
                    adjacency[ra].add(rb); adjacency[rb].add(ra)
        from collections import deque
        err = {rids[0]: (0.0, np.zeros(2))}          # root room: zero error
        dq = deque([rids[0]])
        while dq:
            u = dq.popleft()
            for v in adjacency[u]:
                if v not in err:
                    pr, pp = err[u]
                    err[v] = (pr + noise_deg * rng.standard_normal(),
                              pp + (noise_m / meters) * rng.standard_normal(2))
                    dq.append(v)
        for r in rids:                                # disconnected rooms: own independent draw
            if r not in err:
                err[r] = (noise_deg * rng.standard_normal(), (noise_m / meters) * rng.standard_normal(2))
        for s in infos:
            dr, dp = err[fl.panos[s]["room"]]
            infos[s]["rot_deg"] += dr; infos[s]["pos"] = infos[s]["pos"] + dp
    elif model in ("salve", "badgr", "covispose"):
        if not (pose_file and Path(pose_file).exists()):
            raise NotImplementedError(
                f"pose_model={model}: run {model} on this floor and pass --pose_file <json of "
                f"stem->{{rot_deg,pos}}>. Until then use --pose_model noise for the sensitivity sweep.")
        raw = json.load(open(pose_file))
        missing = [s for s in infos if s not in raw]
        for s in list(infos):
            if s in raw:
                infos[s]["rot_deg"] = float(raw[s]["rot_deg"]); infos[s]["pos"] = np.asarray(raw[s]["pos"], float)
        if missing:
            # Panos the method did not localize must NOT silently keep GT (that flatters the result).
            # Drop them so build/eval only use estimated poses.
            print(f"[poses] {model}: {len(missing)}/{len(infos)} panos not localized -> dropped "
                  f"(no GT fallback): {missing[:6]}{'...' if len(missing) > 6 else ''}")
            for s in missing:
                infos.pop(s)
    elif model != "gt":
        raise ValueError(f"unknown pose_model {model!r}")

    build = {s: pose_c2w_gt(infos[s], meters) for s in infos}
    render = {s: pose_c2w_render(infos[s], meters) for s in infos}
    gt = {s: pose_c2w_gt(fl.panos[s], meters) for s in infos}
    return build, render, gt
