"""
overlap_probe.adapters — one uniform interface over every pose/reconstruction model.

    class PoseModel:
        name: str
        available: bool
        metric: bool                       # does it claim METRIC scale?
        def predict(self, scene, workdir) -> Prediction

A Prediction returns camera-to-world poses in the SAME ORDER as scene.image_paths. The
evaluation harness (metrics.py) is gauge-robust, so a model may return poses in any frame /
scale; metric models are additionally scored on scale error.

The two ORACLE adapters need no external code and validate the whole pipeline end-to-end
(OracleModel must give ~0 error; NoisyOracle a controlled non-zero error). The real adapters
are thin SUBPROCESS wrappers: they dump the scene's images to a temp folder, invoke the model's
own inference (via overlap_probe_infer.py copied into the model repo), and read back a standard
`pred.npz` with key `poses` of shape (N,4,4) camera-to-world.
"""
import os, subprocess, shutil
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Prediction:
    poses_c2w: np.ndarray            # (N,4,4)
    metric: bool = False
    ok: bool = True
    note: str = ""
    points: "np.ndarray|None" = None    # optional dumped point cloud (M,3), model world frame
    colors: "np.ndarray|None" = None    # optional per-point RGB (M,3) uint8


class PoseModel:
    name = "base"
    available = False
    metric = False

    def predict(self, scene, workdir) -> Prediction:
        raise NotImplementedError


# ----------------------------------------------------------------------------- oracles
class OracleModel(PoseModel):
    """Returns GT poses (optionally rotated/scaled into a foreign frame to prove the metrics
    are gauge-invariant). Sanity target: ate_norm ~ 0, relrot ~ 0."""
    name = "oracle"; available = True; metric = True

    def __init__(self, foreign_frame=True):
        self.foreign_frame = foreign_frame

    def predict(self, scene, workdir):
        P = scene.gt_c2w.copy()
        if self.foreign_frame:
            th = 0.7
            G = np.eye(4)
            G[:3, :3] = np.array([[np.cos(th), 0, np.sin(th)],
                                  [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
            G[:3, 3] = [3.0, 0.0, -2.0]
            s = 1.7
            P = np.einsum("ij,njk->nik", G, P)
            P[:, :3, 3] *= s
        return Prediction(poses_c2w=P, metric=True, note="oracle")


class NoisyOracle(PoseModel):
    """GT + Gaussian yaw/translation noise. Use to calibrate what a given ATE/relrot 'means'."""
    name = "noisy"; available = True; metric = True

    def __init__(self, sigma_deg=8.0, sigma_m=0.25, seed=0):
        self.sd, self.sm, self.seed = sigma_deg, sigma_m, seed

    def predict(self, scene, workdir):
        rng = np.random.default_rng(self.seed + hash(scene.home + scene.regime) % 10000)
        P = scene.gt_c2w.copy()
        for k in range(len(P)):
            a = np.radians(rng.normal(0, self.sd))
            dR = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            P[k, :3, :3] = dR @ P[k, :3, :3]
            P[k, :3, 3] += rng.normal(0, self.sm, 3) * [1, 0, 1]
        return Prediction(poses_c2w=P, metric=True, note=f"noisy sd={self.sd} sm={self.sm}")


# ----------------------------------------------------------- real models (subprocess wrappers)
class _SubprocModel(PoseModel):
    """Common plumbing: stage images -> run a command -> load workdir/pred.npz['poses']."""
    entry_env = None      # env var that must point at the model's inference script/dir
    py_env = None         # env var for THIS model's python (its own venv); falls back to 'python'
    cmd_template = None

    def _py(self):
        return os.environ.get(self.py_env, "python") if self.py_env else "python"

    def _cmd(self, imgs_dir, out_dir):
        raise NotImplementedError

    def predict(self, scene, workdir):
        workdir = Path(workdir); imgs = workdir / "imgs"; imgs.mkdir(parents=True, exist_ok=True)
        for k, p in enumerate(scene.image_paths):
            shutil.copy(p, imgs / f"{k:03d}_{Path(p).stem}.jpg")
        out = workdir / "out"; out.mkdir(exist_ok=True)
        cmd = self._cmd(imgs, out)
        # Isolate the model's own venv: the harness runs under a different env (roomconn +
        # cineca-ai module) whose PYTHONPATH would otherwise SHADOW the model venv's torch.
        env = os.environ.copy()
        for var in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "LD_LIBRARY_PATH"):
            env.pop(var, None)
        try:
            subprocess.run(cmd, check=True, cwd=os.environ.get(self.entry_env, "."),
                           timeout=1800, env=env)
            data = np.load(out / "pred.npz")
            poses = data["poses"].astype(float)
            assert poses.shape == (scene.n, 4, 4), f"expected {(scene.n,4,4)}, got {poses.shape}"
            pts = data["points"] if "points" in data.files else None
            cols = data["colors"] if "colors" in data.files else None
            return Prediction(poses_c2w=poses, metric=self.metric, note="ok",
                              points=pts, colors=cols)
        except Exception as e:
            return Prediction(poses_c2w=np.tile(np.eye(4), (scene.n, 1, 1)),
                              metric=self.metric, ok=False, note=f"FAILED: {e}")


class ArgusModel(_SubprocModel):
    """Realsee Argus (ECCV'26), METRIC. CODE repo: github.com/realsee-developer/Argus
    (NOT the RealSee3D dataset repo). Set ARGUS_DIR to that checkout, and copy
    overlap_probe/model_wrappers/argus_infer.py into it as `overlap_probe_infer.py`.
    Weights: HF RealseeTechnology/argus-realsee3d (gated) -> point ARGUS_CKPT at the .pt."""
    name = "argus"; metric = True; entry_env = "ARGUS_DIR"; py_env = "ARGUS_PY"
    available = bool(os.environ.get("ARGUS_DIR"))

    def _cmd(self, imgs, out):
        return [self._py(), "overlap_probe_infer.py", "--images", str(imgs), "--out", str(out)]


class PanoVGGTModel(_SubprocModel):
    """PanoVGGT (CVPR'26). Repo: github.com/YijingGuo-June/PanoVGGT. NOT metric (Sim3 scale).
    Set PANOVGGT_DIR to the checkout; copy overlap_probe/model_wrappers/panovggt_infer.py into
    it as `overlap_probe_infer.py`. Set PANOVGGT_CONFIG + PANOVGGT_CKPT (model.pt)."""
    name = "panovggt"; metric = False; entry_env = "PANOVGGT_DIR"; py_env = "PANOVGGT_PY"
    available = bool(os.environ.get("PANOVGGT_DIR"))

    def _cmd(self, imgs, out):
        return [self._py(), "overlap_probe_infer.py", "--images", str(imgs), "--out", str(out)]


class VGGTTiledModel(_SubprocModel):
    """Perspective VGGT (CVPR'25) baseline (left out for now): each 360 pano is cut into
    perspective tiles, VGGT runs on the union, per-pano pose recovered from its tiles. Set
    VGGT_DIR + wire run_vggt_pano.py to emit out/pred.npz['poses'] (N,4,4)."""
    name = "vggt_tiled"; metric = False; entry_env = "VGGT_DIR"; py_env = "VGGT_PY"
    available = bool(os.environ.get("VGGT_DIR"))

    def _cmd(self, imgs, out):
        return [self._py(), "run_vggt_pano.py", "--pano_dir", str(imgs), "--out", str(out)]


REGISTRY = {m.name: m for m in [OracleModel, NoisyOracle,
                                ArgusModel, PanoVGGTModel, VGGTTiledModel]}


def make(name, **kw):
    cls = REGISTRY[name]
    return cls(**kw) if kw else cls()
