# Codex task: restructure SparsePanoramaRoomConnectivity into a clean, dataset-agnostic pipeline

You are refactoring an existing PhD research repo. **Do not rewrite it from scratch** and **do not lose
working code or results.** Reorganize it into well-structured topical folders, replace hard-coded
dataset assumptions with a pluggable abstraction, and add a proper evaluation/reporting layer so every
stage can be measured. Preserve the behaviour of the validated pipeline (numbers below must still
reproduce).

Work incrementally with small, reviewable git commits. After each phase, run the smoke/regression
checks and keep the repo runnable. If a genuine ambiguity blocks you, STOP and leave a clearly-marked
`TODO(codex):` note rather than guessing destructively.

---

## 1. Project context (so your structure matches the science)

Goal: from **sparse 360° panoramas (~one per room, near-zero overlap except through doorways)** infer
**room connectivity**, **metric floor layout (camera poses)**, and eventually **3D**. The target
pipeline (name the stages after this):

```
panos → per-room metric geometry (depth+normals; foundation model, e.g. PaGeR)
      → door detection + cross-view door embedding
      → connectivity graph + door-anchored relative pose (flip resolved by embedding+normals)
      → global SE(2)/SE(3) layout
      → [later] feed-forward 3DGS + diffusion, conditioned on connectivity  (Paper 2)
```

Key validated results that MUST still reproduce after the refactor (regression targets):
- Connectivity assign-AP on 197 held-out ZInD homes: **0.913 with GT doors, 0.842 with detected doors**.
- These come from the door embedding (`former-former-src/contrastive.py`) + assign scoring (`exp23`/`exp12`) and the
  GT-free path (`exp28`). The trained weights exist at repo root: `best.pt`, `best_hardneg.pt`,
  `door_encoder.pt` — keep them.

The `overlap_probe/` folder is a clean, self-contained module (foundation-model pose benchmark) and is
the STYLE TEMPLATE for the whole repo: a small package with `common.py` (data/scene), `metrics.py`,
`adapters.py` (pluggable model interface), a CLI runner, and `slurm/`. Keep it and mirror its clarity.

---

## 2. Prime directives (in priority order)

1. **De-hardcode the dataset.** Today ZInD is baked in (`former-former-src/zind.py`, `config.py`, paths, field names).
   Introduce a dataset abstraction so a new dataset = implement one adapter; nothing else changes. ZInD
   is the reference implementation. Do NOT implement other datasets now — just make adding them trivial,
   and expose per-dataset **capability flags** (has_gt_poses / has_gt_depth / has_gt_doors / has_gt_rooms)
   so evaluators gracefully skip what a dataset lacks (e.g. the Immersight real captures have no GT).
2. **Evaluate every aspect, easily and correctly.** One evaluation layer with a metric module per stage,
   each producing (a) aggregate numbers, (b) per-scene diagnostics, and (c) a written report. Must cover:
   door **detection** (recall/precision @ angular tolerance, and *which* GT doors were missed per scene),
   **depth/geometry** quality (AbsRel, RMSE, δ<1.25 — only when GT depth exists), **connectivity**
   (AP + precision/recall/F1 at the operating point + a per-home breakdown of wrong/missing edges),
   **pose/flip** (relative-rotation error, Sim3-aligned ATE, flip accuracy), and **layout** (metric
   position error vs #rooms).
3. **Keep what is needed; fold code in, don't blindly archive.** Migrate the reusable logic from the 33
   `experiments/exp*.py` into the new modules/pipelines. Move the genuinely superseded one-offs to
   `legacy/` (kept in git, not deleted) with a short manifest. Never `rm` results or weights.
4. **Preserve behaviour.** Refactor by moving/reorganising, not rewriting algorithms. Add regression
   tests that assert the connectivity numbers reproduce (see §6).
5. **Clean the top level.** Loose result dirs, generated datasets (`data_doorpairs/`, `data_floors/`,
   `runs/`, `torchhub.tgz`, `__pycache__`) should be organised and gitignored, not scattered at root.

---

## 3. Target structure (adapt sensibly; topical folders, NOT one giant rewrite)

```
sparsepano/                     # the core library (topical subpackages, mirror overlap_probe's clarity)
  datasets/                     # <-- the de-hardcoding
    base.py                     # Pano, Door, Room, Floor, Scene dataclasses + Dataset ABC (the contract)
    zind.py                     # ZindDataset(Dataset) — migrated from former-src/zind.py
    registry.py                 # get_dataset(name, root, **cfg); decorator to register adapters
    README.md                   # "How to add a dataset: implement Dataset + register it" (with the contract)
  geometry/                     # panoproj.py (e2p), cubemap.py (new: sphere<->cubemap), depth/normals
                                #   providers.py, and a PaGeR adapter stub (foundation-model geometry)
  doors/                        # detection (semantic/geometric), embedding (contrastive), dataset builder
  pose/                         # door_pose, posegraph, register, matching, geom
  gs/                           # gsplat init/optim (Paper 2; keep, low priority)
  metrics/                      # detection.py depth.py connectivity.py pose.py layout.py (pure functions)
  viz/                          # overlays, pointcloud export (fold in tools/viewer.py + exp14)
  config.py                     # typed config loader (replaces hard-coded root config.py)
pipelines/                      # dataset-agnostic stage runners + orchestrator
  build_door_dataset.py  train_embedding.py  connectivity.py  pose_layout.py  geometry.py  evaluate.py
  run.py                        # single CLI entry (see §5)
benchmarks/
  overlap_probe/                # MOVE the existing overlap_probe here unchanged (it already fits)
configs/                        # YAML: dataset roots, thresholds, model choices, split files
scripts/                        # cluster env + slurm (keep; fix any moved paths)
legacy/                         # archived exp*.py one-offs + old one-off result dirs (kept in git)
docs/                           # move the 4 canonical ContextMDs docs here (ProjectOverview, ResearchLog,
                                #   Roadmap, RelatedWork) and DELETE the '[Deprecated — consolidated]' stub .md files
tests/                          # ported self-tests + regression checks
results/                        # gitignored; per-run <run_id>/ with numbers + plots + report.md
README.md                       # rewritten quickstart: install, add a dataset, run a stage, read a report
```

Use `pyproject.toml` (or keep `requirements.txt`) so `pip install -e .` works and imports are
`from sparsepano.datasets import get_dataset` rather than `sys.path` hacks.

---

## 4. The dataset abstraction (most important deliverable — get this right)

Define in `sparsepano/datasets/base.py` a minimal, dataset-agnostic contract. Suggested shape (refine as
needed, but keep it this small and explicit):

```python
@dataclass
class Door:      # a doorway/opening seen from one pano
    pano_id: str
    bearing_deg: float          # azimuth of door centre in the pano
    width_m: float | None
    endpoints_xy: tuple | None  # global 2D endpoints if GT geometry exists
    uid: str | None             # SAME uid across panos == same physical door (GT correspondence)

@dataclass
class Pano:
    id: str
    image_path: str
    room_id: str
    pose_c2w: np.ndarray | None    # (4,4) metric camera-to-world if GT poses exist, else None
    cam_height_m: float | None
    gt_depth_path: str | None      # if the dataset ships GT depth
    doors: list[Door]              # GT doors for this pano (may be empty / detector-filled at runtime)

@dataclass
class Scene:                       # one floor/home
    dataset: str
    scene_id: str
    panos: list[Pano]
    meters_per_unit: float
    caps: dict                     # {"gt_poses":bool,"gt_depth":bool,"gt_doors":bool,"gt_rooms":bool}

class Dataset(ABC):
    name: str
    def scenes(self, split=None) -> Iterable[Scene]: ...
    def scene(self, scene_id) -> Scene: ...
    def splits(self) -> dict: ...          # {"train":[ids], "val":[ids], "heldout":[ids]}
```

Migrate `former-former-src/zind.py` into `ZindDataset` implementing this. Keep `former-former-src/stanford.py` logic as a second
reference adapter IF it still works, otherwise move it to legacy with a `TODO(codex)` note — but the
point is the *interface*, not implementing many datasets now. Register adapters via
`registry.get_dataset("zind", root=...)`. All pipelines/metrics consume `Scene`/`Pano`/`Door` ONLY —
no direct ZInD field access anywhere outside `datasets/zind.py`.

Every hard-coded path/scale/field currently in `config.py` and the experiments must move behind either
`configs/*.yaml` (roots, thresholds, split files) or the dataset adapter. Grep the repo for `zind`,
`ZIND`, `zind_data.json`, `meters_per_coordinate`, hard-coded home ids, and route them through the
abstraction/config.

---

## 5. Unified CLI + reporting

Single entry point:

```
python -m pipelines.run --dataset zind --root <path> --split heldout \
       --stage {geometry,doors,connectivity,pose,layout,all} \
       --doors {gt,detected} --out results/<run_id>
```

Each stage writes to `results/<run_id>/`: a machine-readable `metrics.json` + `*.csv`, plots, and a
human-readable `report.md` summarising the numbers and pointing to per-scene diagnostics. The evaluate
stage aggregates all available stages into one `report.md` with a table. Reports must degrade gracefully
using `Scene.caps` (e.g. skip depth metrics when `gt_depth` is False, and say so).

Per-aspect diagnostics you must provide (this is the "analyse if doors weren't recognised / depth is bad"
requirement):
- **Doors:** per scene, list GT doors matched vs MISSED by the detector (with bearing + which pano), and
  false positives; aggregate recall/precision @ a configurable angular tolerance.
- **Depth:** per pano error vs GT depth when available, plus an error heatmap; aggregate AbsRel/RMSE/δ.
- **Connectivity:** per home, the predicted vs GT edge set with wrong/missing edges flagged; aggregate
  AP + P/R/F1 at the chosen threshold.
- **Pose/flip/layout:** the top-down GT-vs-predicted overlay (reuse `overlap_probe`'s viz idea) + relRot,
  ATE, flip accuracy, layout error vs #rooms.

---

## 6. Safety, regression, and what to preserve

- **Weights:** keep `best.pt`, `best_hardneg.pt`, `door_encoder.pt` (move to `weights/` and gitignore
  large binaries via LFS or .gitignore, but do not delete).
- **Regression test (required):** add `tests/test_connectivity_regression.py` that runs the connectivity
  eval on a small fixed ZInD subset with the shipped weights and asserts assign-AP is within tolerance of
  the known values (GT-doors ≈0.91, detected ≈0.84). If you cannot run it in CI, make it a `--slow` test
  with clear instructions. Also port the existing `--selftest` paths (exp12/exp28/exp29/exp30/exp32) into
  `tests/` as fast plumbing checks.
- **Do not touch** `overlap_probe/`'s internals beyond moving it under `benchmarks/` and fixing imports;
  it is validated and in active use.
- Keep `scripts/` (cluster env + slurm) working; update any paths you moved. The slurm/env specifics
  (Leonardo `env_leonardo.sh`, offline caches) must keep functioning.

Migrated-experiment mapping (guidance — verify before moving):
- Core → migrate into modules/pipelines: exp09 (build door dataset)→`pipelines/build_door_dataset.py`;
  exp10 (train)→`pipelines/train_embedding.py`; exp12/exp23 (connectivity+heldout)→`pipelines/connectivity.py`
  + `metrics/connectivity.py`; exp28 (GT-free connectivity)→same, `--doors detected`; exp18/exp27 (floor
  graph real)→`pipelines/pose_layout.py`; exp29 (pose/flip benchmark)→same + `metrics/pose.py`; exp30
  (floor dataset)→dataset builder; exp31/exp32 (distance)→`pose/`; exp33 (flip diagnose)→`metrics/pose.py`
  diagnostic. src modules (contrastive, doors*, door_pose, posegraph, panoproj, register, matching, geom,
  hard_neg, providers, gs_optim, gsplat_init) → the matching topical subpackage.
- Superseded one-offs → `legacy/` with a one-line note each in `legacy/README.md`: exp01–exp08, exp11,
  exp13, exp15, exp16, exp17, exp19–exp22, exp24–exp26. (Scan each for a reusable helper before moving;
  if a function is still used, migrate it, don't strand it.)

---

## 7. Process & deliverables

1. Start by writing `MIGRATION.md`: the proposed tree, and a table mapping every current file →
   {moved-to X | folded-into Y | legacy | deleted-because-generated}. Commit this first.
2. Execute in phases, committing after each and keeping tests green:
   (a) scaffolding + `pyproject.toml` + `.gitignore` (ignore results/, runs/, data_*/, *.tgz, __pycache__);
   (b) datasets abstraction + ZInD adapter + config; (c) move src modules into topical subpackages, fix
   imports; (d) pipelines + CLI; (e) metrics + reporting; (f) move overlap_probe→benchmarks, docs→docs,
   experiments→legacy; (g) regression + self-tests; (h) rewrite README + update `docs/Roadmap.md`
   pipeline stages to match §1.
3. Final deliverable: a clean tree, `pip install -e .` works, `python -m pipelines.run ... --stage
   connectivity --doors gt` reproduces ~0.91 AP, a `report.md` is produced, and `MIGRATION.md` documents
   every move. Nothing from git history is lost.

Assumptions if unspecified: prefer archive over delete; prefer moving over rewriting; keep everything
runnable at every commit; don't add new heavy dependencies without noting them in `MIGRATION.md`.
