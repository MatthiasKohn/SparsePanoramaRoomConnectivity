#!/bin/bash
# =============================================================================
# paper_runs.sh — every GPU/compute run needed to close the Paper-1 gaps
# (NextStage.md §6). NOT meant to run top-to-bottom in one go: sections A–C are
# CLUSTER (bwUniCluster, ZIND_ROOT with full data, no depth needed); section D
# is the LOCAL laptop GPU (DAP lives there); E–F follow after their inputs exist.
# Run each section's block from the repo root; comment in/out what you need.
# =============================================================================
set -e
ZIND_ROOT=${ZIND_ROOT:-/home/ul/ul_student/ul_fnm03/data/zind/full_dataset}
VAL=runs/hardneg/val_homes.txt            # FROZEN 197-home test list — never retrain on it

# =============================================================================
# A) Training hygiene (G4): reproduce hard-neg run cleanly, early-stop on val_top5.
#    best.pt was epoch 15 of a run killed at 32/60 — rerun with a longer limit and
#    re-submit on timeout (--resume). ~2 h/16 epochs on A100 => request 12-24 h.
# =============================================================================
# sbatch --time=24:00:00 --wrap "python -m pipelines.train_embedding \
#     --data data_doorpairs --out runs/hardneg2 \
#     --epochs 60 --bs 128 --lr 3e-4 --wd 1e-4 --eval_every 2 --workers 8 \
#     --val_frac 0.15 --seed 0 --unfreeze --hard_neg --homes_per_batch 8"
# on timeout:   add --resume and resubmit
#
# Leakage check (must print 0 — no val home contributed training crops):
# python - <<'PY'
# import csv
# val = set(open("runs/hardneg/val_homes.txt").read().split())
# rows = list(csv.DictReader(open("data_doorpairs/pairs.csv")))
# key = "scene" if "scene" in rows[0] else list(rows[0])[0]
# leak = {r[key] for r in rows if r[key] in val}
# print(f"leaked homes: {len(leak)}", sorted(leak)[:10])
# PY

# =============================================================================
# B) exp28 — GT-FREE connectivity (G1, THE gap-closer). GPU (SegFormer+DINOv2).
#    Detections cache to results/gtfree/det_cache, so scoring re-runs are free.
#    ~8 SegFormer passes/pano; budget a few hours for 197 homes on A100.
# =============================================================================
# python -m pipelines.connectivity --root $ZIND_ROOT --only $VAL \
#     --ckpt runs/hardneg/best.pt --doors detected --scoring assign --max 200 --tag hn_det
# # oracle-door control on the SAME homes (paired comparison for the paper table):
# python -m pipelines.connectivity --root $ZIND_ROOT --only $VAL \
#     --ckpt runs/hardneg/best.pt --doors gt --scoring assign --max 200 --tag hn_gt
# # sensitivity: include windows as candidates / plain max scoring
# python -m pipelines.connectivity --root $ZIND_ROOT --only $VAL \
#     --ckpt runs/hardneg/best.pt --doors detected --scoring max --max 200 --tag hn_det_max

# =============================================================================
# C) Ablation grid (G5) — fills the method table.
# =============================================================================
# # 2x2 {frozen, unfreeze} x {std, hard-neg} (runs/full frozen-std exists; add:)
# python -m pipelines.train_embedding --data data_doorpairs --out runs/frozen_hn \
#     --epochs 60 --bs 128 --lr 3e-4 --eval_every 2 --workers 8 --hard_neg --homes_per_batch 8
# python -m pipelines.train_embedding --data data_doorpairs --out runs/ft_std \
#     --epochs 60 --bs 128 --lr 3e-4 --eval_every 2 --workers 8 --unfreeze
# for run in full frozen_hn ft_std hardneg2; do
#   for m in max assign; do
#     python -m pipelines.heldout_eval --root $ZIND_ROOT --only $VAL \
#         --ckpt runs/$run/best.pt --scoring $m --tag ${run}_${m} --max 200
#   done
# done
#
# # data-scaling curve (the key figure):
# bash scripts/scaling_curve.sh $ZIND_ROOT
#
# # crop fov 50 vs 70 (rebuild dataset, retrain, eval):
# python -m pipelines.build_door_dataset --zind_root $ZIND_ROOT --out data_doorpairs_fov50 --fov 50
# python -m pipelines.train_embedding --data data_doorpairs_fov50 --out runs/hn_fov50 \
#     --epochs 60 --bs 128 --lr 3e-4 --eval_every 2 --workers 8 --unfreeze --hard_neg --homes_per_batch 8
#
# # mechanism probe (door-region vs through-region masking) + saliency figures:
# python legacy/experiments/exp15_match_saliency.py --data data_doorpairs --ckpt runs/hardneg/best.pt --grid 12

# =============================================================================
# D) LOCAL laptop GPU — depth for the pose benchmark (G2), then exp29 (CPU).
# =============================================================================
# # 1. pick floors worth the depth (cyclic-first, held-out only; CPU, minutes):
# python scripts/find_cyclic_homes.py --root ../data/zind/full_dataset --only $VAL --top 20
# # 2. DAP depth overnight (laptop GPU):
# while read H; do
#   python scripts/generate_depth.py \
#       --input_dir  ../data/zind/full_dataset/$H/panos \
#       --output_dir ../data/zind/full_dataset/$H/dap_depth --pattern "*.jpg"
# done < scripts/depth_homes.txt
# # 3. the pose/flip benchmark table (CPU + small GPU for the embedding):
# python -m pipelines.pose_layout --root ../data/zind/full_dataset \
#     --homes scripts/depth_homes.txt --ckpt runs/hardneg/best.pt --device cuda --tag hn

# =============================================================================
# E) COLMAP hybrid (G3) — perspective-split pipeline (needs colmap on PATH).
#    Run per home: 0025 first (cyclic, depth exists), then 2-3 from depth_homes.txt.
# =============================================================================
# python scripts/colmap_perspective.py --home ../data/zind/full_dataset/0025 \
#     --n_views 12 --fov 90 --size 1024
# python -m pipelines.colmap_compare --home ../data/zind/full_dataset/0025 \
#     --model ../data/zind/full_dataset/0025/colmap_persp/pano_poses.json
# #   (mirrored layout / huge error with correct shape? rerun:  --stage recover --flip_x)
# python -m pipelines.hybrid_real --home ../data/zind/full_dataset/0025 \
#     --depth_dir ../data/zind/full_dataset/0025/dap_depth/depth_meters \
#     --ckpt runs/hardneg/best.pt \
#     --model ../data/zind/full_dataset/0025/colmap_persp/pano_poses.json

# =============================================================================
# F) VGGT qualitative baseline (cheap, motivating figure): feed one home's panos
#    (or its perspective tiles from colmap_persp/tiles) to VGGT and screenshot the
#    collapsed multi-room result. https://github.com/facebookresearch/vggt
# =============================================================================
# pip install vggt  # (or clone; see repo)  -> run on 0025 tiles, save figure to results/baselines/

echo "This file is a runbook — open it and run the section you need."
