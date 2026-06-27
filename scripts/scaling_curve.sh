#!/bin/bash
# Data-scaling curve: held-out connectivity AP vs #training homes.
# Builds a capped dataset, trains a short run, evaluates on held-out homes, for
# several dataset sizes. Run on a GPU node (or sbatch a wrapper around it).
#
#   bash scripts/scaling_curve.sh /path/to/zind/full_dataset
set -e
ZIND_ROOT=${1:?"give ZInD root dir"}
SIZES="50 150 400 9999"          # 9999 = all available homes
EPOCHS=30

for N in $SIZES; do
  echo "=== N=$N homes ==="
  python experiments/exp09_build_door_dataset.py --zind_root "$ZIND_ROOT" \
      --out data_scale_$N --max_homes $N
  python experiments/exp10_train_contrastive.py --data data_scale_$N \
      --out runs/scale_$N --epochs $EPOCHS --bs 128 --eval_every 5 --workers 8
  echo "--- connectivity on held-out homes (N=$N) ---"
  python experiments/exp12_connectivity_graph.py --root "$ZIND_ROOT" \
      --only runs/scale_$N/val_homes.txt --ckpt runs/scale_$N/best.pt \
      | tee runs/scale_$N/connectivity.txt
done
echo "Collect the 'MEAN AP' line from each runs/scale_*/connectivity.txt -> plot vs N."
