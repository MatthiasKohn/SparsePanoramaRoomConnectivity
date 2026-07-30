"""Generate SALVe W/D/O alignment hypotheses for ONE building (the official CLI only does whole splits).

Reuses SALVe's exact per-building routine, so it's faithful -- just scoped to one building so we
don't process an entire ZInD split. Runs inside the salve-v1 conda env.

  python salve_integration/gen_hypotheses_one_building.py \
      --building_id 0021 --raw_dataset_dir $ZIND_ROOT \
      --hypotheses_save_root <out> --wdo_source horizon_net \
      --mhnet_predictions_data_root <MHNet preds>
"""
import argparse
from pathlib import Path
# export_alignment_hypotheses.py lives in SALVe's top-level scripts/ (not the installed package),
# so the slurm puts $SALVE_ROOT/scripts on PYTHONPATH and we import it as a plain module.
from export_alignment_hypotheses import export_single_building_wdo_alignment_hypotheses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building_id", required=True)
    ap.add_argument("--raw_dataset_dir", required=True)
    ap.add_argument("--hypotheses_save_root", required=True)
    ap.add_argument("--wdo_source", choices=["horizon_net", "ground_truth"], default="horizon_net")
    ap.add_argument("--mhnet_predictions_data_root", default=None)
    a = ap.parse_args()
    use_inferred = a.wdo_source == "horizon_net"
    json_annot = f"{a.raw_dataset_dir}/{a.building_id}/zind_data.json"
    if not Path(json_annot).exists():
        raise SystemExit(f"missing {json_annot}")
    export_single_building_wdo_alignment_hypotheses(
        hypotheses_save_root=a.hypotheses_save_root,
        building_id=a.building_id,
        json_annot_fpath=json_annot,
        raw_dataset_dir=a.raw_dataset_dir,
        use_inferred_wdos_layout=use_inferred,
        mhnet_predictions_data_root=a.mhnet_predictions_data_root,
    )
    print(f"[salve] hypotheses done for building {a.building_id} -> {a.hypotheses_save_root}")


if __name__ == "__main__":
    main()
