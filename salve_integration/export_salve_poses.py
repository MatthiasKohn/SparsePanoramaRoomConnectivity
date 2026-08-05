"""Run SALVe's back-end for one building/floor and export the global poses in OUR --pose_file format.

Runs INSIDE the `salve-v1` conda env (imports the official salve package). It replicates the exact
path of scripts/run_sfm.py up to `est_floor_pose_graph` (edge classification -> most-likely relative
pose per edge -> vanishing-angle axis alignment -> spanning tree -> pose-graph optimization), then
aligns the estimate to GT (Sim3) and writes:

  <out>/<building>_<floor>_est.json  : {stem: {rot_deg, pos:[x,y]}}  <- SALVe's ESTIMATED poses
  <out>/<building>_<floor>_gt.json   : {stem: {rot_deg, pos:[x,y]}}  <- SALVe's GT graph, same extraction

The _gt.json is a CONVENTION UNIT-TEST: feed it to `floor.py --pose_model salve` and pose RMSE must be
~0. If it is, the ZInD<->SALVe pose convention matches ours and the _est.json is trustworthy.

CLI:
  python salve_integration/export_salve_poses.py \
      --building_id 0000 --floor_id floor_01 \
      --raw_dataset_dir $ZIND_ROOT \
      --serialized_preds_json_dir <from test.py> \
      --hypotheses_save_root <from export_alignment_hypotheses.py> \
      --mhnet_predictions_data_root <MHNet preds> \
      --out <dir>
"""
import argparse, json, os
from pathlib import Path
import numpy as np

import salve.common.edge_classification as edge_classification
import salve.common.posegraph2d as posegraph2d
import salve.dataset.hnet_prediction_loader as hnet_prediction_loader
import salve.algorithms.spanning_tree as spanning_tree
import salve.algorithms.pose2_slam as pose2_slam
import salve.utils.axis_alignment_utils as axis_alignment_utils


def _pg_to_posefile(pg):
    """PoseGraph2d -> {stem: {rot_deg, pos}} in OUR pipeline's convention.

    SALVe's Sim2 (see generate_Sim2_from_floorplan_transform) works in a right-handed frame obtained
    from ZInD by an x-reflection: its angle = -(our rot_deg) and its world pano position (s*t via
    transform_from) is x-reflected vs ours. We convert back so our pose_c2w_gt reproduces GT exactly:
        rot_deg = -theta ;  pos = (-world_x, world_y)   where world = Sim2.transform_from(origin).
    Validated: feeding SALVe's GT graph through this gives pose RMSE 0 and door_gap 0."""
    out = {}
    for i, pano in pg.nodes.items():
        s = pano.global_Sim2_local
        world = s.transform_from(np.zeros((1, 2)))[0]        # s*(R@0 + t) = world pano position
        stem = Path(pano.image_path).stem if pano.image_path else str(i)
        out[stem] = {"rot_deg": float(-s.theta_deg), "pos": [float(-world[0]), float(world[1])]}
    return out


def export_floor(building_id, floor_id, raw_dataset_dir, serialized_preds_json_dir,
                 hypotheses_save_root, mhnet_predictions_data_root, out_dir,
                 confidence_threshold=0.93, use_axis_alignment=True):
    allowed_wdo_types = ["door", "window", "opening"]
    plot_save_dir = os.path.join(out_dir, "_salve_plots"); os.makedirs(plot_save_dir, exist_ok=True)

    gt_pg = posegraph2d.get_gt_pose_graph(building_id, floor_id, raw_dataset_dir)
    inferred_pg = hnet_prediction_loader.load_inferred_floor_pose_graph(
        building_id=building_id, floor_id=floor_id,
        raw_dataset_dir=raw_dataset_dir, predictions_data_root=mhnet_predictions_data_root)

    fdict = edge_classification.get_edge_classifications_from_serialized_preds(
        query_building_id=building_id, query_floor_id=floor_id,
        serialized_preds_json_dir=serialized_preds_json_dir,
        hypotheses_save_root=hypotheses_save_root,
        allowed_wdo_types=allowed_wdo_types, confidence_threshold=confidence_threshold)
    measurements = fdict[(building_id, floor_id)]
    high_conf = edge_classification.get_conf_thresholded_edge_measurements(measurements, confidence_threshold)
    if len(high_conf) == 0:
        raise SystemExit(f"[salve] no high-confidence measurements for {building_id}/{floor_id} "
                         f"(conf>={confidence_threshold}). Nothing to reconstruct.")

    i2Si1_dict, two_view_reports_dict, per_edge_wdo_dict, _ = \
        edge_classification.get_most_likely_relative_pose_per_edge(
            high_conf, hypotheses_save_root, building_id, floor_id, gt_pg)

    if use_axis_alignment:
        i2Si1_dict = axis_alignment_utils.align_pairs_by_vanishing_angle(
            i2Si1_dict=i2Si1_dict, inferred_floor_pose_graph=inferred_pg, per_edge_wdo_dict=per_edge_wdo_dict)

    wSi_list = spanning_tree.greedily_construct_st_Sim2(i2Si1_dict, verbose=False)
    wSi_list = pose2_slam.execute_planar_slam(
        measurements=high_conf, gt_floor_pg=gt_pg, hypotheses_save_root=hypotheses_save_root,
        building_id=building_id, floor_id=floor_id, wSi_list=wSi_list, plot_save_dir=plot_save_dir,
        optimize_poses_only=True, use_axis_alignment=use_axis_alignment,
        per_edge_wdo_dict=per_edge_wdo_dict, inferred_floor_pose_graph=inferred_pg)

    est_pg = posegraph2d.PoseGraph2d.from_wSi_list(wSi_list, gt_pg)
    aligned_est, _ = est_pg.align_by_Sim3_to_ref_pose_graph(ref_pose_graph=gt_pg)

    n_est = len(aligned_est.nodes); n_gt = len(gt_pg.nodes)
    print(f"[salve] {building_id}/{floor_id}: localized {n_est}/{n_gt} panos")

    os.makedirs(out_dir, exist_ok=True)
    est_fp = os.path.join(out_dir, f"{building_id}_{floor_id}_est.json")
    gt_fp = os.path.join(out_dir, f"{building_id}_{floor_id}_gt.json")
    json.dump(_pg_to_posefile(aligned_est), open(est_fp, "w"), indent=2)
    json.dump(_pg_to_posefile(gt_pg), open(gt_fp, "w"), indent=2)
    print(f"[salve] wrote {est_fp}\n[salve] wrote {gt_fp}  (feed this to floor.py; pose RMSE must be ~0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building_id", required=True)
    ap.add_argument("--floor_id", required=True)
    ap.add_argument("--raw_dataset_dir", required=True)
    ap.add_argument("--serialized_preds_json_dir", required=True)
    ap.add_argument("--hypotheses_save_root", required=True)
    ap.add_argument("--mhnet_predictions_data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confidence_threshold", type=float, default=0.93)
    a = ap.parse_args()
    export_floor(a.building_id, a.floor_id, a.raw_dataset_dir, a.serialized_preds_json_dir,
                 a.hypotheses_save_root, a.mhnet_predictions_data_root, a.out, a.confidence_threshold)


if __name__ == "__main__":
    main()
