"""Unified pipeline entry point.

This is a bridge CLI: it exposes the new dataset-agnostic interface while
delegating connectivity scoring to the migrated ``pipelines.connectivity``
implementation (formerly exp28). Other stages are scaffolded (see run_skipped_stage).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sparsepano.config import dataset_config_from_mapping, load_mapping
from sparsepano.datasets import get_dataset


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _summarise_connectivity(rows: list[dict[str, str]]) -> dict:
    if not rows:
        return {"num_scenes": 0, "ap_mean": None, "ap_std": None}
    ap = np.array([float(r["ap"]) for r in rows], dtype=float)
    f1 = np.array([float(r.get("f1", "nan")) for r in rows], dtype=float)
    out = {
        "num_scenes": int(len(rows)),
        "ap_mean": float(np.nanmean(ap)),
        "ap_std": float(np.nanstd(ap)),
        "f1_mean": float(np.nanmean(f1)),
    }
    if "det_p" in rows[0]:
        out["det_precision_mean"] = float(np.nanmean([float(r["det_p"]) for r in rows]))
        out["det_recall_mean"] = float(np.nanmean([float(r["det_r"]) for r in rows]))
    return out


def _write_report(out_dir: Path, args, metrics: dict, csv_path: Path | None = None) -> None:
    lines = [
        "# SparsePano Pipeline Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Stage: `{args.stage}`",
        f"- Split: `{args.split}`",
        f"- Doors: `{args.doors}`",
        f"- Output: `{out_dir}`",
        "",
        "## Metrics",
        "",
    ]
    if metrics.get("skipped"):
        lines.append(f"Skipped: {metrics['skipped']}")
    elif metrics.get("num_scenes", 0) == 0:
        lines.append("No evaluable scenes were produced.")
    else:
        lines.extend(
            [
                f"- Scenes: {metrics['num_scenes']}",
                f"- Mean AP: {metrics['ap_mean']:.3f}",
                f"- AP std: {metrics['ap_std']:.3f}",
                f"- Mean F1: {metrics['f1_mean']:.3f}",
            ]
        )
        if "det_precision_mean" in metrics:
            lines.extend(
                [
                    f"- Detector precision: {metrics['det_precision_mean']:.3f}",
                    f"- Detector recall: {metrics['det_recall_mean']:.3f}",
                ]
            )
    if csv_path is not None:
        lines.extend(["", "## Diagnostics", "", f"- Per-scene CSV: `{csv_path}`"])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def run_connectivity(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate the dataset early so bad roots/splits fail before GPU model load.
    ds = get_dataset(
        args.dataset,
        root=args.root,
        heldout_file=args.only,
        split_file=args.split_file,
    )
    available = ds.splits()
    if args.split not in available and args.split != "all":
        raise SystemExit(f"split {args.split!r} not available; choices: {sorted(available)}")

    os.environ["RESULTS_ROOT"] = str(out_dir)
    from sparsepano import config

    config.RESULTS_ROOT = out_dir
    from pipelines import connectivity

    tag = args.tag or f"{args.dataset}_{args.split}_{args.doors}_{args.scoring}"
    legacy_args = SimpleNamespace(
        root=args.root,
        home=None,
        only=args.only,
        ckpt=args.ckpt,
        max=args.max,
        doors=args.doors,
        scoring=args.scoring,
        fov=args.fov,
        det_fov=args.det_fov,
        n_views=args.n_views,
        tol_deg=args.tol_deg,
        det_cache=str(out_dir / "det_cache"),
        include_windows=args.include_windows,
        device=args.device,
        selftest=args.selftest,
        tag=tag,
    )
    connectivity.main(legacy_args)

    csv_path = out_dir / "gtfree" / f"gtfree_ap_{tag}.csv"
    rows = _csv_rows(csv_path)
    metrics = _summarise_connectivity(rows)
    metrics.update(
        {
            "dataset": args.dataset,
            "stage": "connectivity",
            "split": args.split,
            "doors": args.doors,
            "scoring": args.scoring,
            "csv": str(csv_path),
            "regression_target_ap": 0.913 if args.doors == "gt" else 0.842,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_report(out_dir, args, metrics, csv_path)
    return metrics


def run_skipped_stage(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": args.dataset,
        "stage": args.stage,
        "skipped": (
            "TODO(codex): this stage is scaffolded but not migrated yet. "
            "Archived legacy scripts remain under legacy/experiments/."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_report(out_dir, args, metrics)
    return metrics


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="SparsePano dataset-agnostic pipeline runner")
    ap.add_argument("--config", help="Optional JSON/YAML config with dataset.name/root.")
    ap.add_argument("--dataset")
    ap.add_argument("--root", help="Dataset root, e.g. ZInD full_dataset.")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--split_file")
    ap.add_argument("--only", help="Home-id file used as heldout split/eval filter.")
    ap.add_argument(
        "--stage",
        choices=["geometry", "doors", "connectivity", "pose", "layout", "evaluate", "all"],
        default="connectivity",
    )
    ap.add_argument("--doors", choices=["gt", "detected"], default="gt")
    ap.add_argument("--out", default="results/dev")
    ap.add_argument("--ckpt", default="weights/best.pt")
    ap.add_argument("--max", type=int, default=9999)
    ap.add_argument("--scoring", choices=["max", "assign"], default="assign")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--det_fov", type=float, default=90.0)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--tol_deg", type=float, default=15.0)
    ap.add_argument("--include_windows", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    if args.config:
        ds_cfg = dataset_config_from_mapping(load_mapping(args.config))
        args.dataset = args.dataset or ds_cfg.name
        args.root = args.root or ds_cfg.root
        args.split_file = args.split_file or ds_cfg.split_file
        args.only = args.only or ds_cfg.heldout_file
    args.dataset = args.dataset or "zind"
    if not args.root:
        ap.error("give --root or --config with dataset.root")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage in {"connectivity", "all", "evaluate"}:
        run_connectivity(args)
    else:
        run_skipped_stage(args)


if __name__ == "__main__":
    main()
