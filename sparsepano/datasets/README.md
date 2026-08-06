# Dataset Adapters

Dataset-specific folder layouts and annotation schemas belong only in this
package. Everything else should consume:

- `Scene`
- `Pano`
- `Door`
- `Dataset`

To add a dataset:

1. Implement `Dataset.scenes()`, `Dataset.scene()`, and `Dataset.splits()`.
2. Convert native annotations into the dataclasses in `base.py`.
3. Set capability flags on every `Scene`, for example:
   `{"gt_poses": False, "gt_depth": False, "gt_doors": False, "gt_rooms": True}`.
4. Register the adapter with `@register_dataset("name")`.
5. Keep all dataset-specific field names and scale conventions inside the adapter.

Evaluators must skip metrics that require missing capabilities and explain the
skip in their report.

## Floor pipeline adapters

The legacy reconstruction pipeline consumes the narrower `ZindFloor` object
contract. `floor_factory.load_floor()` selects either that implementation or
`Structured3DFloor` without dataset checks elsewhere in the pipeline.

Structured3D expects `--home` to be one extracted scene directory:

```bash
python -m pipelines.floor \
  --dataset structured3d \
  --home /path/to/Structured3D/scene_00000 \
  --config full --pose_model gt --metrics_only
```

Use `scripts/download_structured3d.sh` on a networked login node. It extracts
archives beneath `$DEST/Structured3D/`, so the path passed to `--home` is
`$DEST/Structured3D/scene_XXXXX`. Only panorama and `annotation_3d` archives
are fetched; perspective renders are intentionally excluded.
