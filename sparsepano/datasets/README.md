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

