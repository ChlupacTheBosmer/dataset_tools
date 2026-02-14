# dst -> od_training Handoff Contract

When exporting curated datasets from `dst` for `od_training`, keep these
parameters explicit and aligned.

## Required Parameters

- dataset name in FiftyOne
- label field used for finalized detections
- split tags (`train`, `val`, `test` or custom)
- export media mode (`symlink` default or `copy`)

## Export Expectations

`dst` should produce:

- YOLO structure (`images/*`, `labels/*`, `dataset.yaml`)
- split-specific COCO sidecars for RF-DETR compatibility

## Mapping Into `od_training`

- label field -> `odt dataset manage --label-field`
- class discovery field -> `odt dataset manage --classes-field`
- split tags -> `--train-tag`, `--val-tag`, `--test-tag`
- portable media -> `--copy-images` (if symlinks are not acceptable)
- confidence column -> `--include-confidence` (only when 6-column YOLO labels are desired)

## Copy-Paste Handoff

```bash
# Export curated labels for training
odt dataset manage \
  --name <fiftyone_dataset_name> \
  --export-dir runs/export/<run_name> \
  --label-field <curated_label_field> \
  --train-tag <train_tag> --val-tag <val_tag> --test-tag <test_tag>
```

## Safety Notes

- Avoid mixing unfinished and corrected fields in one export.
- Keep one label field per training run for reproducibility.
