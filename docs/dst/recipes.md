# Operational Recipes

These recipes are production-oriented command flows for `dst`.

Before running recipes, make sure your Python environment is active and dependencies are installed.

```bash
./dst config show
```

## 1) Load YOLO Dataset (single root)

Use when your data layout is `<root>/images` and `<root>/labels`.

```bash
./dst data load yolo \
  --dataset visitors_dataset \
  --root /path/to/dataset_root \
  --overwrite
```

## 2) Load YOLO Dataset (mirrored roots)

Use when images and labels are in separate mirrored trees.

```bash
./dst data load yolo \
  --dataset visitors_dataset \
  --images-root /path/to/images_root \
  --labels-root /path/to/labels_root
```

## 3) Run Curation Metrics

```bash
./dst metrics embeddings --dataset visitors_dataset --model-ref hf:facebook/dinov2-base
./dst metrics uniqueness --dataset visitors_dataset --embeddings-field embeddings
./dst metrics mistakenness --dataset visitors_dataset --pred-field predictions --gt-field ground_truth
./dst metrics representativeness --dataset visitors_dataset --embeddings-field embeddings
```

## 4) Run Brain Analyses

```bash
./dst brain visualization --dataset visitors_dataset --method umap --brain-key umap_main
./dst brain similarity --dataset visitors_dataset --embeddings-field embeddings --brain-key sim_main
./dst brain duplicates near --dataset visitors_dataset --threshold 0.2 --embeddings-field embeddings
./dst brain leaky-splits --dataset visitors_dataset --splits train,val,test --threshold 0.2
```

## 5) Label Studio Roundtrip (recommended path)

Use `sdk_batched` for robust task upload at larger batch sizes.

```bash
./dst workflow roundtrip \
  --dataset visitors_dataset \
  --tag fix \
  --project visitors_fix_batch_001 \
  --upload-strategy sdk_batched \
  --strict-preflight \
  --output-json /tmp/roundtrip_result.json
```

## 6) Tag Workflow From JSON

```bash
./dst workflow tags run \
  --workflow examples/tag_workflow.example.json \
  --dataset visitors_dataset \
  --output-json /tmp/tag_workflow_result.json
```

## 7) Disk Sync (safe first, then write)

Dry-run:

```bash
./dst sync disk --dataset visitors_dataset --tag fix --dry-run
```

Write:

```bash
./dst sync disk --dataset visitors_dataset --tag fix
```

## 8) Anomaly (embedding-distance backend)

```bash
./dst anomaly run \
  --dataset visitors_dataset \
  --backend embedding_distance \
  --embeddings-field embeddings \
  --normal-tag normal \
  --score-field anomaly_score \
  --flag-field is_anomaly \
  --reference-json /tmp/embedding_distance_reference.json
```

## 9) Anomaly (anomalib backend)

Train/export artifact:

```bash
./dst anomaly train \
  --dataset visitors_dataset \
  --model-ref anomalib:padim \
  --normal-tag normal \
  --abnormal-tag abnormal \
  --artifact-format torch \
  --artifact-dir /tmp/anomaly_artifacts \
  --max-epochs 1 \
  --overwrite-data
```

Score with artifact:

```bash
./dst anomaly score \
  --dataset visitors_dataset \
  --backend anomalib \
  --artifact /tmp/anomaly_artifacts/anomalib_artifact.json \
  --trust-remote-code \
  --score-field anomaly_score \
  --flag-field is_anomaly \
  --label-field anomaly_label \
  --map-field anomaly_map \
  --mask-field anomaly_mask
```

## 10) Label Studio Project Hygiene

```bash
./dst ls test --list-projects
./dst ls project list --contains visitors --with-task-count
./dst ls project cleanup --keyword tmp --dry-run
```

## 11) Open FiftyOne App

```bash
./dst app open --dataset visitors_dataset --port 5151 --address 0.0.0.0
```
