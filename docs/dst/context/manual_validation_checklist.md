# Manual Validation Checklist

Use this checklist for real-world validation before production use on important datasets.

## 0. Safety Preconditions

- Never start on irreplaceable production datasets.
- Use isolated test dataset/project names first.
- Prefer dry-run modes before write operations.

## 1. CLI and Environment

- `./dst --help`
- `./dst config show`
- `./dst ls test --list-projects` (when LS is configured)

Expected:

- commands run without import/config errors
- config is resolved as expected

## 2. Data Load and App Loop

- load a small YOLO dataset
- open app and tag samples (`fix`, `delete`, `duplicate`)

Expected:

- dataset fields and tags are visible
- selected tags persist

## 3. Label Studio Send/Pull

Send only:

```bash
./dst workflow roundtrip --dataset <dataset> --tag fix --project <project> --skip-pull --skip-sync-disk
```

Pull only:

```bash
./dst workflow roundtrip --dataset <dataset> --tag fix --project <project> --skip-send --skip-sync-disk
```

Expected:

- tasks created in LS with valid images
- pull writes corrections to configured corrections field

## 4. Disk Sync Safety

Dry run:

```bash
./dst sync disk --dataset <dataset> --tag fix --dry-run
```

Real run:

```bash
./dst sync disk --dataset <dataset> --tag fix
```

Expected:

- dry-run reports expected files
- write run updates labels and creates backups

## 5. Analysis Regression Checks

Run at least one of each class:

- metrics: embeddings, uniqueness, mistakenness
- brain: visualization, similarity, duplicates near, leaky-splits
- anomaly: embedding-distance run (and anomalib path if enabled)

Expected:

- output fields/runs/indexes created
- no cross-command regressions in existing fields

## 6. Project Hygiene

- avoid reusing mixed-history LS projects without clearing tasks
- if reusing, clear tasks first
- keep one batch per project for reliable pull behavior
