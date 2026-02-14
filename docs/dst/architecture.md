# Dataset Tools Architecture

This document explains how `dataset_tools` is structured so you can safely extend or refactor it without breaking production workflows.

## Design Goals

- Single operational entrypoint: everything callable via `dst`
- Separation of concerns: adapters, domain operations, orchestration, and CLI are split by layer
- Reuse first: rodent- or dataset-specific behavior should live in config/rules, not core implementations
- Safe automation: commands return structured payloads and support dry-run patterns where possible

## Layer Model

1. configuration layer
- `dataset_tools/config.py`
- Owns runtime schema (`AppConfig`) and config resolution precedence.

2. integration adapters
- `dataset_tools/label_studio/*`
- `dataset_tools/models/*`
- Convert between external APIs (Label Studio SDK, model providers) and internal normalized contracts.

3. domain operations
- `dataset_tools/loaders/*`
- `dataset_tools/metrics/*`
- `dataset_tools/brain/*`
- `dataset_tools/anomaly/*`
- Encapsulate concrete operations over FiftyOne datasets/views.

4. orchestration
- `dataset_tools/tag_workflow/*`
- `dataset_tools/workflows/roundtrip.py`
- Compose operations into multi-step workflows with rule/config-driven behavior.

5. CLI surface
- `dataset_tools/dst.py`
- Parses args, validates combinations, dispatches to command handlers, prints/writes results.

## Dependency Direction

Dependency arrows should flow inward:

- `dst.py` -> any orchestration/domain/integration module
- orchestration -> domain + integration + config
- domain -> provider/integration utilities as needed
- integration -> config (and external SDKs)
- config -> no workflow/domain dependencies

This keeps orchestration replaceable and makes domain modules testable independently of CLI.

## Runtime Flows

## 1) Dataset Load

1. `dst data load yolo|coco`
2. Path resolver maps media -> annotation files
3. Loader creates/updates FiftyOne dataset and fields

Core modules:
- `dataset_tools/loaders/path_resolvers.py`
- `dataset_tools/loaders/yolo.py`
- `dataset_tools/loaders/coco.py`

## 2) Curation Enrichment

1. Compute embeddings and metrics
2. Run Brain indexes/analyses
3. Persist sample fields and brain runs

Core modules:
- `dataset_tools/metrics/*`
- `dataset_tools/brain/*`
- `dataset_tools/models/*` (for model resolution)

## 3) Labeling Roundtrip

1. Select view/tag and push tasks to LS
2. Annotate in LS
3. Pull completed tasks back to FiftyOne fields
4. Optional sync back to disk labels

Core modules:
- `dataset_tools/label_studio/sync.py`
- `dataset_tools/tag_workflow/operations/core.py`
- `dataset_tools/workflows/roundtrip.py`
- `dataset_tools/sync_from_fo_to_disk.py`

## 4) Anomaly Workflows

Backends:
- `embedding_distance`: lightweight fit+score using embeddings already in dataset
- `anomalib`: train/export artifact and infer with artifact runtime

Core modules:
- `dataset_tools/anomaly/pipeline.py`
- `dataset_tools/anomaly/anomalib.py`

## Extension Points

- Add a new metric: implement in `dataset_tools/metrics/` and wire to `dst metrics ...`
- Add a new brain operation: implement in `dataset_tools/brain/` and wire to `dst brain ...`
- Add new workflow actions: implement `TagOperation` and register in `default_operations_registry()`
- Add model backend: implement `ModelProvider` and register in `dataset_tools/models/registry.py`

## Non-Production Scope

- `dataset_tools/debug/*` contains diagnostics scripts and is intentionally not part of stable production API.
