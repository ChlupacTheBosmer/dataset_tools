# Dataset Tools Test Suite

This document explains what is tested, how tests are executed, and how to manually verify that the suite aligns with intended production behavior.

## Scope and Intent

The authoritative project test suite for `dataset_tools` is the `tests/` directory at repo root.

The suite focuses on:

- command/API correctness for `dst` and module functions
- integration contracts between loaders, metrics, Brain ops, LS sync, anomaly workflows
- regression protection for known fragile behavior (batched LS upload, pull mapping, storage/mount checks)
- real-life end-to-end flows using generated fixtures and optional network/model downloads

## Test Environment

Activate the Python environment for your setup before running tests.

`pytest` and `pytest-cov` are expected to be available in that environment.

## How to Run

## Recommended full project suite

```bash
python -m pytest -q tests
```

## With coverage report

```bash
python -m pytest -q tests --cov=dataset_tools --cov-report=term-missing
```

## Include optional real-life integration tests

```bash
RUN_REAL_LIFE_TESTS=1 python -m pytest -q tests/test_real_life_suite.py
```

## Include optional network/model-download tests

```bash
RUN_REAL_LIFE_TESTS=1 RUN_REAL_LIFE_NETWORK_TESTS=1 \
python -m pytest -q tests/test_real_life_suite.py
```

## Important: do not use plain `pytest` at repo root

`python -m pytest -q` at repo root also collects vendored/experimental test trees (`ref_libs/*`, `experiments/*`) and may fail on unrelated optional dependencies. For `dataset_tools` validation, run `tests/` explicitly.

## Latest Results (February 13, 2026)

Command:

```bash
python -m pytest -q tests
```

Result:

- `138 passed`
- `3 skipped`
- `36 subtests passed`

Skip reasons:

- `RUN_REAL_LIFE_TESTS` not set (2 tests)
- `RUN_REAL_LIFE_NETWORK_TESTS` not set (1 test)

Coverage command result:

- `TOTAL dataset_tools coverage: 84%`

Additional optional runs executed on February 13, 2026:

- `RUN_REAL_LIFE_TESTS=1 python -m pytest -q tests/test_real_life_suite.py -k "not network"` -> `2 passed, 1 deselected`
- `RUN_REAL_LIFE_TESTS=1 RUN_REAL_LIFE_NETWORK_TESTS=1 python -m pytest -q tests/test_real_life_suite.py` -> `3 passed`

## Test Inventory by File

## Core config/CLI

- `tests/test_config.py`
  - Config precedence (defaults/local/env/overrides)
  - Default upload strategy behavior
- `tests/test_dst.py`
  - Parser wiring
  - config masking/reveal behavior
  - helper parsing/path logic
  - optional log capture and output writing
- `tests/test_dst_commands.py`
  - Command handler dispatch for data/metrics/brain/models/anomaly/workflow/sync/app
  - `main()` error/success behavior

## Label Studio integration

- `tests/test_label_studio_client.py`
  - SDK import compatibility paths
  - token resolution and connection checks
- `tests/test_label_studio_storage_uploader.py`
  - project/storage bootstrap behavior
  - uploader monkeypatch and batched upload paths
- `tests/test_label_studio_sync.py`
  - annotate push path and preflight validation
- `tests/test_label_studio_annotation_run_pull.py`
  - annotation-run pull mapping path and missing-run failures
- `tests/test_translator.py`
  - FiftyOne <-> LS rectangle conversion
- `tests/test_label_studio_json_main.py`
  - LS import JSON generation from YOLO layout

## Data loading and sync

- `tests/test_loaders_and_wrappers.py`
  - loader contracts, resolvers, YOLO/COCO loading, wrapper compatibility
- `tests/test_sync_utils.py`
  - path inference helper rules
- `tests/test_sync_from_fo_to_disk_extended.py`
  - backup creation, dry-run behavior, disk write path, missing dataset failure

## Metrics and Brain

- `tests/test_metrics_operations.py`
  - required-field guards and metric behavior
  - label-type and min-sample guards
- `tests/test_metrics_embeddings_extended.py`
  - embedding model adapters and embeddings workflow branches
- `tests/test_brain_operations.py`
  - visualization/similarity/duplicates/leaky-splits wrappers and validation
- `tests/test_fiftyone_contracts.py`
  - expected FiftyOne API signature contracts

## Models and anomaly

- `tests/test_models_registry.py`
  - model ref parsing/normalization/provider dispatch/capability checks
- `tests/test_anomaly_pipeline.py`
  - embedding-distance reference fit/score/run and anomalib delegation
- `tests/test_anomalib_workflow.py`
  - anomalib prep/train/score helpers and trust-remote-code guard

## Workflow engine integration

- `tests/test_tag_workflow_core_operations.py`
  - delete/move/send/pull/sync core operations and registry composition
- `tests/test_tag_workflow_analysis.py`
  - analysis ops (uniqueness/hardness/representativeness/similarity/duplicates/leaky/anomaly)
- `tests/test_workflows_roundtrip.py`
  - roundtrip workflow rule construction and skip-flag behavior

## Real-life integration

- `tests/test_real_life_suite.py`
  - end-to-end flows using generated dummy datasets/media/annotations
  - optional network-backed model-zoo/HF download and inference verification

## Manual Review Checklist

Use this checklist to verify the suite still matches intended behavior:

1. Confirm CLI contract coverage:
   - inspect `tests/test_dst.py` and `tests/test_dst_commands.py` for every `dst` command family.
2. Confirm fragile LS paths are covered:
   - verify tests for `sdk_batched`, annotation-run pull mapping, and storage preflight exist and pass.
3. Confirm disk-sync safety:
   - ensure dry-run and backup behavior are explicitly tested.
4. Confirm workflow composition:
   - validate `roundtrip` and tag-engine operation coverage includes both happy and failure paths.
5. Confirm anomaly/model paths:
   - review both embedding-distance and anomalib tests, including trust gating.
6. Confirm real-life tests are runnable when desired:
   - run with environment flags and inspect resulting artifacts/log output.

## Recommended CI Commands

```bash
python -m pytest -q tests
python -m pytest -q tests --cov=dataset_tools --cov-report=term-missing
```
