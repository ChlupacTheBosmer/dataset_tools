# Module Guide

This guide explains each module family, what it is responsible for, and what must be configured for it to work correctly.

For complete callable signatures and per-function doc summaries, use `docs/dst/api-reference.md`.

## Top-Level Modules

## `dataset_tools/__init__.py`

Purpose:
- stable, minimal public exports (`AppConfig`, `load_config`, `sync_corrections_to_disk`)

Use when:
- another module/script wants a short import path for these stable entrypoints

## `dataset_tools/dst.py`

Purpose:
- unified CLI parser and command handlers

What it requires:
- Python environment with FiftyOne and optional provider dependencies
- Label Studio connectivity for `ls` / workflow commands

Key behavior:
- all leaf commands map to `cmd_*`
- handlers return JSON-like payloads
- `main()` converts runtime/config errors to CLI errors

## `dataset_tools/config.py`

Purpose:
- typed runtime configuration and precedence-aware loading

What it requires:
- optional local config JSON and/or env variables

Important contract:
- all runtime modules should consume `AppConfig` rather than ad hoc global constants

## Label Studio Family (`dataset_tools/label_studio/*`)

## `client.py`

Purpose:
- SDK import compatibility and authenticated client creation

Requires:
- reachable LS URL
- valid API key

## `storage.py`

Purpose:
- project creation/retrieval and source/target storage assurance

Requires:
- valid mount mapping to resolve media URLs/files

## `uploader.py`

Purpose:
- monkeypatch support for robust batched task upload

Requires:
- LS client + FiftyOne annotate compatibility

## `translator.py`

Purpose:
- conversion between FiftyOne detection representation and LS rectangle result schema

Requires:
- consistent label naming and coordinate assumptions

## `sync.py`

Purpose:
- send/pull orchestration between FiftyOne views and LS projects

Requires:
- correctly configured project + storages + mapping strategy

Notes:
- strategy choice (`annotate_batched` vs `sdk_batched`) affects task metadata and pull behavior

## Data Loading Family

## `dataset_tools/loaders/base.py`

Purpose:
- common loader contracts (`LoaderResult`, `BaseDatasetLoader`)

## `dataset_tools/loaders/path_resolvers.py`

Purpose:
- path strategies for mirrored roots or `images/` + `labels/` layouts

Requires:
- predictable filesystem layout per resolver choice

## `dataset_tools/loaders/yolo.py`

Purpose:
- recursive YOLO loader with optional confidence-column handling

Requires:
- valid class map for readable labels (optional but recommended)

## `dataset_tools/loaders/coco.py`

Purpose:
- COCO dataset loader with explicit config object

## `dataset_tools/loader.py`

Purpose:
- convenience wrapper layer around loaders package

Guidance:
- new code should prefer loaders package directly; wrappers exist for convenience/transition

## Metrics Family (`dataset_tools/metrics/*`)

Core idea:
- each computation writes dataset sample fields and returns metadata about the run

Common requirement:
- target dataset exists and required fields are present

Modules:
- `embeddings.py`: embeddings + optional projection/cluster enrichment
- `uniqueness.py`: uniqueness field population
- `mistakenness.py`: mistakenness + missing/spurious fields
- `hardness.py`: hardness scores for classification-style labels
- `representativeness.py`: representativeness scoring
- `field_metric.py`: required-field validation scaffold
- `base.py`: base metric runner contract

## Brain Family (`dataset_tools/brain/*`)

Core idea:
- wrappers around FiftyOne Brain run/index APIs, with reusable base operation semantics

Modules:
- `visualization.py`: dimensionality reduction runs
- `similarity.py`: similarity index runs
- `duplicates.py`: exact/near duplicate detection
- `leaky_splits.py`: split leakage detection
- `base.py`: shared operation behavior

## Model Provider Family (`dataset_tools/models/*`)

Purpose:
- normalize model references and hide provider-specific loading details

Modules:
- `spec.py`: `ModelRef`, `LoadedModel`, parsing helpers
- `base.py`: provider interface
- `registry.py`: provider registry/dispatch
- `providers/huggingface.py`: HF embedding model adapter
- `providers/fiftyone_zoo.py`: FiftyOne model zoo adapter
- `providers/anomalib.py`: anomalib model adapter

Requires:
- provider-specific dependencies installed for selected provider

## Anomaly Family (`dataset_tools/anomaly/*`)

## `pipeline.py`

Purpose:
- backend-agnostic orchestration for fit/score runs

Backends:
- embedding-distance (lightweight, no training)
- anomalib artifact scoring path

## `anomalib.py`

Purpose:
- tutorial-aligned anomalib dataset prep, training/export, and artifact scoring

Requires:
- anomalib runtime dependencies
- valid normal/abnormal sample selection (tags/fields)

## `base.py`

Purpose:
- serializable reference object for embedding-distance backend

## Workflow Family

## `dataset_tools/tag_workflow/*`

Purpose:
- generalized rule engine to execute operations per tag/view

Modules:
- `config.py`, `context.py`: workflow schema/context
- `engine.py`: runner
- `operations/core.py`: mutation + LS roundtrip ops
- `operations/analysis.py`: metric/brain/anomaly operations
- `operations/base.py`: operation interface

Requires:
- operations registry alignment with rule names
- dataset/field names in rules must exist

## `dataset_tools/workflows/roundtrip.py`

Purpose:
- predefined end-to-end curation loop (send -> pull -> optional sync)

Requires:
- LS + mount config correctness
- selected strategy compatibility with pull method

## Sync Helper

## `dataset_tools/sync_from_fo_to_disk.py`

Purpose:
- write corrections back to disk label files with backup support

Requires:
- correct path replacement mapping and class-id mapping

Safety:
- run in dry-run first

## Utility/Debug

## `dataset_tools/label_studio_json.py`

Purpose:
- build LS import JSON tasks directly from filesystem layout

## `dataset_tools/debug/*`

Purpose:
- diagnostics utilities for local debugging

Guidance:
- not stable API; avoid importing into production workflows
