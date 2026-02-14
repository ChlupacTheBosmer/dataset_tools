# First-Time Agent Launchpad

This document is the operational onboarding protocol for agents working on `dataset_tools`.

## 1. Mission and Boundaries

Mission:

- maintain and extend a reusable curation toolkit centered on FiftyOne + Label Studio
- keep production workflow reliable (especially batched LS upload + pull mapping + disk sync safety)

Boundaries:

- `dataset_tools` should contain reusable logic, not one-off dataset hacks
- ad hoc dataset-specific logic belongs in consuming repositories, not here

## 2. Repository Mental Model

`dataset_tools` has a layered architecture:

- interface layer: `src/dataset_tools/dst.py` (single CLI entrypoint)
- domain/service modules:
  - `loaders/`, `metrics/`, `brain/`, `models/`, `anomaly/`, `label_studio/`, `tag_workflow/`, `workflows/`
- integration/safety modules:
  - `config.py`, `sync_from_fo_to_disk.py`
- docs/tests as first-class artifacts:
  - `docs/dst/*`, `tests/*`

## 3. Critical Invariants (do not break)

1. `sdk_batched` upload path must stay stable.
2. task metadata mapping (`fiftyone_id`) must remain valid for pull.
3. disk sync must support `--dry-run` and backup behavior.
4. config must remain secret-safe and portable (`DST_CONFIG_PATH`, local config fallback chain).
5. `dst` CLI remains single source of operational entrypoints.

## 4. Required Context Pass Before Coding

Read these in order:

1. `README.md`
2. `docs/dst/architecture.md`
3. `docs/dst/module-guide.md`
4. `docs/dst/configuration.md`
5. `docs/dst/cli-guide.md`
6. `docs/dst/test-suite.md`

Then inspect source modules relevant to task.

## 5. Decision Framework for New Code

When adding functionality, decide placement by responsibility:

- sample/detection field computation -> `metrics/`
- Brain runs/indexes -> `brain/`
- model resolution/backends -> `models/`
- LS transfer/project/storage logic -> `label_studio/`
- multi-step process composition -> `workflows/` or `tag_workflow/`
- user-facing command surface -> extend `dst.py`

If unsure, prefer composition over copy-paste logic.

## 6. Validation Protocol

Minimum for any non-trivial change:

1. targeted tests for touched modules
2. `python -m pytest -q tests`
3. if workflow-affecting: run checklist in `docs/dst/context/manual_validation_checklist.md`

For risky integration changes:

- include small smoke-run evidence in PR/notes
- mention what was not tested (if anything)

## 7. Warning Policy During Tests

- first-party deprecations from `dataset_tools` are treated as errors
- known third-party deprecation noise is filtered in `pyproject.toml`
- new warnings from package code should be treated as regressions to fix

## 8. Safe Operational Behavior

- no secret writes into tracked files
- no destructive dataset operations in tests without explicit isolation
- use throwaway dataset names and dry-run modes for validation
- never assume environment paths are fixed; keep scripts portable

## 9. High-Value References

- architecture: `docs/dst/architecture.md`
- module responsibilities: `docs/dst/module-guide.md`
- CLI details: `docs/dst/cli-reference.md`
- API surface map: `docs/dst/api-reference.md`
- FiftyOne references: `docs/dst/context/fiftyone_reference_hub.md`
- Label Studio references: `docs/dst/context/label_studio_reference_hub.md`
- LS integration details: `docs/dst/context/label_studio_integration_notes.md`

## 10. First Session Checklist

1. run `dst --help`
2. run `dst config show`
3. skim `tests/test_dst.py` and `tests/test_dst_commands.py` for CLI contract shape
4. inspect target modules to change
5. implement smallest coherent change
6. run tests
7. update docs if behavior/contract changed
