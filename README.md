# dataset_tools (`dst`)

This repository is the standalone home for your `dataset_tools` package and its unified `dst` CLI.

It exists so your FiftyOne + Label Studio curation tooling is no longer coupled to the larger `dino_test` monorepo.

## What This Repo Is For

`dataset_tools` is the operational toolkit you use to:

- load datasets (YOLO/COCO) into FiftyOne
- compute curation metrics and Brain analyses
- run robust batched FiftyOne -> Label Studio -> FiftyOne roundtrips
- sync reviewed corrections back to disk labels safely
- run anomaly workflows (embedding distance + anomalib path)
- orchestrate repeatable tag-driven workflows through one CLI (`dst`)

## 5-Minute Memory Refresh (when returning later)

1. confirm env and CLI:
   - `dst --help`
   - `dst config show`
2. open docs entrypoint:
   - `docs/dst/index.md`
3. if debugging workflow logic:
   - `docs/dst/architecture.md`
   - `docs/dst/module-guide.md`
4. if operating on real data:
   - `docs/dst/recipes.md`
   - `docs/dst/context/manual_validation_checklist.md`
5. if an agent is doing the work:
   - `docs/dst/agent/index.md`

Cross-repo references:
- `docs/dst/context/cross_repo_compatibility_matrix.md`
- `docs/dst/context/od_training_handoff_contract.md`

## Repository Layout

- `src/dataset_tools/`
  - package source code
- `tests/`
  - automated test suite (unit + integration-style + optional real-life)
- `docs/dst/`
  - production docs for architecture, CLI, recipes, tests, and context hubs
- `docs/dst/agent/`
  - first-time agent onboarding and execution guidance
- `scripts/`
  - utility scripts (coverage, precode gate, docs generation, app launch)
- `examples/`
  - workflow/config examples
- `local_config.example.json`
  - local config template for secrets and environment-specific settings

## Package Capability Map

### Data loading

- `data load yolo`
- `data load coco`
- wrapper utilities in `dataset_tools.loaders` and `dataset_tools.loader`

### Metrics (field-populating operations)

- embeddings
- uniqueness
- mistakenness
- hardness
- representativeness

### FiftyOne Brain operations

- visualization
- similarity
- duplicates (exact/near)
- leaky-splits

### Label Studio integration

- LS client + token handling
- project/storage bootstrapping
- batched upload (`sdk_batched` recommended)
- pull mapping back to FiftyOne sample IDs

### Workflow orchestration

- `workflow roundtrip`
- `workflow tags run`
- `workflow tags inline`

### Disk synchronization

- correction-field -> YOLO label files
- dry-run support and backups

### Models and anomaly

- model providers (HF, FiftyOne zoo, anomalib provider wiring)
- embedding-distance anomaly path
- anomalib train/score path

## CLI Command Families

- `dst config`
- `dst data`
- `dst metrics`
- `dst brain`
- `dst models`
- `dst ls`
- `dst workflow`
- `dst sync`
- `dst anomaly`
- `dst app`

For exhaustive arguments, see `docs/dst/cli-reference.md`.

## Installation

## Editable install (recommended)

```bash
python -m pip install -e .
```

## Optional extras

```bash
python -m pip install -e .[dev]
python -m pip install -e .[anomaly]
```

## Local wrapper execution

```bash
./dst --help
```

## Installed console script

```bash
dst --help
```

## Configuration and Secrets

Default config resolution order:

1. `--config <path>` (if provided)
2. `DST_CONFIG_PATH`
3. package-local `src/dataset_tools/local_config.json` (if present)
4. `~/.config/dst/local_config.json` (or `$XDG_CONFIG_HOME/dst/local_config.json`)

Recommended setup:

```bash
mkdir -p ~/.config/dst
cp local_config.example.json ~/.config/dst/local_config.json
```

`local_config.json` is intentionally gitignored.

## Typical Production Workflow

1. load dataset:
   - `dst data load yolo ...`
2. inspect/tag in FiftyOne app:
   - `dst app open --dataset ...`
3. send tagged samples to LS:
   - `dst workflow roundtrip --skip-pull --skip-sync-disk ...`
4. annotate in LS UI
5. pull corrections:
   - `dst workflow roundtrip --skip-send --skip-sync-disk ...`
6. dry-run sync to disk:
   - `dst sync disk --dry-run ...`
7. real sync:
   - `dst sync disk ...`

Copy/paste command variants are in `docs/dst/recipes.md`.

## Testing

## Main suite

```bash
python -m pytest -q tests
```

## Coverage

```bash
python -m pytest -q tests --cov=dataset_tools --cov-report=term-missing
```

## Real-life suite (optional)

```bash
RUN_REAL_LIFE_TESTS=1 python -m pytest -q tests/test_real_life_suite.py
```

Detailed test documentation: `docs/dst/test-suite.md`.

## Warnings During Tests

Current warning output is dominated by third-party deprecations in pinned dependencies (`setuptools/pkg_resources`, `fiftyone/strawberry`, `torchao`).

Status:

- no warning indicates a current functional break in `dataset_tools`
- tests are passing (`138 passed, 3 skipped` in baseline run)
- first-party deprecations from `dataset_tools` are configured to fail tests

Policy:

- third-party known warnings are filtered in `pyproject.toml`
- if a new warning appears from `dataset_tools`, treat as regression and fix

## Documentation Entry Points

- `docs/dst/index.md`
- `docs/dst/architecture.md`
- `docs/dst/module-guide.md`
- `docs/dst/configuration.md`
- `docs/dst/recipes.md`
- `docs/dst/test-suite.md`
- `docs/dst/context/index.md`
- `docs/dst/agent/index.md`

## Notes for Future Split Consumers

This repo is intentionally self-contained for reuse by your other repos.

When another project should consume this package:

1. clone this repo as a sibling directory
2. install editable (`pip install -e /path/to/dst`)
3. import `dataset_tools.*` or call `dst` CLI directly
4. keep project-specific ad hoc logic outside this repo
