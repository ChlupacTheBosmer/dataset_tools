# CLI Guide (`dst`)

This guide explains how to operate `dst` safely and predictably in real workflows.

Use `docs/dst/cli-reference.md` for exhaustive flag tables.

## CLI Contract

Entrypoints:

- `./dst ...`
- `python -m dataset_tools.dst ...`

Dispatch model:

- `build_parser()` defines all commands/subcommands
- each leaf command binds to one `cmd_*` handler
- handlers return JSON-serializable payloads (or `None` for blocking app session)

## Command Families

## `dst config`

Purpose: inspect merged runtime configuration and validate secret wiring.

Typical command:

```bash
./dst config show
```

## `dst data`

Purpose: import/export dataset structures.

- `data load yolo`: recursive YOLO import with configurable layout resolver
- `data load coco`: COCO import wrapper
- `data export ls-json`: generate LS import tasks from local layout

## `dst metrics`

Purpose: populate per-sample curation fields.

- embeddings, uniqueness, mistakenness, hardness, representativeness

## `dst brain`

Purpose: run FiftyOne Brain index/analysis jobs.

- visualization, similarity, duplicates (exact/near), leaky-splits

## `dst ls`

Purpose: Label Studio connectivity and project maintenance.

- `ls test`
- `ls project list|clear-tasks|cleanup`

## `dst workflow`

Purpose: multi-step pipelines.

- `workflow roundtrip`: send -> pull -> optional disk sync
- `workflow tags run|inline`: generic rule-driven tag workflow engine

## `dst sync`

Purpose: synchronize corrected FiftyOne fields to disk label files.

- `sync disk` with dry-run and path replacement controls

## `dst anomaly`

Purpose: anomaly detection workflows.

- `fit|score|run` for `embedding_distance`
- `train|score|run` using anomalib artifact backend

## `dst app`

Purpose: open FiftyOne app for a dataset.

- blocking mode keeps session alive in terminal
- `--no-block` returns immediately

## Error Model

Handlers raise `ValueError` / `RuntimeError` for user-recoverable issues. `main()` maps these into argparse-style CLI errors with readable messages.

## Logging and Noise Control

`workflow` commands support `--quiet-logs`, which captures noisy stdout/stderr from underlying libraries while preserving error context if a run fails.

## Output Contract

- default: prints JSON payload to stdout
- optional: `--output-json <path>` for workflow commands to persist payloads for automation systems

## Operational Safety

- Use dry-run options first on potentially destructive operations:
  - `ls project cleanup --dry-run`
  - `sync disk --dry-run`
- Prefer unique LS project titles per run to avoid task mixing.
- Validate LS connectivity before large workflow runs:

```bash
./dst ls test --list-projects
```

## Advanced Usage Pattern

Use config defaults for stable environment setup, and `--overrides`/command args for per-run variance. This keeps workflows reproducible while still flexible.
