# dataset_tools (`dst`)

Standalone repository scaffold for the `dataset_tools` package and unified `dst` CLI.

## Layout

- `src/dataset_tools/`: package source
- `tests/`: test suite
- `docs/dst/`: production documentation and context hubs
- `scripts/`: helper scripts (coverage/precode gate/docs generation)
- `examples/`: example workflow configs

## Install (editable)

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e .[dev]
python -m pip install -e .[anomaly]
```

## CLI

After install:

```bash
dst --help
```

Without install (repo-local wrapper):

```bash
./dst --help
```

## Config

Use a local config file outside git, for example:

```bash
mkdir -p ~/.config/dst
cp local_config.example.json ~/.config/dst/local_config.json
```

You can also point to any file via:

- env: `DST_CONFIG_PATH`
- CLI: `--config /path/to/local_config.json`

## Tests

```bash
python -m pytest -q tests
python -m pytest -q tests --cov=dataset_tools --cov-report=term-missing
```

See `docs/dst/test-suite.md` for detailed test documentation.
