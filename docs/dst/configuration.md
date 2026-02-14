# Configuration

Runtime config is defined in `dataset_tools/config.py` and loaded by `load_config()`.

## Resolution Order

Config is merged in this exact order (later wins):

1. hardcoded defaults (`_default_config_dict()`)
2. local JSON file (if provided or discovered)
3. environment variables
4. runtime overrides (`--overrides '{...}'`)

This order is used consistently by all CLI commands that call `_load_app_config()`.

## Default Local Config Path Resolution

When `--config` is not provided, `load_config()` resolves the local config path in this order:

1. `DST_CONFIG_PATH`
2. package-local `local_config.json` (when present)
3. `$XDG_CONFIG_HOME/dst/local_config.json` (or `~/.config/dst/local_config.json`)

This behavior avoids hard dependency on repo/package layout and is suitable for installed package usage.

## Schema

Top-level `AppConfig` sections:

- `mount`
  - host/container path mapping used for Label Studio local-files URL conversion
- `label_studio`
  - URL, API key, default project/storage settings, upload strategy, and batching
- `dataset`
  - default dataset and field names used by workflows
- `disk_sync`
  - path replacement and backup naming controls for sync-to-disk

## Label Studio Critical Fields

These fields are operationally critical for correct media resolution in LS:

- `mount.host_root`
- `mount.ls_document_root`
- `mount.local_files_prefix`
- `label_studio.source_path`
- `label_studio.target_path`

If these are misconfigured, LS tasks may be created but media links will be broken.

## Environment Variables

Supported overrides:

- `DST_CONFIG_PATH`
- `LABEL_STUDIO_URL`
- `LABEL_STUDIO_API_KEY`
- `LABEL_STUDIO_PROJECT_TITLE`
- `LABEL_STUDIO_BATCH_SIZE`
- `LABEL_STUDIO_UPLOAD_STRATEGY`
- `FIFTYONE_DATASET_NAME`

## Secret Handling

- `require_label_studio_api_key(config)` enforces key presence before LS calls.
- `dst config show` masks API key by default.
- `dst config show --show-secrets` should be used only in trusted local terminals.

## Recommended Production Setup

1. create a local config from template:

```bash
cp local_config.example.json ~/.config/dst/local_config.json
```

2. keep secrets out of git (`local_config.json` should not be tracked).
3. for jobs/CI, prefer environment variables and small `--overrides` JSON for per-run adjustments.
4. verify resolved config before running destructive commands:

```bash
./dst config show
```

## Common Misconfigurations

- Missing API key -> LS commands fail fast.
- Wrong mount/prefix pairing -> LS tasks created with broken media URLs.
- Dataset field mismatch (`label_field`, `corrections_field`) -> pull/sync operations appear to run but write to unexpected fields.
