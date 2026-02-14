"""Central configuration schema and loader for ``dataset_tools``.

This module defines typed config dataclasses and a deterministic merge order:

1. in-code defaults
2. optional local JSON config file
3. environment variables
4. runtime overrides (for one-off CLI runs/jobs)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_LOCAL_CONFIG_FILENAME = "local_config.json"
DEFAULT_CONFIG_ENV_VAR = "DST_CONFIG_PATH"


@dataclass(frozen=True)
class PathMountConfig:
    """Host/container mount mapping used for Label Studio local-files URLs.

    ``host_root`` must be a parent of sample filepaths stored in FiftyOne.
    ``ls_document_root`` is the corresponding path inside the Label Studio pod.
    ``local_files_prefix`` is the URL prefix consumed by LS local-files storage.
    """
    host_root: str = "/home/user1000/shared_data"
    ls_document_root: str = "/data/images"
    local_files_prefix: str = "/data/local-files/?d="


@dataclass(frozen=True)
class LabelStudioConfig:
    """Runtime settings for Label Studio connectivity and task transfer.

    Includes API endpoint credentials, default project/storage metadata, and
    upload strategy controls such as batch size and backend mode.
    """
    url: str = "https://chlup-ls.dyn.cloud.e-infra.cz/"
    api_key: str = ""
    project_title: str = "FiftyOne Curation"
    source_path: str = "/data/images/frames_visitors_ndb"
    source_title: str = "Frames Visitors NDB"
    target_path: str = "/data/images/annotations_export"
    batch_size: int = 10
    clear_existing_tasks: bool = False
    upload_strategy: str = "sdk_batched"  # annotate_batched | sdk_batched


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-level defaults used by curation and sync workflows.

    Defines the default FiftyOne dataset name and canonical field names used
    for source labels and pulled LS corrections.
    """
    name: str = "visitors_dataset"
    label_field: str = "ground_truth"
    corrections_field: str = "ls_corrections"
    default_class_id: int = 0
    label_to_class_id: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DiskSyncConfig:
    """Filesystem sync defaults for writing corrections back to label files."""
    path_replacements: tuple[tuple[str, str], ...] = (
        ("/frames/", "/labels/"),
        ("/images/", "/labels/"),
    )
    backup_suffix_format: str = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class AppConfig:
    """Top-level immutable runtime configuration consumed by dataset_tools."""
    mount: PathMountConfig
    label_studio: LabelStudioConfig
    dataset: DatasetConfig
    disk_sync: DiskSyncConfig


def _default_config_dict() -> dict[str, Any]:
    """Return the baseline config dictionary used before any overrides."""
    return {
        "mount": {
            "host_root": "/home/user1000/shared_data",
            "ls_document_root": "/data/images",
            "local_files_prefix": "/data/local-files/?d=",
        },
        "label_studio": {
            "url": "https://chlup-ls.dyn.cloud.e-infra.cz/",
            "api_key": "",
            "project_title": "FiftyOne Curation",
            "source_path": "/data/images/frames_visitors_ndb",
            "source_title": "Frames Visitors NDB",
            "target_path": "/data/images/annotations_export",
            "batch_size": 10,
            "clear_existing_tasks": False,
            "upload_strategy": "sdk_batched",
        },
        "dataset": {
            "name": "visitors_dataset",
            "label_field": "ground_truth",
            "corrections_field": "ls_corrections",
            "default_class_id": 0,
            "label_to_class_id": {},
        },
        "disk_sync": {
            "path_replacements": [["/frames/", "/labels/"], ["/images/", "/labels/"]],
            "backup_suffix_format": "%Y%m%d_%H%M%S",
        },
    }


def resolve_default_local_config_path() -> Path:
    """Resolve default local config path in a portable, install-friendly way.

    Resolution order:
    1. ``DST_CONFIG_PATH`` environment variable
    2. package-local ``local_config.json`` when present
    3. ``$XDG_CONFIG_HOME/dst/local_config.json`` (or ``~/.config/dst/local_config.json``)
    """
    env_path = os.getenv(DEFAULT_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()

    package_local = Path(__file__).with_name(DEFAULT_LOCAL_CONFIG_FILENAME)
    if package_local.exists():
        return package_local

    xdg_config_home = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return xdg_config_home / "dst" / DEFAULT_LOCAL_CONFIG_FILENAME


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict.

    Nested dictionaries are merged key-by-key; non-dict values replace existing
    values at the same key.

    Args:
        base: Baseline dictionary.
        override: Override dictionary applied on top of ``base``.

    Returns:
        New merged dictionary.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_local_config(path: Path) -> dict[str, Any]:
    """Load optional local JSON config from ``path``.

    Args:
        path: Path to local config JSON.

    Returns:
        Parsed dictionary, or an empty dict when the file does not exist.
    """
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_env_config() -> dict[str, Any]:
    """Build config overrides from supported environment variables.

    Returns:
        Partial config dictionary containing only keys provided in env vars.
    """
    ls_url = os.getenv("LABEL_STUDIO_URL")
    ls_api_key = os.getenv("LABEL_STUDIO_API_KEY")
    ls_project = os.getenv("LABEL_STUDIO_PROJECT_TITLE")
    batch_size = os.getenv("LABEL_STUDIO_BATCH_SIZE")
    upload_strategy = os.getenv("LABEL_STUDIO_UPLOAD_STRATEGY")
    dataset_name = os.getenv("FIFTYONE_DATASET_NAME")

    env_cfg: dict[str, Any] = {}
    ls_cfg: dict[str, Any] = {}
    ds_cfg: dict[str, Any] = {}

    if ls_url:
        ls_cfg["url"] = ls_url
    if ls_api_key:
        ls_cfg["api_key"] = ls_api_key
    if ls_project:
        ls_cfg["project_title"] = ls_project
    if batch_size:
        ls_cfg["batch_size"] = int(batch_size)
    if upload_strategy:
        ls_cfg["upload_strategy"] = upload_strategy

    if dataset_name:
        ds_cfg["name"] = dataset_name

    if ls_cfg:
        env_cfg["label_studio"] = ls_cfg
    if ds_cfg:
        env_cfg["dataset"] = ds_cfg

    return env_cfg


def load_config(
    local_config_path: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Resolve and validate the runtime ``AppConfig``.

    Merge precedence is:
    defaults -> local config file -> environment variables -> ``overrides``.

    Args:
        local_config_path: Optional path to local JSON config. If omitted,
            defaults are resolved by ``resolve_default_local_config_path()``.
        overrides: Optional runtime override dictionary (typically from CLI).

    Returns:
        Immutable ``AppConfig`` instance.
    """
    path = Path(local_config_path).expanduser() if local_config_path else resolve_default_local_config_path()

    cfg = _default_config_dict()
    cfg = _deep_merge(cfg, _load_local_config(path))
    cfg = _deep_merge(cfg, _load_env_config())
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    replacements = tuple((src, dst) for src, dst in cfg["disk_sync"]["path_replacements"])

    return AppConfig(
        mount=PathMountConfig(**cfg["mount"]),
        label_studio=LabelStudioConfig(**cfg["label_studio"]),
        dataset=DatasetConfig(**cfg["dataset"]),
        disk_sync=DiskSyncConfig(
            path_replacements=replacements,
            backup_suffix_format=cfg["disk_sync"]["backup_suffix_format"],
        ),
    )


def require_label_studio_api_key(config: AppConfig) -> str:
    """Return Label Studio API key or raise a clear configuration error.

    Args:
        config: Resolved application config.

    Returns:
        Non-empty Label Studio API key string.

    Raises:
        RuntimeError: If the API key is missing from config/env/local file.
    """
    if not config.label_studio.api_key:
        raise RuntimeError(
            "Label Studio API key is not configured. Set LABEL_STUDIO_API_KEY or "
            "provide a local config file via --config (or DST_CONFIG_PATH)."
        )
    return config.label_studio.api_key
