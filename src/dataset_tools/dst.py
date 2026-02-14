"""Dataset Tools unified CLI entrypoint (`dst`).

This module defines argument parsing, command dispatch, and thin orchestration handlers that call
into the `dataset_tools` packages (loaders, metrics, brain, workflows, Label Studio sync, anomaly, and models).
Each `cmd_*` function returns structured payloads so execution can be inspected, tested, and optionally
written to JSON by automation jobs.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


def _parse_json_value(raw: str, label: str) -> Any:
    """Parse raw JSON text for a named CLI argument and raise readable errors.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for {label}: {e.msg}") from e


def _parse_json_dict(raw: str | None, label: str) -> dict[str, Any]:
    """Parse an optional JSON object argument and normalize missing values to empty dicts.
    """
    if raw is None:
        return {}

    payload = _parse_json_value(raw, label)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object for {label}")

    return payload


def _parse_json_list(raw: str, label: str) -> list[Any]:
    """Parse and validate a JSON array argument from the CLI.
    """
    payload = _parse_json_value(raw, label)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array for {label}")
    return payload


def _parse_csv_list(raw: str, label: str) -> list[str]:
    """Parse a comma-separated CLI value into a non-empty list of tokens.
    """
    items = [part.strip() for part in raw.split(",")]
    out = [item for item in items if item]
    if not out:
        raise ValueError(f"Expected a comma-separated list for {label}")
    return out


def _parse_class_map(raw: str | None) -> dict[int, str]:
    """Convert class-map JSON into an integer-keyed mapping for YOLO loaders.
    """
    if raw is None:
        return {}

    payload = _parse_json_dict(raw, "--class-map")
    out: dict[int, str] = {}
    for key, value in payload.items():
        out[int(key)] = str(value)

    return out


def _parse_path_replacements(entries: Iterable[str] | None) -> tuple[tuple[str, str], ...] | None:
    """Parse repeated SRC=DST replacement rules used by disk sync routines.
    """
    if not entries:
        return None

    pairs = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --path-replacement '{entry}', expected SRC=DST")

        src, dst = entry.split("=", 1)
        if not src:
            raise ValueError(f"Invalid --path-replacement '{entry}', source is empty")

        pairs.append((src, dst))

    return tuple(pairs)


def _parse_optional_text(raw: str | None) -> str | None:
    """Normalize empty/None-like strings into Python None for optional fields.
    """
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if value.lower() == "none":
        return None
    return value


def _parse_devices(raw: str | int | None) -> str | int | None:
    """Normalize anomaly `--devices` input to an int, string selector, or None.
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    value = raw.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    return value


def _load_app_config(args) -> Any:
    """Load the resolved application configuration from defaults, local config, and overrides.
    """
    from dataset_tools.config import load_config

    overrides = _parse_json_dict(args.overrides, "--overrides") if hasattr(args, "overrides") else None
    local_config_path = getattr(args, "config", None)
    return load_config(local_config_path=local_config_path, overrides=overrides)


def _list_projects(ls) -> list[Any]:
    """List Label Studio projects while supporting API version differences.
    """
    if hasattr(ls, "list_projects"):
        return list(ls.list_projects())
    if hasattr(ls, "get_projects"):
        return list(ls.get_projects())
    raise RuntimeError("Label Studio client does not support listing projects")


def _project_payload(project, include_task_count: bool = False) -> dict[str, Any]:
    """Convert a Label Studio project object into a stable dictionary payload.
    """
    payload: dict[str, Any] = {
        "id": getattr(project, "id", None),
        "title": getattr(project, "title", None),
    }

    if include_task_count:
        try:
            payload["task_count"] = len(project.get_tasks())
        except Exception:
            payload["task_count"] = None

    return payload


def _mask_api_key(value: str) -> str:
    """Mask secret tokens before printing config payloads.
    """
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{'*' * (len(value) - 4)}{value[-4:]}"


def _print_result(result: Any):
    """Render command output payloads to stdout in a consistent format.
    """
    if result is None:
        return

    if isinstance(result, (dict, list)):
        print(json.dumps(result, indent=2))
        return

    print(result)


def _write_json_output(path: str | None, payload: Any):
    """Optionally persist command results to a JSON file path.
    """
    if not path:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _execute_with_optional_log_capture(fn, quiet_logs: bool):
    """Execute a callable while optionally suppressing noisy library logs.

When `quiet_logs=True`, stdout/stderr and selected library loggers are captured to keep CLI output clean. If execution fails, captured logs are replayed to stderr for debugging context before the exception is re-raised.
    """
    if not quiet_logs:
        return fn()

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    fo = None
    old_progress = None
    loggers = [logging.getLogger("fiftyone"), logging.getLogger("eta")]
    old_levels = [logger.level for logger in loggers]

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                import fiftyone as fo  # type: ignore

                old_progress = fo.config.show_progress_bars
                fo.config.show_progress_bars = False
            except Exception:
                fo = None
                old_progress = None

            for logger in loggers:
                logger.setLevel(logging.ERROR)

            return fn()
    except Exception:
        captured_out = stdout_buffer.getvalue()
        captured_err = stderr_buffer.getvalue()
        if captured_out:
            print(captured_out, file=sys.stderr, end="" if captured_out.endswith("\n") else "\n")
        if captured_err:
            print(captured_err, file=sys.stderr, end="" if captured_err.endswith("\n") else "\n")
        raise
    finally:
        if fo is not None and old_progress is not None:
            fo.config.show_progress_bars = old_progress
        for logger, level in zip(loggers, old_levels):
            logger.setLevel(level)


def cmd_config_show(args):
    """Run the `dst config show` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `config show` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    cfg = _load_app_config(args)
    payload = asdict(cfg)

    if not args.show_secrets:
        payload["label_studio"]["api_key"] = _mask_api_key(payload["label_studio"].get("api_key", ""))

    return payload


def cmd_ls_test(args):
    """Run the `dst ls test` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `ls test` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.label_studio.client import ensure_label_studio_client

    cfg = _load_app_config(args)
    ls = ensure_label_studio_client(cfg)
    projects = _list_projects(ls)

    payload: dict[str, Any] = {
        "url": cfg.label_studio.url,
        "connected": True,
        "project_count": len(projects),
    }

    if args.list_projects:
        payload["projects"] = [_project_payload(p) for p in projects]

    return payload


def cmd_ls_project_list(args):
    """Run the `dst ls project list` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `ls project list` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.label_studio.client import ensure_label_studio_client

    cfg = _load_app_config(args)
    ls = ensure_label_studio_client(cfg)

    projects = _list_projects(ls)
    if args.contains:
        needle = args.contains if args.case_sensitive else args.contains.lower()

        def _match(project_title: str | None) -> bool:
            if project_title is None:
                return False
            haystack = project_title if args.case_sensitive else project_title.lower()
            return needle in haystack

        projects = [p for p in projects if _match(getattr(p, "title", None))]

    if args.limit is not None:
        projects = projects[: args.limit]

    return {
        "count": len(projects),
        "projects": [_project_payload(p, include_task_count=args.with_task_count) for p in projects],
    }


def _resolve_project(ls, project_id: int | None, project_title: str | None):
    """Resolve a Label Studio project by id or exact title.
    """
    if project_id is not None:
        return ls.get_project(project_id)

    if not project_title:
        raise ValueError("Either --id or --title must be provided")

    matches = [p for p in _list_projects(ls) if getattr(p, "title", None) == project_title]
    if not matches:
        raise RuntimeError(f"Project '{project_title}' not found")

    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple projects named '{project_title}'. Use --id for disambiguation"
        )

    return matches[0]


def cmd_ls_project_clear_tasks(args):
    """Run the `dst ls project clear tasks` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `ls project clear tasks` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.label_studio.client import ensure_label_studio_client

    cfg = _load_app_config(args)
    ls = ensure_label_studio_client(cfg)
    project = _resolve_project(ls, args.id, args.title)

    task_count = len(project.get_tasks())
    payload = {
        "project": _project_payload(project, include_task_count=False),
        "task_count_before": task_count,
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        project.delete_all_tasks()
        payload["cleared"] = task_count

    return payload


def cmd_ls_project_cleanup(args):
    """Run the `dst ls project cleanup` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `ls project cleanup` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.label_studio.client import ensure_label_studio_client

    cfg = _load_app_config(args)
    ls = ensure_label_studio_client(cfg)

    keywords = args.keyword
    projects = _list_projects(ls)

    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for project in projects:
        title = getattr(project, "title", None) or ""
        haystack = title if args.case_sensitive else title.lower()
        needles = keywords if args.case_sensitive else [k.lower() for k in keywords]
        matched = any(keyword in haystack for keyword in needles)

        if matched:
            entry = _project_payload(project)
            deleted.append(entry)
            if not args.dry_run:
                ls.delete_project(project.id)
        else:
            skipped.append(_project_payload(project))

    return {
        "dry_run": bool(args.dry_run),
        "matched_count": len(deleted),
        "deleted": deleted,
        "skipped_count": len(skipped),
    }


def cmd_data_load_yolo(args):
    """Run the `dst data load yolo` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `data load yolo` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.loaders import (
        ImagesLabelsSubdirResolver,
        MirroredRootsPathResolver,
        YoloDatasetLoader,
        YoloParserConfig,
    )

    cfg = _load_app_config(args)
    dataset_name = args.dataset or cfg.dataset.name
    class_map = _parse_class_map(args.class_map)

    if args.root:
        resolver = ImagesLabelsSubdirResolver(
            root_dir=Path(args.root),
            images_subdir=args.images_subdir,
            labels_subdir=args.labels_subdir,
        )
    else:
        if not (args.images_root and args.labels_root):
            raise ValueError("Provide either --root or both --images-root and --labels-root")

        resolver = MirroredRootsPathResolver(
            images_root=Path(args.images_root),
            labels_root=Path(args.labels_root),
        )

    loader = YoloDatasetLoader(
        resolver=resolver,
        parser_config=YoloParserConfig(
            class_id_to_label=class_map,
            include_confidence=not args.no_confidence,
        ),
    )
    result = loader.load(
        dataset_name=dataset_name,
        overwrite=args.overwrite,
        persistent=args.persistent,
    )

    return {
        "dataset": result.dataset_name,
        "samples": result.sample_count,
        "persistent": bool(args.persistent),
    }


def cmd_data_load_coco(args):
    """Run the `dst data load coco` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `data load coco` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.loaders import CocoDatasetLoader, CocoLoaderConfig

    cfg = _load_app_config(args)
    dataset_name = args.dataset or cfg.dataset.name

    loader = CocoDatasetLoader(
        CocoLoaderConfig(
            dataset_dir=Path(args.dataset_dir),
            data_path=args.data_path,
            labels_path=args.labels_path,
        )
    )
    result = loader.load(
        dataset_name=dataset_name,
        overwrite=args.overwrite,
        persistent=args.persistent,
    )

    return {
        "dataset": result.dataset_name,
        "samples": result.sample_count,
        "persistent": bool(args.persistent),
    }


def cmd_data_export_ls_json(args):
    """Run the `dst data export ls json` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `data export ls json` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.label_studio_json import build_tasks

    root_dir = Path(args.root)
    tasks = build_tasks(root_dir=root_dir, ls_root=args.ls_root)

    output_path = Path(args.output) if args.output else root_dir / "import_tasks.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    return {
        "tasks": len(tasks),
        "output": str(output_path),
    }


def cmd_metrics_embeddings(args):
    """Run the `dst metrics embeddings` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `metrics embeddings` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.metrics import EmbeddingsComputation

    job = EmbeddingsComputation(
        dataset_name=args.dataset,
        model_name=args.model,
        model_ref=args.model_ref,
        embeddings_field=args.embeddings_field,
        patches_field=args.patches_field,
        use_umap=not args.no_umap,
        use_cluster=not args.no_cluster,
        n_clusters=args.n_clusters,
    )
    return job.run()


def cmd_metrics_uniqueness(args):
    """Run the `dst metrics uniqueness` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `metrics uniqueness` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.metrics import UniquenessComputation

    job = UniquenessComputation(
        dataset_name=args.dataset,
        embeddings_field=args.embeddings_field,
        output_field=args.output_field,
    )
    return job.run()


def cmd_metrics_mistakenness(args):
    """Run the `dst metrics mistakenness` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `metrics mistakenness` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.metrics import MistakennessComputation

    job = MistakennessComputation(
        dataset_name=args.dataset,
        pred_field=args.pred_field,
        gt_field=args.gt_field,
        mistakenness_field=args.mistakenness_field,
        missing_field=args.missing_field,
        spurious_field=args.spurious_field,
    )
    return job.run()


def cmd_metrics_hardness(args):
    """Run the `dst metrics hardness` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `metrics hardness` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.metrics import HardnessComputation

    job = HardnessComputation(
        dataset_name=args.dataset,
        label_field=args.label_field,
        output_field=args.output_field,
    )
    return job.run()


def cmd_metrics_representativeness(args):
    """Run the `dst metrics representativeness` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `metrics representativeness` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.metrics import RepresentativenessComputation

    job = RepresentativenessComputation(
        dataset_name=args.dataset,
        output_field=args.output_field,
        method=args.method,
        embeddings_field=args.embeddings_field,
        roi_field=args.roi_field,
    )
    return job.run()


def cmd_brain_visualization(args):
    """Run the `dst brain visualization` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `brain visualization` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.brain import VisualizationOperation

    job = VisualizationOperation(
        dataset_name=args.dataset,
        method=args.method,
        num_dims=args.num_dims,
        embeddings=args.embeddings_field,
        patches_field=args.patches_field,
        brain_key=args.brain_key,
    )
    return job.run()


def cmd_brain_similarity(args):
    """Run the `dst brain similarity` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `brain similarity` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.brain import SimilarityOperation

    job = SimilarityOperation(
        dataset_name=args.dataset,
        embeddings=args.embeddings_field,
        patches_field=args.patches_field,
        roi_field=args.roi_field,
        backend=args.backend,
        brain_key=args.brain_key,
    )
    return job.run()


def cmd_brain_duplicates_exact(args):
    """Run the `dst brain duplicates exact` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `brain duplicates exact` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.brain import ExactDuplicatesOperation

    job = ExactDuplicatesOperation(dataset_name=args.dataset)
    return job.run()


def cmd_brain_duplicates_near(args):
    """Run the `dst brain duplicates near` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `brain duplicates near` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.brain import NearDuplicatesOperation

    job = NearDuplicatesOperation(
        dataset_name=args.dataset,
        threshold=args.threshold,
        embeddings=args.embeddings_field,
        roi_field=args.roi_field,
    )
    return job.run()


def cmd_brain_leaky_splits(args):
    """Run the `dst brain leaky splits` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `brain leaky splits` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.brain import LeakySplitsOperation

    splits = _parse_csv_list(args.splits, "--splits")

    job = LeakySplitsOperation(
        dataset_name=args.dataset,
        splits=splits,
        threshold=args.threshold,
        embeddings=args.embeddings_field,
        roi_field=args.roi_field,
    )
    return job.run()


def cmd_models_list(args):
    """Run the `dst models list` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `models list` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.models import list_providers, provider_model_list

    if args.provider is None:
        providers = list_providers()
        return {
            "providers": providers,
        }

    models = provider_model_list(
        args.provider,
        contains=args.contains,
        limit=args.limit,
    )
    return {
        "provider": args.provider,
        "count": len(models),
        "models": models,
    }


def cmd_models_resolve(args):
    """Run the `dst models resolve` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `models resolve` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.models import load_model

    loaded = load_model(
        args.model_ref,
        default_provider=args.default_provider,
        task=args.task,
        capability=args.capability,
    )
    return {
        "model_ref": f"{loaded.ref.provider}:{loaded.ref.model_id}",
        "provider": loaded.ref.provider,
        "model_id": loaded.ref.model_id,
        "model_type": type(loaded.model).__name__,
        "capabilities": list(loaded.capabilities),
        "metadata": dict(loaded.metadata),
    }


def cmd_models_validate(args):
    """Run the `dst models validate` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `models validate` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    payload = cmd_models_resolve(args)
    payload["valid"] = True
    return payload


def cmd_anomaly_fit(args):
    """Run the `dst anomaly fit` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `anomaly fit` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.anomaly import fit_embedding_distance_reference, save_reference

    if args.backend != "embedding_distance":
        raise ValueError("Only backend='embedding_distance' supports standalone fit")

    reference = fit_embedding_distance_reference(
        dataset_name=args.dataset,
        embeddings_field=args.embeddings_field,
        normal_tag=args.normal_tag,
        threshold=args.threshold,
        threshold_quantile=args.threshold_quantile,
    )
    if args.reference_json:
        save_reference(args.reference_json, reference)

    payload = {
        "dataset": args.dataset,
        "reference": reference.to_dict(),
    }
    if args.reference_json:
        payload["reference_json"] = args.reference_json
    return payload


def cmd_anomaly_train(args):
    """Run the `dst anomaly train` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `anomaly train` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.anomaly import train_and_export_anomalib

    artifact = train_and_export_anomalib(
        dataset_name=args.dataset,
        model_ref=args.model_ref,
        normal_tag=args.normal_tag,
        abnormal_tag=args.abnormal_tag,
        mask_field=args.mask_field,
        artifact_dir=args.artifact_dir,
        data_dir=args.data_dir,
        artifact_format=args.artifact_format,
        image_size=args.image_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        normal_split_ratio=args.normal_split_ratio,
        test_split_mode=args.test_split_mode,
        test_split_ratio=args.test_split_ratio,
        val_split_mode=args.val_split_mode,
        val_split_ratio=args.val_split_ratio,
        seed=args.seed,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        symlink=not bool(args.copy_media),
        overwrite_data=bool(args.overwrite_data),
        artifact_json=args.artifact_json,
    )

    manifest_path = (
        Path(args.artifact_json).expanduser().resolve()
        if args.artifact_json
        else Path(artifact.artifact_dir) / "anomalib_artifact.json"
    )
    return {
        "backend": "anomalib",
        "dataset": args.dataset,
        "action": "train_export",
        "artifact_json": str(manifest_path),
        "artifact": artifact.to_dict(),
    }


def cmd_anomaly_score(args):
    """Run the `dst anomaly score` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `anomaly score` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.anomaly import (
        fit_embedding_distance_reference,
        load_reference,
        score_with_anomalib,
        score_with_embedding_distance,
    )

    if args.backend == "anomalib":
        if not args.artifact:
            raise ValueError(
                "backend='anomalib' requires --artifact (path to anomalib artifact JSON "
                "or exported model file). Run `dst anomaly train` first."
            )
        return score_with_anomalib(
            dataset_name=args.dataset,
            artifact_path=args.artifact,
            artifact_format=args.artifact_format,
            threshold=args.anomaly_threshold,
            score_field=args.score_field,
            flag_field=args.flag_field,
            label_field=_parse_optional_text(args.label_field),
            map_field=_parse_optional_text(args.map_field),
            mask_field=_parse_optional_text(args.mask_field),
            tag_filter=args.tag,
            device=args.device,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        )

    if args.backend != "embedding_distance":
        raise ValueError("Unsupported backend. Use 'embedding_distance' or 'anomalib'")

    if args.reference_json:
        reference = load_reference(args.reference_json)
    else:
        reference = fit_embedding_distance_reference(
            dataset_name=args.dataset,
            embeddings_field=args.embeddings_field,
            normal_tag=args.normal_tag,
            threshold=args.threshold,
            threshold_quantile=args.threshold_quantile,
        )

    payload = score_with_embedding_distance(
        dataset_name=args.dataset,
        reference=reference,
        score_field=args.score_field,
        flag_field=args.flag_field,
        tag_filter=args.tag,
    )
    payload["reference"] = reference.to_dict()
    if args.reference_json:
        payload["reference_json"] = args.reference_json
    return payload


def cmd_anomaly_run(args):
    """Run the `dst anomaly run` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `anomaly run` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.anomaly import run_embedding_distance, score_with_anomalib

    if args.backend == "anomalib":
        if not args.artifact:
            raise ValueError(
                "backend='anomalib' requires --artifact. "
                "Use `dst anomaly train` to produce an artifact first."
            )
        return score_with_anomalib(
            dataset_name=args.dataset,
            artifact_path=args.artifact,
            artifact_format=args.artifact_format,
            threshold=args.anomaly_threshold,
            score_field=args.score_field,
            flag_field=args.flag_field,
            label_field=_parse_optional_text(args.label_field),
            map_field=_parse_optional_text(args.map_field),
            mask_field=_parse_optional_text(args.mask_field),
            tag_filter=args.tag,
            device=args.device,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        )

    if args.backend != "embedding_distance":
        raise ValueError("Unsupported backend. Use 'embedding_distance' or 'anomalib'")

    return run_embedding_distance(
        dataset_name=args.dataset,
        embeddings_field=args.embeddings_field,
        normal_tag=args.normal_tag,
        score_tag=args.tag,
        score_field=args.score_field,
        flag_field=args.flag_field,
        threshold=args.threshold,
        threshold_quantile=args.threshold_quantile,
        reference_path=args.reference_json,
    )


def cmd_workflow_roundtrip(args):
    """Run the `dst workflow roundtrip` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `workflow roundtrip` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    app_config = _load_app_config(args)

    send_params = _parse_json_dict(args.send_params, "--send-params")
    pull_params = _parse_json_dict(args.pull_params, "--pull-params")
    sync_params = _parse_json_dict(args.sync_params, "--sync-params")

    send_params["strict_preflight"] = bool(args.strict_preflight)
    send_params["overwrite_annotation_run"] = bool(args.overwrite_annotation_run)

    if args.launch_editor:
        send_params["launch_editor"] = True

    if args.annotation_key:
        send_params["annotation_key"] = args.annotation_key
        pull_params["annotation_key"] = args.annotation_key

    if args.pull_strategy:
        pull_params["pull_strategy"] = args.pull_strategy

    if args.create_if_missing:
        pull_params["create_if_missing"] = True

    def _run_workflow():
        from dataset_tools.workflows import CurationRoundtripWorkflow, RoundtripWorkflowConfig

        workflow = CurationRoundtripWorkflow(app_config=app_config)
        workflow_config = RoundtripWorkflowConfig(
            dataset_name=args.dataset,
            send_tag=args.tag,
            project_title=args.project,
            label_field=args.label_field,
            corrections_field=args.corrections_field,
            send_to_label_studio=not args.skip_send,
            pull_from_label_studio=not args.skip_pull,
            sync_to_disk=not args.skip_sync_disk,
            dry_run_sync=args.dry_run_sync,
            clear_project_tasks=args.clear_project_tasks,
            upload_strategy=args.upload_strategy,
            additional_send_params=send_params,
            additional_pull_params=pull_params,
            additional_sync_params=sync_params,
        )
        return workflow.run(workflow_config)

    result = _execute_with_optional_log_capture(
        _run_workflow,
        quiet_logs=bool(getattr(args, "quiet_logs", False)),
    )
    _write_json_output(getattr(args, "output_json", None), result)
    return result


def _build_tag_rules(payload_rules: list[dict[str, Any]]):
    """Translate JSON workflow rules into typed tag-workflow rule objects.
    """
    from dataset_tools.tag_workflow import TagOperationRule

    return [
        TagOperationRule(
            operation=rule["operation"],
            tag=rule.get("tag"),
            params=rule.get("params", {}),
        )
        for rule in payload_rules
    ]


def cmd_workflow_tags_run(args):
    """Run the `dst workflow tags run` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `workflow tags run` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    app_config = _load_app_config(args)

    with Path(args.workflow).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if not args.dataset:
            raise ValueError("When workflow file is a JSON array of rules, --dataset is required")
        dataset_name = args.dataset
        rules_payload = payload
        fail_fast = True if args.fail_fast is None else args.fail_fast
    elif isinstance(payload, dict):
        dataset_name = args.dataset or payload.get("dataset_name")
        if not dataset_name:
            raise ValueError("Workflow JSON must include dataset_name or provide --dataset")
        rules_payload = payload.get("rules", [])
        if not isinstance(rules_payload, list):
            raise ValueError("Workflow JSON key 'rules' must be a list")

        if args.fail_fast is None:
            fail_fast = bool(payload.get("fail_fast", True))
        else:
            fail_fast = bool(args.fail_fast)
    else:
        raise ValueError("Workflow JSON must be either an object or an array")

    def _run_workflow():
        from dataset_tools.tag_workflow import TagWorkflowConfig, TagWorkflowEngine

        config = TagWorkflowConfig(
            dataset_name=dataset_name,
            rules=_build_tag_rules(rules_payload),
            fail_fast=fail_fast,
        )
        engine = TagWorkflowEngine(app_config=app_config)
        return engine.run(config)

    result = _execute_with_optional_log_capture(
        _run_workflow,
        quiet_logs=bool(getattr(args, "quiet_logs", False)),
    )
    _write_json_output(getattr(args, "output_json", None), result)
    return result


def cmd_workflow_tags_inline(args):
    """Run the `dst workflow tags inline` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `workflow tags inline` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    app_config = _load_app_config(args)
    rules_payload = [_parse_json_dict(item, "--rule") for item in args.rule]

    def _run_workflow():
        from dataset_tools.tag_workflow import TagWorkflowConfig, TagWorkflowEngine

        config = TagWorkflowConfig(
            dataset_name=args.dataset,
            rules=_build_tag_rules(rules_payload),
            fail_fast=not args.no_fail_fast,
        )
        engine = TagWorkflowEngine(app_config=app_config)
        return engine.run(config)

    result = _execute_with_optional_log_capture(
        _run_workflow,
        quiet_logs=bool(getattr(args, "quiet_logs", False)),
    )
    _write_json_output(getattr(args, "output_json", None), result)
    return result


def cmd_sync_disk(args):
    """Run the `dst sync disk` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `sync disk` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    from dataset_tools.sync_from_fo_to_disk import sync_corrections_to_disk

    cfg = _load_app_config(args)
    label_map = _parse_json_dict(args.label_to_class_id, "--label-to-class-id") if args.label_to_class_id else None
    replacements = _parse_path_replacements(args.path_replacement)

    dataset_name = args.dataset or cfg.dataset.name

    synced = sync_corrections_to_disk(
        dataset_name=dataset_name,
        dry_run=args.dry_run,
        tag_filter=args.tag,
        corrections_field=args.corrections_field,
        label_to_class_id=label_map,
        default_class_id=args.default_class_id,
        path_replacements=replacements,
        backup_suffix_format=args.backup_suffix_format,
    )

    return {
        "dataset": dataset_name,
        "synced_files": synced,
        "dry_run": bool(args.dry_run),
    }


def cmd_app_open(args):
    """Run the `dst app open` command handler and return a JSON-serializable result.

This handler is invoked by argparse for the `app open` command path. It performs argument normalization, calls into the corresponding dataset_tools module, and returns a payload that can be printed or persisted with `--output-json`.

Expected setup:
    - Python environment has FiftyOne and optional provider dependencies installed.
    - External services (for LS commands) are reachable and credentials are configured.
    - Dataset names/fields referenced by CLI arguments exist when required.

Raises:
    ValueError: If argument combinations are invalid for the selected backend.
    RuntimeError: If required datasets/projects/services cannot be resolved.
    """
    import fiftyone as fo  # type: ignore

    if args.dataset not in fo.list_datasets():
        raise RuntimeError(f"Dataset '{args.dataset}' not found")

    dataset = fo.load_dataset(args.dataset)
    session = fo.launch_app(dataset, port=args.port, address=args.address)

    if args.no_block:
        return {
            "dataset": args.dataset,
            "address": args.address,
            "port": args.port,
            "session_open": True,
            "no_block": True,
        }

    print(f"FiftyOne app launched for '{args.dataset}' on {args.address}:{args.port}")
    print("Press Ctrl+C to close")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        session.close()

    return None


def _add_common_config_args(parser: argparse.ArgumentParser):
    """Attach shared `--config` and `--overrides` arguments to a parser.
    """
    parser.add_argument(
        "--config",
        default=None,
        help="Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json)",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="JSON object with runtime config overrides",
    )


def _add_persistent_args(parser: argparse.ArgumentParser):
    """Attach mutually-exclusive persistence flags to loader commands.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--persistent", dest="persistent", action="store_true", default=True)
    group.add_argument("--non-persistent", dest="persistent", action="store_false")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the full `dst` argparse command tree.

The parser intentionally maps each subcommand to a dedicated `cmd_*` handler via `set_defaults(func=...)` so behavior is explicit and testable. Use this function as the canonical source of CLI contract.
    """
    parser = argparse.ArgumentParser(
        prog="dst",
        description="Dataset Tools CLI",
    )
    top = parser.add_subparsers(dest="command", required=True)

    # config
    config_parser = top.add_parser("config", help="Inspect and validate dataset_tools configuration")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_show = config_sub.add_parser("show", help="Print resolved configuration")
    _add_common_config_args(config_show)
    config_show.add_argument("--show-secrets", action="store_true", help="Do not mask sensitive fields")
    config_show.set_defaults(func=cmd_config_show)

    # ls
    ls_parser = top.add_parser("ls", help="Label Studio operations")
    ls_sub = ls_parser.add_subparsers(dest="ls_command", required=True)

    ls_test = ls_sub.add_parser("test", help="Test Label Studio connection")
    _add_common_config_args(ls_test)
    ls_test.add_argument("--list-projects", action="store_true", help="Include project list in output")
    ls_test.set_defaults(func=cmd_ls_test)

    ls_project = ls_sub.add_parser("project", help="Project-level Label Studio operations")
    ls_project_sub = ls_project.add_subparsers(dest="ls_project_command", required=True)

    ls_project_list = ls_project_sub.add_parser("list", help="List projects")
    _add_common_config_args(ls_project_list)
    ls_project_list.add_argument("--contains", default=None, help="Filter title by substring")
    ls_project_list.add_argument("--case-sensitive", action="store_true")
    ls_project_list.add_argument("--limit", type=int, default=None)
    ls_project_list.add_argument("--with-task-count", action="store_true")
    ls_project_list.set_defaults(func=cmd_ls_project_list)

    ls_project_clear = ls_project_sub.add_parser("clear-tasks", help="Delete all tasks in a project")
    _add_common_config_args(ls_project_clear)
    ls_project_clear.add_argument("--id", type=int, default=None, help="Project ID")
    ls_project_clear.add_argument("--title", default=None, help="Project title (exact match)")
    ls_project_clear.add_argument("--dry-run", action="store_true")
    ls_project_clear.set_defaults(func=cmd_ls_project_clear_tasks)

    ls_project_cleanup = ls_project_sub.add_parser("cleanup", help="Delete projects by keyword")
    _add_common_config_args(ls_project_cleanup)
    ls_project_cleanup.add_argument("--keyword", action="append", required=True, help="Keyword to match")
    ls_project_cleanup.add_argument("--dry-run", action="store_true")
    ls_project_cleanup.add_argument("--case-sensitive", action="store_true")
    ls_project_cleanup.set_defaults(func=cmd_ls_project_cleanup)

    # data
    data_parser = top.add_parser("data", help="Dataset import/export operations")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    data_load = data_sub.add_parser("load", help="Load datasets into FiftyOne")
    data_load_sub = data_load.add_subparsers(dest="load_command", required=True)

    data_load_yolo = data_load_sub.add_parser("yolo", help="Load a YOLO dataset")
    _add_common_config_args(data_load_yolo)
    data_load_yolo.add_argument("--dataset", default=None, help="Target FiftyOne dataset name")
    data_load_yolo.add_argument("--root", default=None, help="Root containing images/ and labels/")
    data_load_yolo.add_argument("--images-root", default=None, help="Images root for mirrored layout")
    data_load_yolo.add_argument("--labels-root", default=None, help="Labels root for mirrored layout")
    data_load_yolo.add_argument("--images-subdir", default="images")
    data_load_yolo.add_argument("--labels-subdir", default="labels")
    data_load_yolo.add_argument("--class-map", default=None, help='JSON map like {"0":"rodent"}')
    data_load_yolo.add_argument("--no-confidence", action="store_true", help="Ignore 6th YOLO confidence column")
    data_load_yolo.add_argument("--overwrite", action="store_true")
    _add_persistent_args(data_load_yolo)
    data_load_yolo.set_defaults(func=cmd_data_load_yolo)

    data_load_coco = data_load_sub.add_parser("coco", help="Load a COCO detection dataset")
    _add_common_config_args(data_load_coco)
    data_load_coco.add_argument("--dataset", default=None, help="Target FiftyOne dataset name")
    data_load_coco.add_argument("--dataset-dir", required=True)
    data_load_coco.add_argument("--data-path", default="data")
    data_load_coco.add_argument("--labels-path", default="labels.json")
    data_load_coco.add_argument("--overwrite", action="store_true")
    _add_persistent_args(data_load_coco)
    data_load_coco.set_defaults(func=cmd_data_load_coco)

    data_export = data_sub.add_parser("export", help="Export helper formats")
    data_export_sub = data_export.add_subparsers(dest="export_command", required=True)

    data_export_ls_json = data_export_sub.add_parser("ls-json", help="Generate Label Studio import JSON from YOLO data")
    data_export_ls_json.add_argument("--root", required=True, help="Root containing images/ and labels/")
    data_export_ls_json.add_argument(
        "--ls-root",
        default="/data/local-files/?d=frames_visitors_ndb/images",
        help="Prefix used in generated task image URLs",
    )
    data_export_ls_json.add_argument("--output", default=None, help="Output JSON path")
    data_export_ls_json.set_defaults(func=cmd_data_export_ls_json)

    # metrics
    metrics_parser = top.add_parser("metrics", help="Populate dataset-level metric fields")
    metrics_sub = metrics_parser.add_subparsers(dest="metrics_command", required=True)

    metrics_embeddings = metrics_sub.add_parser("embeddings", help="Compute embeddings (+ optional UMAP/clusters)")
    metrics_embeddings.add_argument("--dataset", required=True)
    metrics_embeddings.add_argument(
        "--model",
        default="facebook/dinov2-base",
        help="Legacy alias for HF model id when --model-ref is not provided",
    )
    metrics_embeddings.add_argument(
        "--model-ref",
        default=None,
        help="Provider-qualified model reference (e.g. hf:facebook/dinov2-base, foz:clip-vit-base32-torch)",
    )
    metrics_embeddings.add_argument("--embeddings-field", default="embeddings")
    metrics_embeddings.add_argument("--patches-field", default=None)
    metrics_embeddings.add_argument("--no-umap", action="store_true")
    metrics_embeddings.add_argument("--no-cluster", action="store_true")
    metrics_embeddings.add_argument("--n-clusters", type=int, default=10)
    metrics_embeddings.set_defaults(func=cmd_metrics_embeddings)

    metrics_uniqueness = metrics_sub.add_parser("uniqueness", help="Compute uniqueness scores")
    metrics_uniqueness.add_argument("--dataset", required=True)
    metrics_uniqueness.add_argument("--embeddings-field", default=None)
    metrics_uniqueness.add_argument("--output-field", default="uniqueness")
    metrics_uniqueness.set_defaults(func=cmd_metrics_uniqueness)

    metrics_mistakenness = metrics_sub.add_parser("mistakenness", help="Compute mistakenness fields")
    metrics_mistakenness.add_argument("--dataset", required=True)
    metrics_mistakenness.add_argument("--pred-field", default="predictions")
    metrics_mistakenness.add_argument("--gt-field", default="ground_truth")
    metrics_mistakenness.add_argument("--mistakenness-field", default="mistakenness")
    metrics_mistakenness.add_argument("--missing-field", default="possible_missing")
    metrics_mistakenness.add_argument("--spurious-field", default="possible_spurious")
    metrics_mistakenness.set_defaults(func=cmd_metrics_mistakenness)

    metrics_hardness = metrics_sub.add_parser("hardness", help="Compute sample hardness scores")
    metrics_hardness.add_argument("--dataset", required=True)
    metrics_hardness.add_argument("--label-field", default="ground_truth")
    metrics_hardness.add_argument("--output-field", default="hardness")
    metrics_hardness.set_defaults(func=cmd_metrics_hardness)

    metrics_representativeness = metrics_sub.add_parser(
        "representativeness",
        help="Compute sample representativeness scores",
    )
    metrics_representativeness.add_argument("--dataset", required=True)
    metrics_representativeness.add_argument("--output-field", default="representativeness")
    metrics_representativeness.add_argument("--method", default="cluster-center")
    metrics_representativeness.add_argument("--embeddings-field", default=None)
    metrics_representativeness.add_argument("--roi-field", default=None)
    metrics_representativeness.set_defaults(func=cmd_metrics_representativeness)

    # brain
    brain_parser = top.add_parser("brain", help="FiftyOne brain-run and index operations")
    brain_sub = brain_parser.add_subparsers(dest="brain_command", required=True)

    brain_visualization = brain_sub.add_parser("visualization", help="Compute dimensionality-reduction visualization")
    brain_visualization.add_argument("--dataset", required=True)
    brain_visualization.add_argument("--method", default="umap")
    brain_visualization.add_argument("--num-dims", type=int, default=2)
    brain_visualization.add_argument("--embeddings-field", default=None)
    brain_visualization.add_argument("--patches-field", default=None)
    brain_visualization.add_argument("--brain-key", default=None)
    brain_visualization.set_defaults(func=cmd_brain_visualization)

    brain_similarity = brain_sub.add_parser("similarity", help="Compute similarity index/run")
    brain_similarity.add_argument("--dataset", required=True)
    brain_similarity.add_argument("--embeddings-field", default=None)
    brain_similarity.add_argument("--patches-field", default=None)
    brain_similarity.add_argument("--roi-field", default=None)
    brain_similarity.add_argument("--backend", default=None)
    brain_similarity.add_argument("--brain-key", default=None)
    brain_similarity.set_defaults(func=cmd_brain_similarity)

    brain_duplicates = brain_sub.add_parser("duplicates", help="Compute duplicates analysis")
    brain_duplicates_sub = brain_duplicates.add_subparsers(dest="brain_duplicates_command", required=True)

    brain_duplicates_exact = brain_duplicates_sub.add_parser("exact", help="Compute exact duplicates")
    brain_duplicates_exact.add_argument("--dataset", required=True)
    brain_duplicates_exact.set_defaults(func=cmd_brain_duplicates_exact)

    brain_duplicates_near = brain_duplicates_sub.add_parser("near", help="Compute near duplicates")
    brain_duplicates_near.add_argument("--dataset", required=True)
    brain_duplicates_near.add_argument("--threshold", type=float, default=0.2)
    brain_duplicates_near.add_argument("--embeddings-field", default=None)
    brain_duplicates_near.add_argument("--roi-field", default=None)
    brain_duplicates_near.set_defaults(func=cmd_brain_duplicates_near)

    brain_leaky_splits = brain_sub.add_parser("leaky-splits", help="Detect cross-split leakage")
    brain_leaky_splits.add_argument("--dataset", required=True)
    brain_leaky_splits.add_argument(
        "--splits",
        required=True,
        help="Comma-separated split tags/values, e.g. train,val,test",
    )
    brain_leaky_splits.add_argument("--threshold", type=float, default=0.2)
    brain_leaky_splits.add_argument("--embeddings-field", default=None)
    brain_leaky_splits.add_argument("--roi-field", default=None)
    brain_leaky_splits.set_defaults(func=cmd_brain_leaky_splits)

    # models
    models_parser = top.add_parser("models", help="Model providers and model resolution")
    models_sub = models_parser.add_subparsers(dest="models_command", required=True)

    models_list = models_sub.add_parser("list", help="List providers or list models for provider")
    models_list.add_argument(
        "--provider",
        choices=["hf", "foz", "anomalib"],
        default=None,
        help="Provider name. If omitted, lists available providers.",
    )
    models_list.add_argument("--contains", default=None, help="Optional substring filter")
    models_list.add_argument("--limit", type=int, default=50, help="Max models to return")
    models_list.set_defaults(func=cmd_models_list)

    models_resolve = models_sub.add_parser("resolve", help="Resolve and load a model from provider")
    models_resolve.add_argument("--model-ref", required=True, help="Provider-qualified model ref")
    models_resolve.add_argument(
        "--default-provider",
        choices=["hf", "foz", "anomalib"],
        default="hf",
        help="Provider used when --model-ref omits prefix",
    )
    models_resolve.add_argument("--task", default=None, help="Optional task hint (e.g. embeddings, anomaly)")
    models_resolve.add_argument("--capability", default=None, help="Required capability (e.g. embeddings, anomaly)")
    models_resolve.set_defaults(func=cmd_models_resolve)

    models_validate = models_sub.add_parser("validate", help="Validate that a model can be loaded")
    models_validate.add_argument("--model-ref", required=True, help="Provider-qualified model ref")
    models_validate.add_argument(
        "--default-provider",
        choices=["hf", "foz", "anomalib"],
        default="hf",
        help="Provider used when --model-ref omits prefix",
    )
    models_validate.add_argument("--task", default=None, help="Optional task hint")
    models_validate.add_argument("--capability", default=None, help="Required capability")
    models_validate.set_defaults(func=cmd_models_validate)

    # anomaly
    anomaly_parser = top.add_parser("anomaly", help="Anomaly detection operations")
    anomaly_sub = anomaly_parser.add_subparsers(dest="anomaly_command", required=True)

    anomaly_fit = anomaly_sub.add_parser("fit", help="Fit anomaly reference data")
    anomaly_fit.add_argument("--dataset", required=True)
    anomaly_fit.add_argument("--backend", choices=["embedding_distance", "anomalib"], default="embedding_distance")
    anomaly_fit.add_argument("--embeddings-field", default="embeddings")
    anomaly_fit.add_argument("--normal-tag", default=None)
    anomaly_fit.add_argument("--threshold", type=float, default=None)
    anomaly_fit.add_argument("--threshold-quantile", type=float, default=0.95)
    anomaly_fit.add_argument("--reference-json", default=None, help="Optional path to persist fitted reference")
    anomaly_fit.set_defaults(func=cmd_anomaly_fit)

    anomaly_train = anomaly_sub.add_parser("train", help="Train and export an anomalib model artifact")
    anomaly_train.add_argument("--dataset", required=True)
    anomaly_train.add_argument("--model-ref", default="anomalib:padim")
    anomaly_train.add_argument("--normal-tag", default=None, help="Tag selecting normal samples for training")
    anomaly_train.add_argument("--abnormal-tag", default=None, help="Optional tag selecting abnormal samples")
    anomaly_train.add_argument("--mask-field", default=None, help="Optional mask field for abnormal samples")
    anomaly_train.add_argument("--artifact-dir", default=None, help="Directory where trained artifact is stored")
    anomaly_train.add_argument("--data-dir", default=None, help="Directory for generated anomalib Folder data")
    anomaly_train.add_argument("--artifact-format", choices=["openvino", "torch"], default="openvino")
    anomaly_train.add_argument("--artifact-json", default=None, help="Optional explicit output path for artifact JSON")
    anomaly_train.add_argument("--image-size", default=None, help="Image size: N or W,H for resize pre-processing")
    anomaly_train.add_argument("--train-batch-size", type=int, default=8)
    anomaly_train.add_argument("--eval-batch-size", type=int, default=8)
    anomaly_train.add_argument("--num-workers", type=int, default=0)
    anomaly_train.add_argument("--normal-split-ratio", type=float, default=0.2)
    anomaly_train.add_argument("--test-split-mode", default="from_dir")
    anomaly_train.add_argument("--test-split-ratio", type=float, default=0.2)
    anomaly_train.add_argument("--val-split-mode", default="same_as_test")
    anomaly_train.add_argument("--val-split-ratio", type=float, default=0.5)
    anomaly_train.add_argument("--seed", type=int, default=None)
    anomaly_train.add_argument("--max-epochs", type=int, default=None)
    anomaly_train.add_argument("--accelerator", default=None)
    anomaly_train.add_argument("--devices", default=None)
    anomaly_train.add_argument("--copy-media", action="store_true", help="Copy files instead of symlinking")
    anomaly_train.add_argument("--overwrite-data", action="store_true", help="Replace existing generated training data directory")
    anomaly_train.set_defaults(func=cmd_anomaly_train)

    anomaly_score = anomaly_sub.add_parser("score", help="Score samples for anomalies")
    anomaly_score.add_argument("--dataset", required=True)
    anomaly_score.add_argument("--backend", choices=["embedding_distance", "anomalib"], default="embedding_distance")
    anomaly_score.add_argument("--embeddings-field", default="embeddings")
    anomaly_score.add_argument("--normal-tag", default=None)
    anomaly_score.add_argument("--threshold", type=float, default=None)
    anomaly_score.add_argument("--threshold-quantile", type=float, default=0.95)
    anomaly_score.add_argument("--reference-json", default=None, help="Optional path to fitted reference JSON")
    anomaly_score.add_argument("--model-ref", default="anomalib:padim", help="Deprecated for backend=anomalib; use --artifact")
    anomaly_score.add_argument("--artifact", default=None, help="Path to anomalib artifact JSON or exported model file")
    anomaly_score.add_argument("--artifact-format", choices=["openvino", "torch"], default=None)
    anomaly_score.add_argument("--anomaly-threshold", type=float, default=0.5, help="Threshold for anomaly flag/label")
    anomaly_score.add_argument("--device", default=None, help="Inference device for anomalib backend")
    anomaly_score.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading torch anomalib artifacts via pickle (only for trusted artifacts)",
    )
    anomaly_score.add_argument("--tag", default=None, help="Optional sample tag filter for scoring")
    anomaly_score.add_argument("--score-field", default="anomaly_score")
    anomaly_score.add_argument("--flag-field", default="is_anomaly")
    anomaly_score.add_argument("--label-field", default=None, help="Optional Classification output field")
    anomaly_score.add_argument("--map-field", default=None, help="Optional Heatmap output field")
    anomaly_score.add_argument("--mask-field", default=None, help="Optional Segmentation output field")
    anomaly_score.set_defaults(func=cmd_anomaly_score)

    anomaly_run = anomaly_sub.add_parser("run", help="Fit and score in one step")
    anomaly_run.add_argument("--dataset", required=True)
    anomaly_run.add_argument("--backend", choices=["embedding_distance", "anomalib"], default="embedding_distance")
    anomaly_run.add_argument("--embeddings-field", default="embeddings")
    anomaly_run.add_argument("--normal-tag", default=None)
    anomaly_run.add_argument("--threshold", type=float, default=None)
    anomaly_run.add_argument("--threshold-quantile", type=float, default=0.95)
    anomaly_run.add_argument("--reference-json", default=None, help="Optional path to persist fitted reference")
    anomaly_run.add_argument("--model-ref", default="anomalib:padim", help="Deprecated for backend=anomalib; use --artifact")
    anomaly_run.add_argument("--artifact", default=None, help="Path to anomalib artifact JSON or exported model file")
    anomaly_run.add_argument("--artifact-format", choices=["openvino", "torch"], default=None)
    anomaly_run.add_argument("--anomaly-threshold", type=float, default=0.5)
    anomaly_run.add_argument("--device", default=None, help="Inference device for anomalib backend")
    anomaly_run.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading torch anomalib artifacts via pickle (only for trusted artifacts)",
    )
    anomaly_run.add_argument("--tag", default=None, help="Optional sample tag filter for scoring")
    anomaly_run.add_argument("--score-field", default="anomaly_score")
    anomaly_run.add_argument("--flag-field", default="is_anomaly")
    anomaly_run.add_argument("--label-field", default=None, help="Optional Classification output field")
    anomaly_run.add_argument("--map-field", default=None, help="Optional Heatmap output field")
    anomaly_run.add_argument("--mask-field", default=None, help="Optional Segmentation output field")
    anomaly_run.set_defaults(func=cmd_anomaly_run)

    # workflow
    workflow_parser = top.add_parser("workflow", help="Tag-processing and LS roundtrip workflows")
    workflow_sub = workflow_parser.add_subparsers(dest="workflow_command", required=True)

    roundtrip = workflow_sub.add_parser("roundtrip", help="Run FiftyOne -> Label Studio -> FiftyOne pipeline")
    _add_common_config_args(roundtrip)
    roundtrip.add_argument("--dataset", required=True)
    roundtrip.add_argument("--tag", default="fix")
    roundtrip.add_argument("--project", default=None)
    roundtrip.add_argument("--label-field", default="ground_truth")
    roundtrip.add_argument("--corrections-field", default="ls_corrections")
    roundtrip.add_argument("--skip-send", action="store_true")
    roundtrip.add_argument("--skip-pull", action="store_true")
    roundtrip.add_argument("--skip-sync-disk", action="store_true")
    roundtrip.add_argument("--dry-run-sync", action="store_true")
    roundtrip.add_argument("--clear-project-tasks", action="store_true")
    roundtrip.add_argument("--upload-strategy", choices=["annotate_batched", "sdk_batched"], default=None)
    roundtrip.add_argument("--pull-strategy", choices=["sdk_meta", "annotate_run"], default=None)
    roundtrip.add_argument("--annotation-key", default=None)
    roundtrip.add_argument("--launch-editor", action="store_true")
    roundtrip.add_argument("--create-if-missing", action="store_true")
    roundtrip.add_argument("--strict-preflight", dest="strict_preflight", action="store_true", default=True)
    roundtrip.add_argument("--no-strict-preflight", dest="strict_preflight", action="store_false")
    roundtrip.add_argument("--overwrite-annotation-run", dest="overwrite_annotation_run", action="store_true", default=True)
    roundtrip.add_argument("--no-overwrite-annotation-run", dest="overwrite_annotation_run", action="store_false")
    roundtrip.add_argument("--send-params", default=None, help="JSON object merged into send params")
    roundtrip.add_argument("--pull-params", default=None, help="JSON object merged into pull params")
    roundtrip.add_argument("--sync-params", default=None, help="JSON object merged into sync params")
    roundtrip.add_argument(
        "--quiet-logs",
        action="store_true",
        help="Suppress noisy stdout/stderr logs from underlying workflow operations",
    )
    roundtrip.add_argument(
        "--output-json",
        default=None,
        help="Write workflow result payload to a JSON file",
    )
    roundtrip.set_defaults(func=cmd_workflow_roundtrip)

    workflow_tags = workflow_sub.add_parser("tags", help="Generic tag workflow runner")
    workflow_tags_sub = workflow_tags.add_subparsers(dest="workflow_tags_command", required=True)

    workflow_tags_run = workflow_tags_sub.add_parser("run", help="Run workflow from JSON file")
    _add_common_config_args(workflow_tags_run)
    workflow_tags_run.add_argument("--workflow", required=True)
    workflow_tags_run.add_argument("--dataset", default=None, help="Override dataset_name from workflow file")
    workflow_tags_run.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=None)
    workflow_tags_run.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    workflow_tags_run.add_argument(
        "--quiet-logs",
        action="store_true",
        help="Suppress noisy stdout/stderr logs from underlying workflow operations",
    )
    workflow_tags_run.add_argument(
        "--output-json",
        default=None,
        help="Write workflow result payload to a JSON file",
    )
    workflow_tags_run.set_defaults(func=cmd_workflow_tags_run)

    workflow_tags_inline = workflow_tags_sub.add_parser("inline", help="Run workflow from inline JSON rules")
    _add_common_config_args(workflow_tags_inline)
    workflow_tags_inline.add_argument("--dataset", required=True)
    workflow_tags_inline.add_argument(
        "--rule",
        action="append",
        required=True,
        help='JSON object, e.g. {"tag":"delete","operation":"delete_samples"}',
    )
    workflow_tags_inline.add_argument("--no-fail-fast", action="store_true")
    workflow_tags_inline.add_argument(
        "--quiet-logs",
        action="store_true",
        help="Suppress noisy stdout/stderr logs from underlying workflow operations",
    )
    workflow_tags_inline.add_argument(
        "--output-json",
        default=None,
        help="Write workflow result payload to a JSON file",
    )
    workflow_tags_inline.set_defaults(func=cmd_workflow_tags_inline)

    # sync
    sync_parser = top.add_parser("sync", help="Synchronization helpers")
    sync_sub = sync_parser.add_subparsers(dest="sync_command", required=True)

    sync_disk = sync_sub.add_parser("disk", help="Write ls_corrections from FiftyOne back to label files")
    _add_common_config_args(sync_disk)
    sync_disk.add_argument("--dataset", default=None)
    sync_disk.add_argument("--tag", default=None)
    sync_disk.add_argument("--dry-run", action="store_true")
    sync_disk.add_argument("--corrections-field", default=None)
    sync_disk.add_argument("--label-to-class-id", default=None, help='JSON object e.g. {"rodent":0}')
    sync_disk.add_argument("--default-class-id", type=int, default=None)
    sync_disk.add_argument(
        "--path-replacement",
        action="append",
        default=None,
        help="Replacement rule SRC=DST, can be repeated",
    )
    sync_disk.add_argument("--backup-suffix-format", default=None)
    sync_disk.set_defaults(func=cmd_sync_disk)

    # app
    app_parser = top.add_parser("app", help="FiftyOne app helpers")
    app_sub = app_parser.add_subparsers(dest="app_command", required=True)

    app_open = app_sub.add_parser("open", help="Open FiftyOne app for a dataset")
    app_open.add_argument("--dataset", required=True)
    app_open.add_argument("--port", type=int, default=5151)
    app_open.add_argument("--address", default="0.0.0.0")
    app_open.add_argument("--no-block", action="store_true", help="Launch app and return immediately")
    app_open.set_defaults(func=cmd_app_open)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI program entrypoint used by `./dst` and `python -m dataset_tools.dst`.

This function parses argv, dispatches to the resolved command handler, converts known runtime/config errors into argparse-style user-facing errors, and prints the resulting payload.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = args.func(args)
    except (ValueError, RuntimeError) as e:
        parser.error(str(e))

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
