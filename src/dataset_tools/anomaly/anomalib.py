"""Implementation module for anomaly analysis.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import fiftyone as fo  # type: ignore

from dataset_tools.models import load_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedAnomalibDataset:
    """PreparedAnomalibDataset used by anomaly analysis.
    """
    dataset_name: str
    root_dir: str
    normal_dir: str
    abnormal_dir: str | None
    mask_dir: str | None
    normal_tag: str | None
    abnormal_tag: str | None
    mask_field: str | None
    normal_count: int
    abnormal_count: int
    mask_count: int
    missing_masks: int

    def to_dict(self) -> dict[str, Any]:
        """Perform to dict.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return {
            "dataset_name": self.dataset_name,
            "root_dir": self.root_dir,
            "normal_dir": self.normal_dir,
            "abnormal_dir": self.abnormal_dir,
            "mask_dir": self.mask_dir,
            "normal_tag": self.normal_tag,
            "abnormal_tag": self.abnormal_tag,
            "mask_field": self.mask_field,
            "normal_count": self.normal_count,
            "abnormal_count": self.abnormal_count,
            "mask_count": self.mask_count,
            "missing_masks": self.missing_masks,
        }


@dataclass(frozen=True)
class AnomalibArtifact:
    """AnomalibArtifact used by anomaly analysis.
    """
    dataset_name: str
    model_ref: str
    export_type: str
    model_path: str
    artifact_dir: str
    checkpoint_path: str | None = None
    prepared_dataset: PreparedAnomalibDataset | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Perform to dict.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return {
            "dataset_name": self.dataset_name,
            "model_ref": self.model_ref,
            "export_type": self.export_type,
            "model_path": self.model_path,
            "artifact_dir": self.artifact_dir,
            "checkpoint_path": self.checkpoint_path,
            "prepared_dataset": self.prepared_dataset.to_dict() if self.prepared_dataset else None,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "AnomalibArtifact":
        """Perform from dict.

Args:
    payload: JSON-like payload consumed by this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        prepared_payload = payload.get("prepared_dataset")
        prepared: PreparedAnomalibDataset | None = None
        if isinstance(prepared_payload, dict):
            prepared = PreparedAnomalibDataset(
                dataset_name=str(prepared_payload["dataset_name"]),
                root_dir=str(prepared_payload["root_dir"]),
                normal_dir=str(prepared_payload["normal_dir"]),
                abnormal_dir=str(prepared_payload["abnormal_dir"]) if prepared_payload.get("abnormal_dir") else None,
                mask_dir=str(prepared_payload["mask_dir"]) if prepared_payload.get("mask_dir") else None,
                normal_tag=str(prepared_payload["normal_tag"]) if prepared_payload.get("normal_tag") else None,
                abnormal_tag=str(prepared_payload["abnormal_tag"]) if prepared_payload.get("abnormal_tag") else None,
                mask_field=str(prepared_payload["mask_field"]) if prepared_payload.get("mask_field") else None,
                normal_count=int(prepared_payload["normal_count"]),
                abnormal_count=int(prepared_payload["abnormal_count"]),
                mask_count=int(prepared_payload["mask_count"]),
                missing_masks=int(prepared_payload["missing_masks"]),
            )
        return AnomalibArtifact(
            dataset_name=str(payload["dataset_name"]),
            model_ref=str(payload["model_ref"]),
            export_type=str(payload["export_type"]),
            model_path=str(payload["model_path"]),
            artifact_dir=str(payload["artifact_dir"]),
            checkpoint_path=(
                str(payload["checkpoint_path"])
                if payload.get("checkpoint_path")
                else (
                    str(payload.get("metadata", {}).get("checkpoint_path"))
                    if payload.get("metadata", {}).get("checkpoint_path")
                    else None
                )
            ),
            prepared_dataset=prepared,
            metadata=dict(payload.get("metadata", {})),
        )


def _load_dataset(dataset_name: str):
    """Load dataset required by this module.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if dataset_name not in fo.list_datasets():
        raise RuntimeError(f"Dataset '{dataset_name}' not found")
    return fo.load_dataset(dataset_name)


def _resolve_view(dataset, tag_filter: str | None):
    """Resolve view from provided inputs.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.
    tag_filter: Sample tag filter used to restrict processing scope.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if tag_filter:
        return dataset.match_tags(tag_filter)
    return dataset


def _safe_name(value: str) -> str:
    """Internal helper for safe name.

Args:
    value: Input value to normalize or validate.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value.strip().lower())
    return cleaned.strip("_") or "dataset"


def _clear_dir(path: Path, overwrite: bool):
    """Internal helper for clear dir.

Args:
    path: Filesystem path used for reading/writing artifacts.
    overwrite: Whether existing resources should be replaced.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if path.exists():
        if not overwrite:
            raise RuntimeError(
                f"Directory '{path}' already exists. "
                "Use overwrite_data=True to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, *, symlink: bool):
    """Internal helper for link or copy.

Args:
    src: Value controlling src for this routine.
    dst: Value controlling dst for this routine.
    symlink: Value controlling symlink for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def _sample_target_name(sample_id: str, filepath: str) -> str:
    """Internal helper for sample target name.

Args:
    sample_id: Value controlling sample id for this routine.
    filepath: Filesystem path to a file.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    suffix = Path(filepath).suffix
    if not suffix:
        suffix = ".jpg"
    return f"{sample_id}{suffix}"


def _extract_mask_path(sample: Any, mask_field: str) -> str | None:
    """Internal helper for extract mask path.

Args:
    sample: Value controlling sample for this routine.
    mask_field: Field containing segmentation masks or anomaly masks.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    value = getattr(sample, mask_field, None)
    if value is None:
        try:
            value = sample[mask_field]
        except Exception:
            value = None
    if value is None:
        return None

    if isinstance(value, str):
        return value
    if hasattr(value, "mask_path"):
        mask_path = getattr(value, "mask_path")
        return str(mask_path) if mask_path else None
    if isinstance(value, dict):
        for key in ("mask_path", "path", "filepath"):
            candidate = value.get(key)
            if candidate:
                return str(candidate)
    return None


def prepare_anomalib_folder_dataset(
    dataset_name: str,
    *,
    output_root: str | Path,
    normal_tag: str | None = None,
    abnormal_tag: str | None = None,
    mask_field: str | None = None,
    symlink: bool = True,
    overwrite_data: bool = False,
) -> PreparedAnomalibDataset:
    """Perform prepare anomalib folder dataset.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    output_root: Value controlling output root for this routine.
    normal_tag: Tag identifying normal samples.
    abnormal_tag: Tag identifying abnormal samples.
    mask_field: Field containing segmentation masks or anomaly masks.
    symlink: Value controlling symlink for this routine.
    overwrite_data: Value controlling overwrite data for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    dataset = _load_dataset(dataset_name)
    normal_view = _resolve_view(dataset, normal_tag)
    abnormal_view = _resolve_view(dataset, abnormal_tag) if abnormal_tag else None

    if len(normal_view) == 0:
        raise RuntimeError(
            "No normal samples were selected for anomalib training. "
            f"dataset='{dataset_name}', normal_tag='{normal_tag}'"
        )

    root_dir = Path(output_root).expanduser().resolve()
    _clear_dir(root_dir, overwrite=overwrite_data)

    normal_dir = root_dir / "normal"
    abnormal_dir = root_dir / "abnormal"
    mask_dir = root_dir / "mask"
    normal_dir.mkdir(parents=True, exist_ok=True)

    normal_count = 0
    abnormal_count = 0
    mask_count = 0
    missing_masks = 0

    for sample in normal_view.iter_samples(progress=False):
        src = Path(sample.filepath).expanduser().resolve()
        dst = normal_dir / _sample_target_name(str(sample.id), sample.filepath)
        _link_or_copy(src, dst, symlink=symlink)
        normal_count += 1

    resolved_abnormal_dir: str | None = None
    resolved_mask_dir: str | None = None
    if abnormal_view is not None and len(abnormal_view) > 0:
        abnormal_dir.mkdir(parents=True, exist_ok=True)
        resolved_abnormal_dir = str(abnormal_dir)
        if mask_field:
            mask_dir.mkdir(parents=True, exist_ok=True)
            resolved_mask_dir = str(mask_dir)

        for sample in abnormal_view.iter_samples(progress=False):
            src = Path(sample.filepath).expanduser().resolve()
            target_name = _sample_target_name(str(sample.id), sample.filepath)
            dst = abnormal_dir / target_name
            _link_or_copy(src, dst, symlink=symlink)
            abnormal_count += 1

            if mask_field:
                mask_path = _extract_mask_path(sample, mask_field)
                if mask_path:
                    mask_src = Path(mask_path).expanduser().resolve()
                    mask_dst = mask_dir / target_name
                    _link_or_copy(mask_src, mask_dst, symlink=symlink)
                    mask_count += 1
                else:
                    missing_masks += 1

    return PreparedAnomalibDataset(
        dataset_name=dataset_name,
        root_dir=str(root_dir),
        normal_dir=str(normal_dir),
        abnormal_dir=resolved_abnormal_dir,
        mask_dir=resolved_mask_dir,
        normal_tag=normal_tag,
        abnormal_tag=abnormal_tag,
        mask_field=mask_field,
        normal_count=normal_count,
        abnormal_count=abnormal_count,
        mask_count=mask_count,
        missing_masks=missing_masks,
    )


def _import_anomalib_components():
    """Internal helper for import anomalib components.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    from anomalib.deploy import ExportType, OpenVINOInferencer, TorchInferencer  # type: ignore
    from anomalib.engine import Engine  # type: ignore

    folder_cls = None
    try:
        from anomalib.data.datamodules.image.folder import Folder as folder_cls  # type: ignore
    except Exception:
        from anomalib.data.image.folder import Folder as folder_cls  # type: ignore

    return Engine, ExportType, OpenVINOInferencer, TorchInferencer, folder_cls


def _parse_image_size(value: str | None) -> tuple[int, int] | None:
    """Parse and validate image size input values.

Args:
    value: Input value to normalize or validate.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if "," in raw:
        w, h = raw.split(",", 1)
        return (int(w.strip()), int(h.strip()))
    side = int(raw)
    return (side, side)


def _create_datamodule(
    prepared: PreparedAnomalibDataset,
    *,
    datamodule_name: str,
    image_size: tuple[int, int] | None = None,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_workers: int = 0,
    normal_split_ratio: float = 0.2,
    test_split_mode: str = "from_dir",
    test_split_ratio: float = 0.2,
    val_split_mode: str = "same_as_test",
    val_split_ratio: float = 0.5,
    seed: int | None = None,
):
    """Internal helper for create datamodule.

Args:
    prepared: Value controlling prepared for this routine.
    datamodule_name: Value controlling datamodule name for this routine.
    image_size: Value controlling image size for this routine.
    train_batch_size: Batch size used during training.
    eval_batch_size: Batch size used during evaluation/inference.
    num_workers: Worker count used by data loading operations.
    normal_split_ratio: Value controlling normal split ratio for this routine.
    test_split_mode: Value controlling test split mode for this routine.
    test_split_ratio: Value controlling test split ratio for this routine.
    val_split_mode: Value controlling val split mode for this routine.
    val_split_ratio: Value controlling val split ratio for this routine.
    seed: Value controlling seed for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    _, _, _, _, folder_cls = _import_anomalib_components()

    kwargs: dict[str, Any] = {
        "name": datamodule_name,
        "root": prepared.root_dir,
        "normal_dir": prepared.normal_dir,
        "abnormal_dir": prepared.abnormal_dir,
        "mask_dir": prepared.mask_dir,
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "num_workers": int(num_workers),
        "normal_split_ratio": float(normal_split_ratio),
        "test_split_mode": str(test_split_mode),
        "test_split_ratio": float(test_split_ratio),
        "val_split_mode": str(val_split_mode),
        "val_split_ratio": float(val_split_ratio),
        "seed": seed,
    }

    if image_size is not None:
        from torchvision.transforms.v2 import Resize  # type: ignore

        resize = Resize(image_size, antialias=True)
        kwargs["train_augmentations"] = resize
        kwargs["val_augmentations"] = resize
        kwargs["test_augmentations"] = resize

    datamodule = folder_cls(**kwargs)
    datamodule.setup()
    return datamodule


def save_anomalib_artifact(path: str | Path, artifact: AnomalibArtifact):
    """Save anomalib artifact to persistent storage.

Args:
    path: Filesystem path used for reading/writing artifacts.
    artifact: Anomaly artifact path or descriptor.

Returns:
    None or lightweight metadata about the persisted artifact.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(artifact.to_dict(), f, indent=2)


def load_anomalib_artifact(path: str | Path) -> AnomalibArtifact:
    """Load anomalib artifact required by this module.

Args:
    path: Filesystem path used for reading/writing artifacts.

Returns:
    Loaded object/data required by downstream workflow steps.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return AnomalibArtifact.from_dict(payload)


def train_and_export_anomalib(
    dataset_name: str,
    *,
    model_ref: str = "anomalib:padim",
    normal_tag: str | None = None,
    abnormal_tag: str | None = None,
    mask_field: str | None = None,
    artifact_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    artifact_format: str = "openvino",
    image_size: str | None = None,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_workers: int = 0,
    normal_split_ratio: float = 0.2,
    test_split_mode: str = "from_dir",
    test_split_ratio: float = 0.2,
    val_split_mode: str = "same_as_test",
    val_split_ratio: float = 0.5,
    seed: int | None = None,
    max_epochs: int | None = None,
    accelerator: str | None = None,
    devices: str | int | None = None,
    symlink: bool = True,
    overwrite_data: bool = False,
    artifact_json: str | Path | None = None,
) -> AnomalibArtifact:
    """Perform train and export anomalib.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    model_ref: Provider-qualified model reference (for example `hf:...`, `foz:...`, `anomalib:...`).
    normal_tag: Tag identifying normal samples.
    abnormal_tag: Tag identifying abnormal samples.
    mask_field: Field containing segmentation masks or anomaly masks.
    artifact_dir: Value controlling artifact dir for this routine.
    data_dir: Value controlling data dir for this routine.
    artifact_format: Artifact runtime format (`openvino` or `torch`).
    image_size: Value controlling image size for this routine.
    train_batch_size: Batch size used during training.
    eval_batch_size: Batch size used during evaluation/inference.
    num_workers: Worker count used by data loading operations.
    normal_split_ratio: Value controlling normal split ratio for this routine.
    test_split_mode: Value controlling test split mode for this routine.
    test_split_ratio: Value controlling test split ratio for this routine.
    val_split_mode: Value controlling val split mode for this routine.
    val_split_ratio: Value controlling val split ratio for this routine.
    seed: Value controlling seed for this routine.
    max_epochs: Value controlling max epochs for this routine.
    accelerator: Value controlling accelerator for this routine.
    devices: Device selection passed through to underlying runtime.
    symlink: Value controlling symlink for this routine.
    overwrite_data: Value controlling overwrite data for this routine.
    artifact_json: Value controlling artifact json for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if artifact_format not in {"openvino", "torch"}:
        raise ValueError("artifact_format must be one of: openvino, torch")
    if artifact_format == "openvino":
        try:
            import openvino  # type: ignore # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "OpenVINO export requested but 'openvino' is not installed in the current environment. "
                "Install OpenVINO or use --artifact-format torch."
            ) from e

    base_artifact_dir = (
        Path(artifact_dir).expanduser().resolve()
        if artifact_dir is not None
        else Path.cwd() / "artifacts" / "anomaly" / _safe_name(dataset_name)
    )
    base_artifact_dir.mkdir(parents=True, exist_ok=True)

    training_data_dir = (
        Path(data_dir).expanduser().resolve()
        if data_dir is not None
        else base_artifact_dir / "training_data"
    )

    prepared = prepare_anomalib_folder_dataset(
        dataset_name=dataset_name,
        output_root=training_data_dir,
        normal_tag=normal_tag,
        abnormal_tag=abnormal_tag,
        mask_field=mask_field,
        symlink=symlink,
        overwrite_data=overwrite_data,
    )

    image_size_tuple = _parse_image_size(image_size)
    datamodule = _create_datamodule(
        prepared,
        datamodule_name=_safe_name(dataset_name),
        image_size=image_size_tuple,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        normal_split_ratio=normal_split_ratio,
        test_split_mode=test_split_mode,
        test_split_ratio=test_split_ratio,
        val_split_mode=val_split_mode,
        val_split_ratio=val_split_ratio,
        seed=seed,
    )

    loaded = load_model(
        model_ref,
        default_provider="anomalib",
        task="anomaly",
        capability="anomaly",
    )
    model = loaded.model

    engine_kwargs: dict[str, Any] = {
        "default_root_dir": str(base_artifact_dir),
    }
    if max_epochs is not None:
        engine_kwargs["max_epochs"] = int(max_epochs)
    if accelerator:
        engine_kwargs["accelerator"] = accelerator
    if devices is not None:
        engine_kwargs["devices"] = devices

    Engine, ExportType, _, _, _ = _import_anomalib_components()
    engine = Engine(**engine_kwargs)
    engine.fit(model=model, datamodule=datamodule)

    checkpoint_path: str | None = None
    try:
        checkpoint_dir = base_artifact_dir / "weights" / "lightning"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "model.ckpt"
        engine.trainer.save_checkpoint(str(checkpoint_file))
        checkpoint_path = str(checkpoint_file.resolve())
    except Exception as e:
        logger.warning("Failed to persist anomalib training checkpoint: %s", e)

    exported = engine.export(
        model=model,
        export_type=ExportType(artifact_format),
        export_root=base_artifact_dir,
        datamodule=datamodule,
        input_size=image_size_tuple,
    )
    if exported is None:
        raise RuntimeError("anomalib export failed: no model path returned")

    model_path = Path(exported).expanduser().resolve()
    artifact = AnomalibArtifact(
        dataset_name=dataset_name,
        model_ref=model_ref,
        export_type=artifact_format,
        model_path=str(model_path),
        artifact_dir=str(base_artifact_dir),
        checkpoint_path=checkpoint_path,
        prepared_dataset=prepared,
        metadata={
            "model_type": type(model).__name__,
            "image_size": image_size_tuple,
            "normal_tag": normal_tag,
            "abnormal_tag": abnormal_tag,
            "mask_field": mask_field,
            "train_batch_size": int(train_batch_size),
            "eval_batch_size": int(eval_batch_size),
            "num_workers": int(num_workers),
            "max_epochs": int(max_epochs) if max_epochs is not None else None,
            "accelerator": accelerator,
            "devices": devices,
            "checkpoint_path": checkpoint_path,
        },
    )

    artifact_manifest = (
        Path(artifact_json).expanduser().resolve()
        if artifact_json is not None
        else base_artifact_dir / "anomalib_artifact.json"
    )
    save_anomalib_artifact(artifact_manifest, artifact)
    return artifact


def _extract_prediction_fields(prediction: Any) -> tuple[float, bool, np.ndarray | None, np.ndarray | None]:
    """Internal helper for extract prediction fields.

Args:
    prediction: Value controlling prediction for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if isinstance(prediction, dict):
        score = prediction.get("pred_score", prediction.get("anomaly_score", prediction.get("score")))
        label = prediction.get("pred_label", prediction.get("is_anomaly", prediction.get("anomaly")))
        anomaly_map = prediction.get("anomaly_map")
        pred_mask = prediction.get("pred_mask")
    else:
        score = getattr(prediction, "pred_score", None)
        if score is None:
            score = getattr(prediction, "anomaly_score", None)
        if score is None:
            score = getattr(prediction, "score", None)

        label = getattr(prediction, "pred_label", None)
        if label is None:
            label = getattr(prediction, "is_anomaly", None)
        if label is None:
            label = getattr(prediction, "anomaly", None)

        anomaly_map = getattr(prediction, "anomaly_map", None)
        pred_mask = getattr(prediction, "pred_mask", None)

    if score is None:
        raise RuntimeError(
            "Could not extract anomaly score from anomalib prediction. "
            "Expected: pred_score | anomaly_score | score"
        )

    parsed_score = float(_to_numpy(score).reshape(-1)[0])
    parsed_label = bool(_to_numpy(label).reshape(-1)[0]) if label is not None else parsed_score >= 0.5

    map_arr = None
    if anomaly_map is not None:
        map_arr = _to_numpy(anomaly_map)
        if map_arr.ndim >= 3:
            map_arr = np.squeeze(map_arr)
        if map_arr.ndim == 0:
            map_arr = None

    mask_arr = None
    if pred_mask is not None:
        mask_arr = _to_numpy(pred_mask)
        if mask_arr.ndim >= 3:
            mask_arr = np.squeeze(mask_arr)
        if mask_arr.ndim >= 2:
            mask_arr = (mask_arr > 0).astype(np.uint8)
        else:
            mask_arr = None

    return parsed_score, parsed_label, map_arr, mask_arr


def _to_numpy(value: Any) -> np.ndarray:
    """Internal helper for to numpy.

Args:
    value: Input value to normalize or validate.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    arr = value
    if hasattr(arr, "detach"):
        try:
            arr = arr.detach()
        except Exception:
            pass
    if hasattr(arr, "cpu"):
        try:
            arr = arr.cpu()
        except Exception:
            pass
    if hasattr(arr, "numpy"):
        try:
            return np.asarray(arr.numpy())
        except Exception:
            pass
    return np.asarray(arr)


def _assert_trusted_torch_loading(trust_remote_code: bool):
    """Internal helper for assert trusted torch loading.

Args:
    trust_remote_code: Value controlling trust remote code for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if trust_remote_code:
        os.environ["TRUST_REMOTE_CODE"] = "1"
        return

    if os.getenv("TRUST_REMOTE_CODE") == "1":
        return

    raise RuntimeError(
        "Torch anomalib artifacts require loading serialized checkpoints via pickle. "
        "Set environment variable TRUST_REMOTE_CODE=1 or pass trust_remote_code=True "
        "only if you trust the artifact source."
    )


def _infer_export_type(path: Path) -> str:
    """Internal helper for infer export type.

Args:
    path: Filesystem path used for reading/writing artifacts.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    suffix = path.suffix.lower()
    if suffix in {".xml", ".bin"}:
        return "openvino"
    if suffix in {".pt", ".pth", ".ckpt"}:
        return "torch"
    raise ValueError(
        "Unable to infer anomalib artifact format from file extension. "
        "Provide artifact_format explicitly."
    )


def _resolve_anomalib_artifact(
    artifact: str | Path,
    *,
    artifact_format: str | None = None,
) -> AnomalibArtifact:
    """Resolve anomalib artifact from provided inputs.

Args:
    artifact: Anomaly artifact path or descriptor.
    artifact_format: Artifact runtime format (`openvino` or `torch`).

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    artifact_path = Path(artifact).expanduser().resolve()
    if artifact_path.suffix.lower() == ".json":
        loaded = load_anomalib_artifact(artifact_path)
        return loaded

    export_type = artifact_format or _infer_export_type(artifact_path)
    return AnomalibArtifact(
        dataset_name="",
        model_ref="",
        export_type=export_type,
        model_path=str(artifact_path),
        artifact_dir=str(artifact_path.parent),
        checkpoint_path=None,
        prepared_dataset=None,
        metadata={},
    )


def _build_inferencer(
    artifact: AnomalibArtifact,
    *,
    device: str | None = None,
    trust_remote_code: bool = False,
):
    """Build inferencer for downstream steps.

Args:
    artifact: Anomaly artifact path or descriptor.
    device: Runtime device selection for inference/training.
    trust_remote_code: Value controlling trust remote code for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    _, _, OpenVINOInferencer, TorchInferencer, _ = _import_anomalib_components()
    model_path = Path(artifact.model_path).expanduser().resolve()
    if not model_path.exists():
        raise RuntimeError(f"Anomalib model artifact not found: '{model_path}'")

    export_type = artifact.export_type.strip().lower()
    if export_type == "openvino":
        if model_path.suffix.lower() == ".bin":
            xml_candidate = model_path.with_suffix(".xml")
            if xml_candidate.exists():
                model_path = xml_candidate
        return OpenVINOInferencer(path=model_path, device=device or "AUTO")

    if export_type == "torch":
        _assert_trusted_torch_loading(trust_remote_code=trust_remote_code)
        return TorchInferencer(path=model_path, device=device or "auto")

    raise ValueError(f"Unsupported anomalib artifact export_type '{artifact.export_type}'")


def _score_with_engine_predict(
    *,
    artifact: AnomalibArtifact,
    sample_ids: list[str],
    filepaths: list[str],
    threshold: float,
    device: str | None,
    trust_remote_code: bool,
) -> tuple[dict[str, float], dict[str, bool], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Internal helper for score with engine predict.

Args:
    artifact: Anomaly artifact path or descriptor.
    sample_ids: Value controlling sample ids for this routine.
    filepaths: Value controlling filepaths for this routine.
    threshold: Decision/filter threshold used by this operation.
    device: Runtime device selection for inference/training.
    trust_remote_code: Value controlling trust remote code for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    _assert_trusted_torch_loading(trust_remote_code=trust_remote_code)

    if not artifact.model_ref:
        raise RuntimeError("Artifact is missing model_ref required for Engine.predict")

    checkpoint_raw = artifact.checkpoint_path or artifact.metadata.get("checkpoint_path")
    if not checkpoint_raw:
        raise RuntimeError("Artifact is missing checkpoint_path required for Engine.predict")
    checkpoint_path = Path(str(checkpoint_raw)).expanduser().resolve()
    if not checkpoint_path.exists():
        raise RuntimeError(f"Anomalib checkpoint not found for Engine.predict: '{checkpoint_path}'")

    loaded = load_model(
        artifact.model_ref,
        default_provider="anomalib",
        task="anomaly",
        capability="anomaly",
    )
    model = loaded.model

    engine_kwargs: dict[str, Any] = {
        "default_root_dir": str(Path(artifact.artifact_dir).expanduser().resolve()),
    }
    if device:
        lowered = device.strip().lower()
        if lowered in {"cpu"}:
            engine_kwargs["accelerator"] = "cpu"
        elif lowered in {"cuda", "gpu"}:
            engine_kwargs["accelerator"] = "gpu"

    Engine, _, _, _, _ = _import_anomalib_components()
    engine = Engine(**engine_kwargs)

    score_values: dict[str, float] = {}
    flag_values: dict[str, bool] = {}
    map_values: dict[str, np.ndarray] = {}
    mask_values: dict[str, np.ndarray] = {}

    with tempfile.TemporaryDirectory(prefix="dst_anomalib_predict_") as tmpdir:
        predict_root = Path(tmpdir)
        path_to_id: dict[str, str] = {}
        for sample_id, filepath in zip(sample_ids, filepaths):
            src = Path(filepath).expanduser().resolve()
            dst = predict_root / _sample_target_name(sample_id, filepath)
            _link_or_copy(src, dst, symlink=True)
            path_to_id[str(dst)] = sample_id
            path_to_id[str(dst.resolve())] = sample_id

        predictions = engine.predict(
            model=model,
            data_path=str(predict_root),
            ckpt_path=str(checkpoint_path),
            return_predictions=True,
        )

    if predictions is None:
        raise RuntimeError("Engine.predict returned no predictions")

    for batch in predictions:
        items = list(batch) if hasattr(batch, "__iter__") else [batch]
        for item in items:
            image_path = getattr(item, "image_path", None)
            if image_path is None:
                continue
            resolved_path = str(Path(str(image_path)).expanduser().resolve())
            sid = path_to_id.get(str(image_path)) or path_to_id.get(resolved_path)
            if sid is None:
                continue

            score, flag, anomaly_map, pred_mask = _extract_prediction_fields(item)
            is_anomaly = bool(flag) if flag is not None else bool(score >= threshold)
            if score >= threshold and not is_anomaly:
                is_anomaly = True

            score_values[sid] = float(score)
            flag_values[sid] = bool(is_anomaly)
            if anomaly_map is not None:
                map_values[sid] = anomaly_map
            if pred_mask is not None:
                mask_values[sid] = pred_mask

    return score_values, flag_values, map_values, mask_values


def score_with_anomalib_artifact(
    dataset_name: str,
    *,
    artifact: str | Path,
    artifact_format: str | None = None,
    threshold: float = 0.5,
    score_field: str = "anomaly_score",
    flag_field: str = "is_anomaly",
    label_field: str | None = None,
    map_field: str | None = None,
    mask_field: str | None = None,
    tag_filter: str | None = None,
    device: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Perform score with anomalib artifact.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    artifact: Anomaly artifact path or descriptor.
    artifact_format: Artifact runtime format (`openvino` or `torch`).
    threshold: Decision/filter threshold used by this operation.
    score_field: Sample field name where numeric scores are written.
    flag_field: Sample field name where boolean flags are written.
    label_field: Field name containing labels for this operation.
    map_field: Value controlling map field for this routine.
    mask_field: Field containing segmentation masks or anomaly masks.
    tag_filter: Sample tag filter used to restrict processing scope.
    device: Runtime device selection for inference/training.
    trust_remote_code: Value controlling trust remote code for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    resolved = _resolve_anomalib_artifact(
        artifact=artifact,
        artifact_format=artifact_format,
    )

    dataset = _load_dataset(dataset_name)
    view = _resolve_view(dataset, tag_filter)

    sample_ids = list(view.values("id"))
    filepaths = list(view.values("filepath"))

    if len(sample_ids) == 0:
        return {
            "backend": "anomalib",
            "dataset": dataset_name,
            "artifact": str(artifact),
            "artifact_format": resolved.export_type,
            "scored_samples": 0,
            "anomaly_count": 0,
            "score_field": score_field,
            "flag_field": flag_field,
            "label_field": label_field,
            "map_field": map_field,
            "mask_field": mask_field,
            "tag_filter": tag_filter,
            "threshold": float(threshold),
        }

    score_values: dict[str, float] = {}
    flag_values: dict[str, bool] = {}
    label_values: dict[str, fo.Classification] = {}
    map_values: dict[str, fo.Heatmap] = {}
    mask_values: dict[str, fo.Segmentation] = {}

    inference_mode = "inferencer"
    use_engine_predict = resolved.export_type.strip().lower() == "torch"

    if use_engine_predict:
        # Enforce trust policy before attempting any torch-based anomalib runtime path.
        # Without this fast-fail, the engine path error gets swallowed by fallback logic.
        _assert_trusted_torch_loading(trust_remote_code=trust_remote_code)
        try:
            score_values, flag_values, map_arrays, mask_arrays = _score_with_engine_predict(
                artifact=resolved,
                sample_ids=[str(sid) for sid in sample_ids],
                filepaths=[str(path) for path in filepaths],
                threshold=threshold,
                device=device,
                trust_remote_code=trust_remote_code,
            )
            if len(score_values) != len(sample_ids):
                raise RuntimeError(
                    f"Engine.predict produced {len(score_values)} predictions "
                    f"for {len(sample_ids)} samples"
                )
            inference_mode = "engine_predict"
            for sid, arr in map_arrays.items():
                map_values[sid] = fo.Heatmap(map=arr)
            for sid, arr in mask_arrays.items():
                mask_values[sid] = fo.Segmentation(mask=arr)
        except Exception as e:
            logger.warning(
                "Engine.predict path failed for torch artifact, falling back to TorchInferencer: %s",
                e,
            )
            use_engine_predict = False

    if not use_engine_predict:
        inferencer = _build_inferencer(
            resolved,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        for sample_id, filepath in zip(sample_ids, filepaths):
            prediction = inferencer.predict(image=filepath)
            score, flag, anomaly_map, pred_mask = _extract_prediction_fields(prediction)
            is_anomaly = bool(flag) if flag is not None else bool(score >= threshold)
            if score >= threshold and not is_anomaly:
                is_anomaly = True

            sid = str(sample_id)
            score_values[sid] = float(score)
            flag_values[sid] = bool(is_anomaly)
            if map_field and anomaly_map is not None:
                map_values[sid] = fo.Heatmap(map=anomaly_map)
            if mask_field and pred_mask is not None:
                mask_values[sid] = fo.Segmentation(mask=pred_mask)

    for sid, is_anomaly in flag_values.items():
        if label_field:
            label_values[sid] = fo.Classification(label="anomaly" if is_anomaly else "normal")

    view.set_values(score_field, score_values, key_field="id")
    view.set_values(flag_field, flag_values, key_field="id")
    if label_field:
        view.set_values(label_field, label_values, key_field="id")
    if map_field and map_values:
        view.set_values(map_field, map_values, key_field="id")
    if mask_field and mask_values:
        view.set_values(mask_field, mask_values, key_field="id")

    return {
        "backend": "anomalib",
        "dataset": dataset_name,
        "artifact": str(artifact),
        "artifact_format": resolved.export_type,
        "model_path": resolved.model_path,
        "checkpoint_path": resolved.checkpoint_path,
        "inference_mode": inference_mode,
        "scored_samples": len(score_values),
        "anomaly_count": sum(1 for value in flag_values.values() if value),
        "score_field": score_field,
        "flag_field": flag_field,
        "label_field": label_field,
        "map_field": map_field,
        "mask_field": mask_field,
        "tag_filter": tag_filter,
        "threshold": float(threshold),
    }
