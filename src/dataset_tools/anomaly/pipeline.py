"""Implementation module for anomaly analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import fiftyone as fo  # type: ignore

from dataset_tools.anomaly.anomalib import score_with_anomalib_artifact
from dataset_tools.anomaly.base import AnomalyReference


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


def _read_embeddings(view, embeddings_field: str) -> tuple[list[str], list[np.ndarray]]:
    """Internal helper for read embeddings.

Args:
    view: FiftyOne view selecting the sample subset to process.
    embeddings_field: Field containing embeddings vectors.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    sample_ids = list(view.values("id"))
    values = list(view.values(embeddings_field))
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for sample_id, emb in zip(sample_ids, values):
        if emb is None:
            continue
        vec = np.asarray(emb, dtype=float)
        if vec.ndim != 1:
            continue
        ids.append(str(sample_id))
        vectors.append(vec)
    return ids, vectors


def fit_embedding_distance_reference(
    dataset_name: str,
    *,
    embeddings_field: str = "embeddings",
    normal_tag: str | None = None,
    threshold: float | None = None,
    threshold_quantile: float = 0.95,
) -> AnomalyReference:
    """Perform fit embedding distance reference.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    embeddings_field: Field containing embeddings vectors.
    normal_tag: Tag identifying normal samples.
    threshold: Decision/filter threshold used by this operation.
    threshold_quantile: Quantile used to infer a threshold when explicit threshold is omitted.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    dataset = _load_dataset(dataset_name)
    normal_view = _resolve_view(dataset, normal_tag)
    _, vectors = _read_embeddings(normal_view, embeddings_field)
    if not vectors:
        raise RuntimeError(
            f"No valid embeddings found in field '{embeddings_field}' "
            f"for dataset '{dataset_name}' and tag '{normal_tag}'"
        )

    matrix = np.stack(vectors)
    centroid = matrix.mean(axis=0)
    distances = np.linalg.norm(matrix - centroid, axis=1)
    resolved_threshold = float(threshold) if threshold is not None else float(np.quantile(distances, threshold_quantile))

    return AnomalyReference(
        backend="embedding_distance",
        embeddings_field=embeddings_field,
        threshold=resolved_threshold,
        centroid=centroid.astype(float).tolist(),
        metadata={
            "dataset_name": dataset_name,
            "normal_tag": normal_tag,
            "normal_samples": int(len(vectors)),
            "threshold_quantile": float(threshold_quantile),
        },
    )


def score_with_embedding_distance(
    dataset_name: str,
    *,
    reference: AnomalyReference,
    score_field: str = "anomaly_score",
    flag_field: str = "is_anomaly",
    tag_filter: str | None = None,
) -> dict[str, Any]:
    """Perform score with embedding distance.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    reference: Precomputed reference object used for scoring.
    score_field: Sample field name where numeric scores are written.
    flag_field: Sample field name where boolean flags are written.
    tag_filter: Sample tag filter used to restrict processing scope.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    if reference.backend != "embedding_distance":
        raise ValueError(
            "Reference backend mismatch: expected 'embedding_distance', "
            f"got '{reference.backend}'"
        )
    if not reference.centroid:
        raise ValueError("Embedding-distance reference is missing centroid")

    dataset = _load_dataset(dataset_name)
    view = _resolve_view(dataset, tag_filter)
    ids, vectors = _read_embeddings(view, reference.embeddings_field)

    centroid = np.asarray(reference.centroid, dtype=float)
    scores: dict[str, float] = {}
    flags: dict[str, bool] = {}
    for sample_id, vec in zip(ids, vectors):
        distance = float(np.linalg.norm(vec - centroid))
        scores[sample_id] = distance
        flags[sample_id] = bool(distance >= reference.threshold)

    view.set_values(score_field, scores, key_field="id")
    view.set_values(flag_field, flags, key_field="id")

    return {
        "backend": "embedding_distance",
        "dataset": dataset_name,
        "scored_samples": len(scores),
        "anomaly_count": sum(1 for is_anom in flags.values() if is_anom),
        "score_field": score_field,
        "flag_field": flag_field,
        "threshold": reference.threshold,
        "embeddings_field": reference.embeddings_field,
        "tag_filter": tag_filter,
    }


def score_with_anomalib(
    dataset_name: str,
    *,
    artifact_path: str,
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
    """Perform score with anomalib.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    artifact_path: Path to anomaly artifact metadata or model file.
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
    return score_with_anomalib_artifact(
        dataset_name=dataset_name,
        artifact=artifact_path,
        artifact_format=artifact_format,
        threshold=threshold,
        score_field=score_field,
        flag_field=flag_field,
        label_field=label_field,
        map_field=map_field,
        mask_field=mask_field,
        tag_filter=tag_filter,
        device=device,
        trust_remote_code=trust_remote_code,
    )


def save_reference(path: str | Path, reference: AnomalyReference):
    """Save reference to persistent storage.

Args:
    path: Filesystem path used for reading/writing artifacts.
    reference: Precomputed reference object used for scoring.

Returns:
    None or lightweight metadata about the persisted artifact.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(reference.to_dict(), f, indent=2)


def load_reference(path: str | Path) -> AnomalyReference:
    """Load reference required by this module.

Args:
    path: Filesystem path used for reading/writing artifacts.

Returns:
    Loaded object/data required by downstream workflow steps.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return AnomalyReference.from_dict(payload)


def run_embedding_distance(
    dataset_name: str,
    *,
    embeddings_field: str = "embeddings",
    normal_tag: str | None = None,
    score_tag: str | None = None,
    score_field: str = "anomaly_score",
    flag_field: str = "is_anomaly",
    threshold: float | None = None,
    threshold_quantile: float = 0.95,
    reference_path: str | None = None,
) -> dict[str, Any]:
    """Run embedding distance and return execution results.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    embeddings_field: Field containing embeddings vectors.
    normal_tag: Tag identifying normal samples.
    score_tag: Value controlling score tag for this routine.
    score_field: Sample field name where numeric scores are written.
    flag_field: Sample field name where boolean flags are written.
    threshold: Decision/filter threshold used by this operation.
    threshold_quantile: Quantile used to infer a threshold when explicit threshold is omitted.
    reference_path: Path to a serialized reference object.

Returns:
    Result object consumed by the caller or downstream workflow.
    """
    reference = fit_embedding_distance_reference(
        dataset_name=dataset_name,
        embeddings_field=embeddings_field,
        normal_tag=normal_tag,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
    )
    if reference_path:
        save_reference(reference_path, reference)

    payload = score_with_embedding_distance(
        dataset_name=dataset_name,
        reference=reference,
        score_field=score_field,
        flag_field=flag_field,
        tag_filter=score_tag,
    )
    payload["reference"] = reference.to_dict()
    if reference_path:
        payload["reference_path"] = str(reference_path)
    return payload
