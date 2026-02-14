"""Analysis-focused tag-workflow operations.

These operations compute FiftyOne brain/metric/anomaly outputs and are wired
into the same rule engine as core mutation/sync operations.
"""
from __future__ import annotations

from typing import Any

import fiftyone as fo  # type: ignore
import fiftyone.brain as fob  # type: ignore

from dataset_tools.anomaly import (
    fit_embedding_distance_reference,
    score_with_anomalib,
    score_with_embedding_distance,
)
from dataset_tools.metrics.representativeness import RepresentativenessComputation
from dataset_tools.tag_workflow.context import TagWorkflowContext
from dataset_tools.tag_workflow.operations.base import TagOperation


def _resolve_scope_collection(
    context: TagWorkflowContext,
    view: Any,
    params: dict[str, Any],
    *,
    operation: str,
    default_scope: str,
) -> tuple[Any, str]:
    """Resolve whether an operation should run on the full dataset or current view.

    Supported scope values:
    - ``dataset``: run against ``context.dataset``
    - ``view``: run against rule-selected ``view``
    """
    scope = str(params.get("scope", default_scope))
    if scope == "dataset":
        return context.dataset, scope
    if scope == "view":
        return view, scope
    raise ValueError(
        f"Invalid scope '{scope}' for operation '{operation}'. Use 'view' or 'dataset'"
    )


def _require_dataset_scope(params: dict[str, Any], operation: str):
    """Enforce dataset-global scope for operations that cannot run on a view."""
    scope = params.get("scope")
    if scope != "dataset":
        raise ValueError(
            f"Operation '{operation}' is dataset-global. Set params.scope='dataset' explicitly"
        )


def _ensure_sample_field(collection: Any, field: str, dataset_name: str):
    """Raise if ``field`` is missing on target sample collection."""
    if not collection.has_sample_field(field):
        raise RuntimeError(
            f"Dataset '{dataset_name}' is missing required sample field '{field}'"
        )


class ComputeUniquenessOperation(TagOperation):
    """Compute uniqueness scores for samples in view or dataset scope."""
    name = "compute_uniqueness"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Compute and store uniqueness field via ``fiftyone.brain.compute_uniqueness``."""
        target, scope = _resolve_scope_collection(
            context=context,
            view=view,
            params=params,
            operation=self.name,
            default_scope="view",
        )

        embeddings_field = params.get("embeddings_field")
        output_field = str(params.get("output_field", "uniqueness"))

        if embeddings_field:
            _ensure_sample_field(target, str(embeddings_field), context.dataset_name)

        if len(target) == 0:
            return {
                "operation": self.name,
                "tag": tag,
                "scope": scope,
                "field": output_field,
                "sample_count": 0,
                "skipped": True,
            }

        fob.compute_uniqueness(
            target,
            embeddings=embeddings_field,
            uniqueness_field=output_field,
        )
        return {
            "operation": self.name,
            "tag": tag,
            "scope": scope,
            "field": output_field,
            "sample_count": len(target),
        }


class ComputeHardnessOperation(TagOperation):
    """Compute hardness for classification-style fields."""
    name = "compute_hardness"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Validate label type and compute hardness scores for target scope."""
        target, scope = _resolve_scope_collection(
            context=context,
            view=view,
            params=params,
            operation=self.name,
            default_scope="view",
        )

        label_field = str(params.get("label_field", "ground_truth"))
        output_field = str(params.get("output_field", "hardness"))

        _ensure_sample_field(target, label_field, context.dataset_name)

        field = target.get_field(label_field)
        doc_type = getattr(field, "document_type", None)
        if doc_type not in (fo.Classification, fo.Classifications):
            raise RuntimeError(
                "Hardness requires a classification-style field "
                f"({fo.Classification.__name__} or {fo.Classifications.__name__}); "
                f"'{label_field}' is {getattr(doc_type, '__name__', str(doc_type))}"
            )

        if len(target) == 0:
            return {
                "operation": self.name,
                "tag": tag,
                "scope": scope,
                "field": output_field,
                "sample_count": 0,
                "skipped": True,
            }

        fob.compute_hardness(
            target,
            label_field=label_field,
            hardness_field=output_field,
        )
        return {
            "operation": self.name,
            "tag": tag,
            "scope": scope,
            "field": output_field,
            "label_field": label_field,
            "sample_count": len(target),
        }


class ComputeRepresentativenessOperation(TagOperation):
    """Compute representativeness metrics with optional embeddings/ROI settings."""
    name = "compute_representativeness"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Compute representativeness and enforce clustering prerequisites."""
        target, scope = _resolve_scope_collection(
            context=context,
            view=view,
            params=params,
            operation=self.name,
            default_scope="view",
        )

        output_field = str(params.get("output_field", "representativeness"))
        method = str(params.get("method", "cluster-center"))
        embeddings_field = params.get("embeddings_field")
        roi_field = params.get("roi_field")

        if embeddings_field:
            _ensure_sample_field(target, str(embeddings_field), context.dataset_name)
        if roi_field:
            _ensure_sample_field(target, str(roi_field), context.dataset_name)

        if method in ("cluster-center", "cluster-center-downweight") and len(target) < RepresentativenessComputation.MIN_CLUSTER_SAMPLES:
            raise RuntimeError(
                "Representativeness clustering requires at least "
                f"{RepresentativenessComputation.MIN_CLUSTER_SAMPLES} samples; "
                f"selected scope has {len(target)}"
            )

        if len(target) == 0:
            return {
                "operation": self.name,
                "tag": tag,
                "scope": scope,
                "field": output_field,
                "sample_count": 0,
                "skipped": True,
            }

        fob.compute_representativeness(
            target,
            representativeness_field=output_field,
            method=method,
            embeddings=embeddings_field,
            roi_field=roi_field,
        )
        return {
            "operation": self.name,
            "tag": tag,
            "scope": scope,
            "field": output_field,
            "method": method,
            "sample_count": len(target),
        }


class ComputeSimilarityIndexOperation(TagOperation):
    """Compute dataset-level similarity index (brain run)."""
    name = "compute_similarity_index"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Run ``fob.compute_similarity`` with optional embeddings/backend settings."""
        _require_dataset_scope(params, self.name)
        target = context.dataset

        kwargs: dict[str, Any] = {}
        if params.get("embeddings_field"):
            kwargs["embeddings"] = params["embeddings_field"]
        if params.get("patches_field"):
            kwargs["patches_field"] = params["patches_field"]
        if params.get("roi_field"):
            kwargs["roi_field"] = params["roi_field"]
        if params.get("backend"):
            kwargs["backend"] = params["backend"]
        if params.get("brain_key"):
            kwargs["brain_key"] = params["brain_key"]

        result = fob.compute_similarity(target, **kwargs)
        key = params.get("brain_key") or getattr(result, "key", None) or getattr(result, "brain_key", None)

        return {
            "operation": self.name,
            "tag": tag,
            "scope": "dataset",
            "brain_key": key,
            "persisted": bool(key),
            "index_size": getattr(result, "index_size", None),
            "total_index_size": getattr(result, "total_index_size", None),
        }


class ComputeExactDuplicatesOperation(TagOperation):
    """Compute exact-duplicate summary statistics at dataset scope."""
    name = "compute_exact_duplicates"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Run exact duplicate detection and return aggregate counts."""
        _require_dataset_scope(params, self.name)
        result = fob.compute_exact_duplicates(context.dataset)

        if not isinstance(result, dict):
            return {
                "operation": self.name,
                "tag": tag,
                "scope": "dataset",
                "result_type": type(result).__name__,
            }

        duplicate_source_count = len(result)
        duplicate_sample_count = sum(len(dups) for dups in result.values())
        affected_ids = set(result.keys())
        for dups in result.values():
            affected_ids.update(dups)

        return {
            "operation": self.name,
            "tag": tag,
            "scope": "dataset",
            "duplicate_source_count": duplicate_source_count,
            "duplicate_sample_count": duplicate_sample_count,
            "affected_sample_count": len(affected_ids),
            "result_type": type(result).__name__,
        }


class ComputeNearDuplicatesOperation(TagOperation):
    """Compute near-duplicate relationships and summary counts."""
    name = "compute_near_duplicates"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Run near-duplicate search and summarize affected samples/pairs."""
        _require_dataset_scope(params, self.name)
        target = context.dataset

        threshold = float(params.get("threshold", 0.2))
        if threshold <= 0:
            raise ValueError("threshold must be > 0")

        kwargs: dict[str, Any] = {"threshold": threshold}
        if params.get("embeddings_field"):
            kwargs["embeddings"] = params["embeddings_field"]
        if params.get("roi_field"):
            kwargs["roi_field"] = params["roi_field"]

        result = fob.compute_near_duplicates(target, **kwargs)
        if hasattr(result, "find_duplicates"):
            result.find_duplicates(threshold)
        neighbors_map = getattr(result, "neighbors_map", None) or {}

        duplicate_source_count = len(neighbors_map)
        duplicate_pair_count = sum(len(neighbors) for neighbors in neighbors_map.values())
        affected_ids = set(neighbors_map.keys())
        for neighbors in neighbors_map.values():
            for neighbor in neighbors:
                if isinstance(neighbor, (tuple, list)) and neighbor:
                    affected_ids.add(str(neighbor[0]))
                else:
                    affected_ids.add(str(neighbor))

        return {
            "operation": self.name,
            "tag": tag,
            "scope": "dataset",
            "threshold": threshold,
            "duplicate_source_count": duplicate_source_count,
            "duplicate_pair_count": duplicate_pair_count,
            "affected_sample_count": len(affected_ids),
            "result_type": type(result).__name__,
        }


class ComputeLeakySplitsOperation(TagOperation):
    """Detect potential train/val/test leakage across configured splits."""
    name = "compute_leaky_splits"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Run leaky-split detection and return leakage summary payload."""
        _require_dataset_scope(params, self.name)
        target = context.dataset

        splits_raw = params.get("splits")
        if isinstance(splits_raw, str):
            splits = [part.strip() for part in splits_raw.split(",") if part.strip()]
        elif isinstance(splits_raw, list):
            splits = [str(part) for part in splits_raw if str(part).strip()]
        else:
            splits = []

        if not splits:
            raise ValueError("'splits' is required (list or comma-separated string)")

        threshold = float(params.get("threshold", 0.2))
        if threshold <= 0:
            raise ValueError("threshold must be > 0")

        kwargs: dict[str, Any] = {
            "splits": splits,
            "threshold": threshold,
        }
        if params.get("embeddings_field"):
            kwargs["embeddings"] = params["embeddings_field"]
        if params.get("roi_field"):
            kwargs["roi_field"] = params["roi_field"]

        result = fob.compute_leaky_splits(target, **kwargs)
        if hasattr(result, "find_leaks"):
            result.find_leaks(threshold)

        leak_ids = getattr(result, "leak_ids", []) or []
        split_views = getattr(result, "split_views", {}) or {}

        return {
            "operation": self.name,
            "tag": tag,
            "scope": "dataset",
            "splits": splits,
            "threshold": threshold,
            "leak_count": len(leak_ids),
            "sample_count": len(target),
            "split_count": len(split_views),
            "result_type": type(result).__name__,
        }


class ComputeAnomalyScoresOperation(TagOperation):
    """Compute anomaly scores via embedding-distance or anomalib artifact backend."""
    name = "compute_anomaly_scores"

    def execute(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None):
        """Dispatch anomaly scoring to selected backend and return backend payload.

        Backend options:
        - ``embedding_distance``: fit reference from embeddings and score distance
        - ``anomalib``: score with provided exported artifact
        """
        target, scope = _resolve_scope_collection(
            context=context,
            view=view,
            params=params,
            operation=self.name,
            default_scope="view",
        )
        if len(target) == 0:
            return {
                "operation": self.name,
                "tag": tag,
                "scope": scope,
                "sample_count": 0,
                "skipped": True,
            }

        backend = str(params.get("backend", "embedding_distance")).strip().lower()
        score_field = str(params.get("score_field", "anomaly_score"))
        flag_field = str(params.get("flag_field", "is_anomaly"))
        tag_filter = params.get("score_tag")
        if tag_filter is None and scope == "view":
            tag_filter = tag

        if backend == "anomalib":
            artifact = params.get("artifact") or params.get("artifact_path")
            if not artifact:
                raise ValueError(
                    "Anomalib backend requires 'artifact' (or 'artifact_path') "
                    "pointing to an artifact JSON or exported model file."
                )
            payload = score_with_anomalib(
                dataset_name=context.dataset_name,
                artifact_path=str(artifact),
                artifact_format=str(params["artifact_format"]) if params.get("artifact_format") else None,
                threshold=float(params.get("threshold", 0.5)),
                score_field=score_field,
                flag_field=flag_field,
                label_field=str(params["label_field"]) if params.get("label_field") else None,
                map_field=str(params["map_field"]) if params.get("map_field") else None,
                mask_field=str(params["mask_field"]) if params.get("mask_field") else None,
                tag_filter=tag_filter,
                device=str(params["device"]) if params.get("device") else None,
                trust_remote_code=bool(params.get("trust_remote_code", False)),
            )
            return {
                "operation": self.name,
                "tag": tag,
                "scope": scope,
                **payload,
            }

        if backend != "embedding_distance":
            raise ValueError(
                f"Unsupported anomaly backend '{backend}'. "
                "Use 'embedding_distance' or 'anomalib'"
            )

        embeddings_field = str(params.get("embeddings_field", "embeddings"))
        reference = fit_embedding_distance_reference(
            dataset_name=context.dataset_name,
            embeddings_field=embeddings_field,
            normal_tag=params.get("normal_tag"),
            threshold=float(params["threshold"]) if params.get("threshold") is not None else None,
            threshold_quantile=float(params.get("threshold_quantile", 0.95)),
        )
        payload = score_with_embedding_distance(
            dataset_name=context.dataset_name,
            reference=reference,
            score_field=score_field,
            flag_field=flag_field,
            tag_filter=tag_filter,
        )
        return {
            "operation": self.name,
            "tag": tag,
            "scope": scope,
            **payload,
            "reference": reference.to_dict(),
        }
