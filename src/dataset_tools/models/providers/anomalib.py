"""Implementation module for model provider registry.
"""
from __future__ import annotations

from typing import Any

from dataset_tools.models.base import ModelProvider
from dataset_tools.models.spec import LoadedModel, ModelRef

DEFAULT_ANOMALIB_MODELS = (
    "padim",
    "patchcore",
    "reverse_distillation",
    "stfpm",
)


class AnomalibProvider(ModelProvider):
    """Model provider adapter that resolves and loads backend-specific models.
    """
    name = "anomalib"

    def _import_anomalib(self):
        """Internal helper for import anomalib.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        try:
            import anomalib  # type: ignore

            return anomalib
        except ImportError as e:
            raise RuntimeError(
                "anomalib is not installed. Install optional dependency "
                "with: pip install anomalib"
            ) from e

    def load(self, model_ref: ModelRef, *, task: str | None = None, **kwargs: Any) -> LoadedModel:
        """Perform load.

Args:
    model_ref: Provider-qualified model reference (for example `hf:...`, `foz:...`, `anomalib:...`).
    task: Value controlling task for this routine.
    **kwargs: Additional keyword arguments forwarded to downstream APIs.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        resolved_task = (task or "anomaly").strip().lower()
        if resolved_task not in {"anomaly", "anomaly_detection"}:
            raise RuntimeError(
                "Anomalib provider supports anomaly-detection tasks only. "
                f"Received task='{task}'."
            )

        anomalib = self._import_anomalib()
        model_name = model_ref.model_id.strip()
        if not model_name:
            raise ValueError("Anomalib model id cannot be empty")

        model_instance = None
        load_errors: list[str] = []

        # Newer API patterns
        try:
            from anomalib.models import get_model  # type: ignore

            model_instance = get_model({"name": model_name})
        except Exception as e:
            load_errors.append(f"get_model: {e}")

        # Fallback class lookup for older APIs
        if model_instance is None:
            try:
                import anomalib.models as anomalib_models  # type: ignore

                class_name = "".join(part.capitalize() for part in model_name.replace("-", "_").split("_"))
                cls = getattr(anomalib_models, class_name)
                model_instance = cls(**kwargs)
            except Exception as e:
                load_errors.append(f"class_lookup: {e}")

        if model_instance is None:
            joined = "; ".join(load_errors) if load_errors else "unknown error"
            raise RuntimeError(
                f"Failed to load anomalib model '{model_name}'. "
                f"Detected anomalib module: {anomalib.__name__}. Details: {joined}"
            )

        return LoadedModel(
            ref=model_ref,
            model=model_instance,
            capabilities=("anomaly",),
            metadata={
                "task": "anomaly",
                "model_type": type(model_instance).__name__,
            },
        )

    def list_models(self, contains: str | None = None, limit: int | None = None) -> list[str]:
        """List available models.

Args:
    contains: Value controlling contains for this routine.
    limit: Value controlling limit for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        models = list(DEFAULT_ANOMALIB_MODELS)
        if contains:
            needle = contains.lower()
            models = [name for name in models if needle in name.lower()]
        if limit is not None:
            models = models[: max(0, int(limit))]
        return models
