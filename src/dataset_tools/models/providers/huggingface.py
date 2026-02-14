"""Implementation module for model provider registry.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import fiftyone.core.models as fom  # type: ignore

from dataset_tools.models.base import ModelProvider
from dataset_tools.models.spec import LoadedModel, ModelRef


class HuggingFaceEmbeddingModel(fom.Model, fom.EmbeddingsMixin):
    """HuggingFaceEmbeddingModel used by model provider registry.
    """
    def __init__(self, model_name: str):
        """Initialize `HuggingFaceEmbeddingModel` with runtime parameters.

Args:
    model_name: Model identifier used by the selected backend.

Returns:
    None.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @property
    def media_type(self):
        """Perform media type.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return "image"

    @property
    def has_embeddings(self):
        """Perform has embeddings.

Returns:
    Boolean indicating the evaluated condition.
        """
        return True

    def embed(self, arg):
        """Perform embed.

Args:
    arg: Value controlling arg for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        if isinstance(arg, str):
            image = Image.open(arg).convert("RGB")
        elif isinstance(arg, np.ndarray):
            image = Image.fromarray(arg)
        else:
            image = arg

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs[0].mean(dim=1)

        return embeddings.cpu().numpy()[0]

    def embed_all(self, args):
        """Perform embed all.

Args:
    args: Value controlling args for this routine.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        return np.stack([self.embed(arg) for arg in args])


class HuggingFaceProvider(ModelProvider):
    """Model provider adapter that resolves and loads backend-specific models.
    """
    name = "hf"

    def load(self, model_ref: ModelRef, *, task: str | None = None, **kwargs: Any) -> LoadedModel:
        """Perform load.

Args:
    model_ref: Provider-qualified model reference (for example `hf:...`, `foz:...`, `anomalib:...`).
    task: Value controlling task for this routine.
    **kwargs: Additional keyword arguments forwarded to downstream APIs.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        resolved_task = (task or "embeddings").strip().lower()
        if resolved_task not in {"embeddings", "embedding"}:
            raise RuntimeError(
                "HuggingFace provider currently supports task='embeddings' only. "
                f"Received task='{task}'."
            )

        model = HuggingFaceEmbeddingModel(model_ref.model_id)
        return LoadedModel(
            ref=model_ref,
            model=model,
            capabilities=("embeddings",),
            metadata={"task": "embeddings"},
        )
