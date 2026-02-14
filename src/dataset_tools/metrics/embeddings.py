"""Implementation module for metric computation.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

import fiftyone.brain as fob  # type: ignore

from dataset_tools.metrics.base import BaseMetricComputation
from dataset_tools.models import load_model
from dataset_tools.models.providers.huggingface import HuggingFaceEmbeddingModel


class EmbeddingsComputation(BaseMetricComputation):
    """EmbeddingsComputation used by metric computation.
    """
    def __init__(
        self,
        dataset_name: str,
        model_name: str = "facebook/dinov2-base",
        model_ref: str | None = None,
        embeddings_field: str = "embeddings",
        patches_field: str | None = None,
        use_umap: bool = True,
        use_cluster: bool = True,
        n_clusters: int = 10,
    ):
        """Initialize `EmbeddingsComputation` with runtime parameters.

Args:
    dataset_name: Name of the FiftyOne dataset to operate on.
    model_name: Model identifier used by the selected backend.
    model_ref: Provider-qualified model reference (for example `hf:...`, `foz:...`, `anomalib:...`).
    embeddings_field: Field containing embeddings vectors.
    patches_field: Value controlling patches field for this routine.
    use_umap: Value controlling use umap for this routine.
    use_cluster: Value controlling use cluster for this routine.
    n_clusters: Value controlling n clusters for this routine.

Returns:
    None.
        """
        super().__init__(dataset_name)
        self.model_name = model_name
        self.model_ref = model_ref
        self.embeddings_field = embeddings_field
        self.patches_field = patches_field
        self.use_umap = use_umap
        self.use_cluster = use_cluster
        self.n_clusters = n_clusters

    def compute(self, dataset):
        """Perform compute.

Args:
    dataset: FiftyOne dataset or dataset-like collection used by this operation.

Returns:
    Result object consumed by the caller or downstream workflow.
        """
        raw_model_ref = self.model_ref or self.model_name
        loaded = load_model(
            raw_model_ref,
            default_provider="hf",
            task="embeddings",
            capability="embeddings",
        )
        model = loaded.model

        if self.patches_field:
            dataset.compute_patch_embeddings(
                model,
                self.patches_field,
                embeddings_field=self.embeddings_field,
                force_square=True,
                progress=True,
            )
        else:
            dataset.compute_embeddings(
                model,
                embeddings_field=self.embeddings_field,
                progress=True,
            )

        if self.use_umap:
            vis_key = f"{self.embeddings_field}_umap"
            view = dataset
            actual_emb_field = self.embeddings_field
            if self.patches_field:
                view = dataset.to_patches(self.patches_field)
                actual_emb_field = f"{self.patches_field}.{self.embeddings_field}"

            embeddings = view.values(actual_emb_field)
            sample_ids = view.values("id")
            valid_indices = [idx for idx, emb in enumerate(embeddings) if emb is not None]
            if valid_indices:
                matrix = np.stack([embeddings[idx] for idx in valid_indices])
                valid_ids = [sample_ids[idx] for idx in valid_indices]
                fob.compute_visualization(
                    view.select(valid_ids),
                    embeddings=matrix,
                    method="umap",
                    brain_key=vis_key,
                )

        if self.use_cluster:
            cluster_field = f"{self.embeddings_field}_cluster"
            view = dataset
            actual_emb_field = self.embeddings_field
            if self.patches_field:
                view = dataset.to_patches(self.patches_field)
                actual_emb_field = f"{self.patches_field}.{self.embeddings_field}"

            embeddings = view.values(actual_emb_field)
            sample_ids = view.values("id")
            valid_indices = [idx for idx, emb in enumerate(embeddings) if emb is not None]
            if valid_indices:
                matrix = np.stack([embeddings[idx] for idx in valid_indices])
                valid_ids = [sample_ids[idx] for idx in valid_indices]
                cluster_count = min(self.n_clusters, len(matrix))
                labels = KMeans(n_clusters=cluster_count, n_init="auto", random_state=42).fit_predict(matrix)
                label_map = {sample_id: str(label) for sample_id, label in zip(valid_ids, labels)}
                view.set_values(cluster_field, label_map, key_field="id")

        return {
            "dataset": self.dataset_name,
            "field": self.embeddings_field,
            "model_ref": f"{loaded.ref.provider}:{loaded.ref.model_id}",
            "model_provider": loaded.ref.provider,
        }
