# Advanced Analysis Research (Consolidated)

This document consolidates the relevant and non-stale parts of previous FiftyOne research notes into a form aligned with the current `dataset_tools` architecture.

This document consolidates prior internal research notes into a standalone context reference so the future extracted `dst` repository does not depend on legacy docs paths.

## 1. Embeddings as the Foundation

Key idea: most curation analyses (uniqueness, similarity, clustering, anomaly heuristics) depend on good embeddings.

Recommended default model families:

- DINOv2 (`hf:facebook/dinov2-base`) for robust general-purpose image embeddings
- DINOv3 where access and compatibility are available
- CLIP for multimodal/text-image retrieval workflows

`dst` implementation mapping:

- `dst metrics embeddings --model-ref <provider:model>`
- `dst models list|validate|resolve` for provider/model introspection

## 2. Dimensionality Reduction and Cluster Exploration

Methods:

- UMAP: default for visual structure exploration
- t-SNE: small datasets/local neighborhoods
- PCA: fast baseline and deterministic projection

`dst` implementation mapping:

- `dst brain visualization --method umap|tsne|pca`

## 3. Duplicate Control and Split Integrity

Research conclusion: duplicate handling and split leakage checks are high-impact dataset quality controls.

`dst` implementation mapping:

- exact duplicates: `dst brain duplicates exact`
- near duplicates: `dst brain duplicates near`
- leakage across splits: `dst brain leaky-splits --splits train,val,test`

## 4. Label Quality Signals

Signals:

- Mistakenness (prediction vs ground truth disagreement)
- Hardness (sample difficulty signal)
- Representativeness (how central/typical a sample is)
- Uniqueness (diversity/outlier signal)

`dst` implementation mapping:

- `dst metrics mistakenness`
- `dst metrics hardness`
- `dst metrics representativeness`
- `dst metrics uniqueness`

## 5. Anomaly Detection

Two practical paths:

- embedding-distance baseline (lightweight and quick)
- anomalib-backed train/export/score flow for richer anomaly modeling

`dst` implementation mapping:

- `dst anomaly run --backend embedding_distance`
- `dst anomaly train` + `dst anomaly score --backend anomalib`

## 6. Architecture Guidance

Research-driven design constraints that are now part of `dataset_tools`:

1. keep reusable analysis logic in `dataset_tools`, not dataset-specific pipelines
2. expose operations via one CLI surface (`dst`)
3. verify against real FiftyOne API signatures before large feature additions
4. preserve proven LS batched sync behavior during refactors

## 7. External Reference Links

- FiftyOne Brain overview: https://docs.voxel51.com/brain.html
- Image embeddings tutorial: https://docs.voxel51.com/tutorials/image_embeddings.html
- Dimension reduction tutorial: https://docs.voxel51.com/tutorials/dimension_reduction.html
- Clustering tutorial: https://docs.voxel51.com/tutorials/clustering.html
- Uniqueness tutorial: https://docs.voxel51.com/tutorials/uniqueness.html
- Detection mistakes tutorial: https://docs.voxel51.com/tutorials/detection_mistakes.html
- Anomaly detection tutorial: https://docs.voxel51.com/tutorials/anomaly_detection.html
- DINOv3 tutorial: https://docs.voxel51.com/tutorials/dinov3.html
- Anomalib repo: https://github.com/openvinotoolkit/anomalib

## 8. Practical Recommendation for New Datasets

1. Load dataset via `dst data load ...`.
2. Compute embeddings.
3. Run uniqueness/duplicates/leaky-splits.
4. Run mistakenness/hardness/representativeness where labels/predictions exist.
5. Use tag workflow + Label Studio roundtrip for human correction.
6. Sync to disk only after dry-run review.
