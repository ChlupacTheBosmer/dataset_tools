# FiftyOne Reference Hub

This is the canonical external reference index for FiftyOne when working on `dataset_tools`.

Use this file instead of local cloned-doc paths.

## Official Documentation Entry Points

- Docs home: https://docs.voxel51.com/
- Python API reference: https://docs.voxel51.com/api/fiftyone.html
- User guide: https://docs.voxel51.com/user_guide/index.html
- Brain overview: https://docs.voxel51.com/brain.html
- Integrations overview: https://docs.voxel51.com/integrations/index.html

## High-Priority Pages For `dataset_tools`

### Dataset and App fundamentals

- Dataset creation/loading: https://docs.voxel51.com/user_guide/dataset_creation/index.html
- App usage: https://docs.voxel51.com/user_guide/app.html
- Views and filtering: https://docs.voxel51.com/user_guide/using_views.html

### Brain operations used by `dst`

- Visualization: https://docs.voxel51.com/brain.html#visualizing-embeddings
- Similarity: https://docs.voxel51.com/brain.html#similarity
- Uniqueness: https://docs.voxel51.com/brain.html#image-uniqueness
- Mistakenness: https://docs.voxel51.com/brain.html#label-mistakes
- Hardness: https://docs.voxel51.com/brain.html#sample-hardness
- Representativeness: https://docs.voxel51.com/brain.html#image-representativeness
- Near duplicates: https://docs.voxel51.com/brain.html#near-duplicates
- Exact duplicates: https://docs.voxel51.com/brain.html#exact-duplicates
- Leaky splits: https://docs.voxel51.com/brain.html#leaky-splits

### Model and zoo references

- Model zoo: https://docs.voxel51.com/user_guide/model_zoo/index.html
- Model/dataset zoo quick exploration: https://docs.voxel51.com/getting_started/model_dataset_zoo/02_explore.html

### Tutorials aligned with implemented `dst` features

- Image embeddings: https://docs.voxel51.com/tutorials/image_embeddings.html
- Dimension reduction: https://docs.voxel51.com/tutorials/dimension_reduction.html
- Clustering: https://docs.voxel51.com/tutorials/clustering.html
- Uniqueness: https://docs.voxel51.com/tutorials/uniqueness.html
- Detection mistakes: https://docs.voxel51.com/tutorials/detection_mistakes.html
- DINOv3 visual search: https://docs.voxel51.com/tutorials/dinov3.html
- Anomaly detection: https://docs.voxel51.com/tutorials/anomaly_detection.html

## How This Maps To `dst`

- `dst data ...`: dataset loading and field setup
- `dst metrics ...`: field-populating Brain computations
- `dst brain ...`: Brain runs/indexes (visualization, similarity, duplicates, leaks)
- `dst app open`: App session launch for human-in-the-loop curation

## Agent Notes

When updating code that calls `fiftyone.brain`, always validate function signatures against the installed version before implementation changes. The contract checks in `tests/test_fiftyone_contracts.py` are the internal guardrail for this.
