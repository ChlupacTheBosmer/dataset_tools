# Dataset Tools API Reference (Complete)

This reference is generated from the current `dataset_tools` source tree and is intended as an exhaustive API map.

- Scope: every Python module, top-level function, class, and class method in `dataset_tools/`
- Stability: names marked "Internal helper" are implementation details and can change without notice
- Narrative: descriptions are sourced from in-code docstrings for maintainable docs-as-code

## Package Inventory

| Module | Role | Module Summary |
|---|---|---|
| `src.dataset_tools` | Internal module in dataset_tools. See source and call graph for behavior. | Public exports for the dataset_tools package. |
| `src.dataset_tools.anomaly` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.anomaly`. |
| `src.dataset_tools.anomaly.anomalib` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for anomaly analysis. |
| `src.dataset_tools.anomaly.base` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for anomaly analysis. |
| `src.dataset_tools.anomaly.pipeline` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for anomaly analysis. |
| `src.dataset_tools.brain` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.brain`. |
| `src.dataset_tools.brain.base` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for FiftyOne Brain analysis. |
| `src.dataset_tools.brain.duplicates` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for FiftyOne Brain analysis. |
| `src.dataset_tools.brain.leaky_splits` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for FiftyOne Brain analysis. |
| `src.dataset_tools.brain.similarity` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for FiftyOne Brain analysis. |
| `src.dataset_tools.brain.visualization` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for FiftyOne Brain analysis. |
| `src.dataset_tools.config` | Internal module in dataset_tools. See source and call graph for behavior. | Central configuration schema and loader for ``dataset_tools``. |
| `src.dataset_tools.debug` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.debug`. |
| `src.dataset_tools.debug.debug_fo_import` | Debug/support script; not intended as stable production API. | Implementation module for debug and diagnostics. |
| `src.dataset_tools.debug.debug_fob` | Debug/support script; not intended as stable production API. | Implementation module for debug and diagnostics. |
| `src.dataset_tools.debug.debug_imports` | Debug/support script; not intended as stable production API. | Implementation module for debug and diagnostics. |
| `src.dataset_tools.debug.debug_ls_tasks` | Debug/support script; not intended as stable production API. | Implementation module for debug and diagnostics. |
| `src.dataset_tools.debug.debug_ls_urls` | Debug/support script; not intended as stable production API. | Implementation module for debug and diagnostics. |
| `src.dataset_tools.dst` | Internal module in dataset_tools. See source and call graph for behavior. | Dataset Tools unified CLI entrypoint (`dst`). |
| `src.dataset_tools.label_studio` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.label_studio`. |
| `src.dataset_tools.label_studio.client` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for Label Studio integration. |
| `src.dataset_tools.label_studio.storage` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for Label Studio integration. |
| `src.dataset_tools.label_studio.sync` | Internal module in dataset_tools. See source and call graph for behavior. | Push/pull synchronization helpers between FiftyOne and Label Studio. |
| `src.dataset_tools.label_studio.translator` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for Label Studio integration. |
| `src.dataset_tools.label_studio.uploader` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for Label Studio integration. |
| `src.dataset_tools.label_studio_json` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for Label Studio integration. |
| `src.dataset_tools.loader` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for dataset loading. |
| `src.dataset_tools.loaders` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.loaders`. |
| `src.dataset_tools.loaders.base` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for dataset loading. |
| `src.dataset_tools.loaders.coco` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for dataset loading. |
| `src.dataset_tools.loaders.path_resolvers` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for dataset loading. |
| `src.dataset_tools.loaders.yolo` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for dataset loading. |
| `src.dataset_tools.metrics` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.metrics`. |
| `src.dataset_tools.metrics.base` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for metric computation. |
| `src.dataset_tools.metrics.embeddings` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for metric computation. |
| `src.dataset_tools.metrics.field_metric` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for metric computation. |
| `src.dataset_tools.metrics.hardness` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for metric computation. |
| `src.dataset_tools.metrics.mistakenness` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for metric computation. |
| `src.dataset_tools.metrics.representativeness` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for metric computation. |
| `src.dataset_tools.metrics.uniqueness` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for metric computation. |
| `src.dataset_tools.models` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.models`. |
| `src.dataset_tools.models.base` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for model provider registry. |
| `src.dataset_tools.models.providers` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.models.providers`. |
| `src.dataset_tools.models.providers.anomalib` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for model provider registry. |
| `src.dataset_tools.models.providers.fiftyone_zoo` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for model provider registry. |
| `src.dataset_tools.models.providers.huggingface` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for model provider registry. |
| `src.dataset_tools.models.registry` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for model provider registry. |
| `src.dataset_tools.models.spec` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for model provider registry. |
| `src.dataset_tools.sync_from_fo_to_disk` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for dataset tools runtime. |
| `src.dataset_tools.tag_workflow` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.tag_workflow`. |
| `src.dataset_tools.tag_workflow.config` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for tag workflow execution. |
| `src.dataset_tools.tag_workflow.context` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for tag workflow execution. |
| `src.dataset_tools.tag_workflow.engine` | Internal module in dataset_tools. See source and call graph for behavior. | Implementation module for tag workflow execution. |
| `src.dataset_tools.tag_workflow.operations` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.tag_workflow.operations`. |
| `src.dataset_tools.tag_workflow.operations.analysis` | Internal module in dataset_tools. See source and call graph for behavior. | Analysis-focused tag-workflow operations. |
| `src.dataset_tools.tag_workflow.operations.base` | Internal module in dataset_tools. See source and call graph for behavior. | Base operation contract for tag-workflow actions. |
| `src.dataset_tools.tag_workflow.operations.core` | Internal module in dataset_tools. See source and call graph for behavior. | Core tag-workflow operations for mutation, LS sync, and disk sync. |
| `src.dataset_tools.tools` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.tools`. |
| `src.dataset_tools.workflows` | Internal module in dataset_tools. See source and call graph for behavior. | Package initializer for `dataset_tools.workflows`. |
| `src.dataset_tools.workflows.roundtrip` | Internal module in dataset_tools. See source and call graph for behavior. | High-level orchestration for FiftyOne <-> Label Studio curation roundtrip. |

## `src.dataset_tools`

- File: `src/dataset_tools/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Public exports for the dataset_tools package.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.anomaly`

- File: `src/dataset_tools/anomaly/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.anomaly`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.anomaly.anomalib`

- File: `src/dataset_tools/anomaly/anomalib.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for anomaly analysis.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `PreparedAnomalibDataset` | 24 | Public callable | PreparedAnomalibDataset used by anomaly analysis. |
| `AnomalibArtifact` | 63 | Public callable | AnomalibArtifact used by anomaly analysis. |

#### `PreparedAnomalibDataset` methods
- Class Summary: PreparedAnomalibDataset used by anomaly analysis.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `to_dict` | `(self) -> dict[str, Any]` | 40 | Public callable | Perform to dict. |

#### `AnomalibArtifact` methods
- Class Summary: AnomalibArtifact used by anomaly analysis.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `to_dict` | `(self) -> dict[str, Any]` | 75 | Public callable | Perform to dict. |
| `from_dict` | `(payload: dict[str, Any]) -> 'AnomalibArtifact'` | 93 | Public callable | Perform from dict. |

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_load_dataset` | `(dataset_name: str)` | 139 | Internal helper | Load dataset required by this module. |
| `_resolve_view` | `(dataset, tag_filter: str | None)` | 153 | Internal helper | Resolve view from provided inputs. |
| `_safe_name` | `(value: str) -> str` | 168 | Internal helper | Internal helper for safe name. |
| `_clear_dir` | `(path: Path, overwrite: bool)` | 181 | Internal helper | Internal helper for clear dir. |
| `_link_or_copy` | `(src: Path, dst: Path, *, symlink: bool)` | 201 | Internal helper | Internal helper for link or copy. |
| `_sample_target_name` | `(sample_id: str, filepath: str) -> str` | 221 | Internal helper | Internal helper for sample target name. |
| `_extract_mask_path` | `(sample: Any, mask_field: str) -> str | None` | 237 | Internal helper | Internal helper for extract mask path. |
| `prepare_anomalib_folder_dataset` | `(dataset_name: str, *, output_root: str | Path, normal_tag: str | None = None, abnormal_tag: str | None = None, mask_field: str | None = None, symlink: bool = True, overwrite_data: bool = False) -> PreparedAnomalibDataset` | 269 | Public callable | Perform prepare anomalib folder dataset. |
| `_import_anomalib_components` | `()` | 364 | Internal helper | Internal helper for import anomalib components. |
| `_parse_image_size` | `(value: str | None) -> tuple[int, int] | None` | 382 | Internal helper | Parse and validate image size input values. |
| `_create_datamodule` | `(prepared: PreparedAnomalibDataset, *, datamodule_name: str, image_size: tuple[int, int] | None = None, train_batch_size: int = 8, eval_batch_size: int = 8, num_workers: int = 0, normal_split_ratio: float = 0.2, test_split_mode: str = 'from_dir', test_split_ratio: float = 0.2, val_split_mode: str = 'same_as_test', val_split_ratio: float = 0.5, seed: int | None = None)` | 403 | Internal helper | Internal helper for create datamodule. |
| `save_anomalib_artifact` | `(path: str | Path, artifact: AnomalibArtifact)` | 469 | Public callable | Save anomalib artifact to persistent storage. |
| `load_anomalib_artifact` | `(path: str | Path) -> AnomalibArtifact` | 485 | Public callable | Load anomalib artifact required by this module. |
| `train_and_export_anomalib` | `(dataset_name: str, *, model_ref: str = 'anomalib:padim', normal_tag: str | None = None, abnormal_tag: str | None = None, mask_field: str | None = None, artifact_dir: str | Path | None = None, data_dir: str | Path | None = None, artifact_format: str = 'openvino', image_size: str | None = None, train_batch_size: int = 8, eval_batch_size: int = 8, num_workers: int = 0, normal_split_ratio: float = 0.2, test_split_mode: str = 'from_dir', test_split_ratio: float = 0.2, val_split_mode: str = 'same_as_test', val_split_ratio: float = 0.5, seed: int | None = None, max_epochs: int | None = None, accelerator: str | None = None, devices: str | int | None = None, symlink: bool = True, overwrite_data: bool = False, artifact_json: str | Path | None = None) -> AnomalibArtifact` | 500 | Public callable | Perform train and export anomalib. |
| `_extract_prediction_fields` | `(prediction: Any) -> tuple[float, bool, np.ndarray | None, np.ndarray | None]` | 684 | Internal helper | Internal helper for extract prediction fields. |
| `_to_numpy` | `(value: Any) -> np.ndarray` | 744 | Internal helper | Internal helper for to numpy. |
| `_assert_trusted_torch_loading` | `(trust_remote_code: bool)` | 772 | Internal helper | Internal helper for assert trusted torch loading. |
| `_infer_export_type` | `(path: Path) -> str` | 795 | Internal helper | Internal helper for infer export type. |
| `_resolve_anomalib_artifact` | `(artifact: str | Path, *, artifact_format: str | None = None) -> AnomalibArtifact` | 815 | Internal helper | Resolve anomalib artifact from provided inputs. |
| `_build_inferencer` | `(artifact: AnomalibArtifact, *, device: str | None = None, trust_remote_code: bool = False)` | 847 | Internal helper | Build inferencer for downstream steps. |
| `_score_with_engine_predict` | `(*, artifact: AnomalibArtifact, sample_ids: list[str], filepaths: list[str], threshold: float, device: str | None, trust_remote_code: bool) -> tuple[dict[str, float], dict[str, bool], dict[str, np.ndarray], dict[str, np.ndarray]]` | 883 | Internal helper | Internal helper for score with engine predict. |
| `score_with_anomalib_artifact` | `(dataset_name: str, *, artifact: str | Path, artifact_format: str | None = None, threshold: float = 0.5, score_field: str = 'anomaly_score', flag_field: str = 'is_anomaly', label_field: str | None = None, map_field: str | None = None, mask_field: str | None = None, tag_filter: str | None = None, device: str | None = None, trust_remote_code: bool = False) -> dict[str, Any]` | 989 | Public callable | Perform score with anomalib artifact. |

## `src.dataset_tools.anomaly.base`

- File: `src/dataset_tools/anomaly/base.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for anomaly analysis.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `AnomalyReference` | 10 | Public callable | AnomalyReference used by anomaly analysis. |

#### `AnomalyReference` methods
- Class Summary: AnomalyReference used by anomaly analysis.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `to_dict` | `(self) -> dict[str, Any]` | 19 | Public callable | Perform to dict. |
| `from_dict` | `(payload: dict[str, Any]) -> 'AnomalyReference'` | 34 | Public callable | Perform from dict. |

## `src.dataset_tools.anomaly.pipeline`

- File: `src/dataset_tools/anomaly/pipeline.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for anomaly analysis.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_load_dataset` | `(dataset_name: str)` | 17 | Internal helper | Load dataset required by this module. |
| `_resolve_view` | `(dataset, tag_filter: str | None)` | 31 | Internal helper | Resolve view from provided inputs. |
| `_read_embeddings` | `(view, embeddings_field: str) -> tuple[list[str], list[np.ndarray]]` | 46 | Internal helper | Internal helper for read embeddings. |
| `fit_embedding_distance_reference` | `(dataset_name: str, *, embeddings_field: str = 'embeddings', normal_tag: str | None = None, threshold: float | None = None, threshold_quantile: float = 0.95) -> AnomalyReference` | 71 | Public callable | Perform fit embedding distance reference. |
| `score_with_embedding_distance` | `(dataset_name: str, *, reference: AnomalyReference, score_field: str = 'anomaly_score', flag_field: str = 'is_anomaly', tag_filter: str | None = None) -> dict[str, Any]` | 119 | Public callable | Perform score with embedding distance. |
| `score_with_anomalib` | `(dataset_name: str, *, artifact_path: str, artifact_format: str | None = None, threshold: float = 0.5, score_field: str = 'anomaly_score', flag_field: str = 'is_anomaly', label_field: str | None = None, map_field: str | None = None, mask_field: str | None = None, tag_filter: str | None = None, device: str | None = None, trust_remote_code: bool = False) -> dict[str, Any]` | 175 | Public callable | Perform score with anomalib. |
| `save_reference` | `(path: str | Path, reference: AnomalyReference)` | 225 | Public callable | Save reference to persistent storage. |
| `load_reference` | `(path: str | Path) -> AnomalyReference` | 241 | Public callable | Load reference required by this module. |
| `run_embedding_distance` | `(dataset_name: str, *, embeddings_field: str = 'embeddings', normal_tag: str | None = None, score_tag: str | None = None, score_field: str = 'anomaly_score', flag_field: str = 'is_anomaly', threshold: float | None = None, threshold_quantile: float = 0.95, reference_path: str | None = None) -> dict[str, Any]` | 256 | Public callable | Run embedding distance and return execution results. |

## `src.dataset_tools.brain`

- File: `src/dataset_tools/brain/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.brain`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.brain.base`

- File: `src/dataset_tools/brain/base.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for FiftyOne Brain analysis.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `BrainOperation` | 11 | Public callable | Operation class used in FiftyOne Brain analysis. |

#### `BrainOperation` methods
- Class Summary: Operation class used in FiftyOne Brain analysis.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, brain_key: str | None = None)` | 15 | Internal helper | Initialize `BrainOperation` with runtime parameters. |
| `load_dataset` | `(self)` | 28 | Public callable | Load dataset required by this module. |
| `run` | `(self) -> dict[str, Any]` | 38 | Public callable | Run the operation and return execution results. |
| `ensure_brain_run_exists` | `(self, dataset, brain_key: str)` | 52 | Integration/bootstrap API | Ensure brain run exists exists and return it. |
| `_normalize_ids` | `(values) -> list[str]` | 68 | Internal helper | Internal helper for normalize ids. |
| `execute` | `(self, dataset) -> dict[str, Any]` | 80 | Public callable | Perform execute. |

## `src.dataset_tools.brain.duplicates`

- File: `src/dataset_tools/brain/duplicates.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for FiftyOne Brain analysis.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `ExactDuplicatesOperation` | 12 | Public callable | Operation class used in FiftyOne Brain analysis. |
| `NearDuplicatesOperation` | 46 | Public callable | Operation class used in FiftyOne Brain analysis. |

#### `ExactDuplicatesOperation` methods
- Class Summary: Operation class used in FiftyOne Brain analysis.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, dataset) -> dict[str, Any]` | 15 | Public callable | Perform execute. |

#### `NearDuplicatesOperation` methods
- Class Summary: Operation class used in FiftyOne Brain analysis.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, threshold: float = 0.2, embeddings: str | None = None, roi_field: str | None = None)` | 49 | Internal helper | Initialize `NearDuplicatesOperation` with runtime parameters. |
| `execute` | `(self, dataset) -> dict[str, Any]` | 72 | Public callable | Perform execute. |

## `src.dataset_tools.brain.leaky_splits`

- File: `src/dataset_tools/brain/leaky_splits.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for FiftyOne Brain analysis.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `LeakySplitsOperation` | 12 | Public callable | Operation class used in FiftyOne Brain analysis. |

#### `LeakySplitsOperation` methods
- Class Summary: Operation class used in FiftyOne Brain analysis.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, splits: list[str], threshold: float = 0.2, embeddings: str | None = None, roi_field: str | None = None)` | 15 | Internal helper | Initialize `LeakySplitsOperation` with runtime parameters. |
| `execute` | `(self, dataset) -> dict[str, Any]` | 41 | Public callable | Perform execute. |

## `src.dataset_tools.brain.similarity`

- File: `src/dataset_tools/brain/similarity.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for FiftyOne Brain analysis.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `SimilarityOperation` | 12 | Public callable | Operation class used in FiftyOne Brain analysis. |

#### `SimilarityOperation` methods
- Class Summary: Operation class used in FiftyOne Brain analysis.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, embeddings: str | None = None, patches_field: str | None = None, roi_field: str | None = None, backend: str | None = None, brain_key: str | None = None)` | 15 | Internal helper | Initialize `SimilarityOperation` with runtime parameters. |
| `execute` | `(self, dataset) -> dict[str, Any]` | 43 | Public callable | Perform execute. |

## `src.dataset_tools.brain.visualization`

- File: `src/dataset_tools/brain/visualization.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for FiftyOne Brain analysis.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `VisualizationOperation` | 12 | Public callable | Operation class used in FiftyOne Brain analysis. |

#### `VisualizationOperation` methods
- Class Summary: Operation class used in FiftyOne Brain analysis.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, method: str = 'umap', num_dims: int = 2, embeddings: str | None = None, patches_field: str | None = None, brain_key: str | None = None)` | 15 | Internal helper | Initialize `VisualizationOperation` with runtime parameters. |
| `execute` | `(self, dataset) -> dict[str, Any]` | 43 | Public callable | Perform execute. |

## `src.dataset_tools.config`

- File: `src/dataset_tools/config.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Central configuration schema and loader for ``dataset_tools``.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `PathMountConfig` | 23 | Public callable | Host/container mount mapping used for Label Studio local-files URLs. |
| `LabelStudioConfig` | 36 | Public callable | Runtime settings for Label Studio connectivity and task transfer. |
| `DatasetConfig` | 54 | Public callable | Dataset-level defaults used by curation and sync workflows. |
| `DiskSyncConfig` | 68 | Public callable | Filesystem sync defaults for writing corrections back to label files. |
| `AppConfig` | 78 | Public callable | Top-level immutable runtime configuration consumed by dataset_tools. |

#### `PathMountConfig` methods
- Class Summary: Host/container mount mapping used for Label Studio local-files URLs.
- No methods declared in this class body.

#### `LabelStudioConfig` methods
- Class Summary: Runtime settings for Label Studio connectivity and task transfer.
- No methods declared in this class body.

#### `DatasetConfig` methods
- Class Summary: Dataset-level defaults used by curation and sync workflows.
- No methods declared in this class body.

#### `DiskSyncConfig` methods
- Class Summary: Filesystem sync defaults for writing corrections back to label files.
- No methods declared in this class body.

#### `AppConfig` methods
- Class Summary: Top-level immutable runtime configuration consumed by dataset_tools.
- No methods declared in this class body.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_default_config_dict` | `() -> dict[str, Any]` | 86 | Internal helper | Return the baseline config dictionary used before any overrides. |
| `resolve_default_local_config_path` | `() -> Path` | 119 | Parsing/resolution utility | Resolve default local config path in a portable, install-friendly way. |
| `_deep_merge` | `(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]` | 139 | Internal helper | Recursively merge ``override`` into ``base`` and return a new dict. |
| `_load_local_config` | `(path: Path) -> dict[str, Any]` | 161 | Internal helper | Load optional local JSON config from ``path``. |
| `_load_env_config` | `() -> dict[str, Any]` | 177 | Internal helper | Build config overrides from supported environment variables. |
| `load_config` | `(local_config_path: str | os.PathLike[str] | None = None, overrides: dict[str, Any] | None = None) -> AppConfig` | 216 | Public callable | Resolve and validate the runtime ``AppConfig``. |
| `require_label_studio_api_key` | `(config: AppConfig) -> str` | 254 | Public callable | Return Label Studio API key or raise a clear configuration error. |

## `src.dataset_tools.debug`

- File: `src/dataset_tools/debug/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.debug`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.debug.debug_fo_import`

- File: `src/dataset_tools/debug/debug_fo_import.py`
- Role: Debug/support script; not intended as stable production API.
- Module Summary: Implementation module for debug and diagnostics.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.debug.debug_fob`

- File: `src/dataset_tools/debug/debug_fob.py`
- Role: Debug/support script; not intended as stable production API.
- Module Summary: Implementation module for debug and diagnostics.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.debug.debug_imports`

- File: `src/dataset_tools/debug/debug_imports.py`
- Role: Debug/support script; not intended as stable production API.
- Module Summary: Implementation module for debug and diagnostics.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.debug.debug_ls_tasks`

- File: `src/dataset_tools/debug/debug_ls_tasks.py`
- Role: Debug/support script; not intended as stable production API.
- Module Summary: Implementation module for debug and diagnostics.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `debug_tasks` | `()` | 14 | Public callable | Perform debug tasks. |

## `src.dataset_tools.debug.debug_ls_urls`

- File: `src/dataset_tools/debug/debug_ls_urls.py`
- Role: Debug/support script; not intended as stable production API.
- Module Summary: Implementation module for debug and diagnostics.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `main` | `()` | 16 | CLI entrypoint | Perform main. |

## `src.dataset_tools.dst`

- File: `src/dataset_tools/dst.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Dataset Tools unified CLI entrypoint (`dst`).

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_parse_json_value` | `(raw: str, label: str) -> Any` | 22 | Internal helper | Parse raw JSON text for a named CLI argument and raise readable errors. |
| `_parse_json_dict` | `(raw: str | None, label: str) -> dict[str, Any]` | 31 | Internal helper | Parse an optional JSON object argument and normalize missing values to empty dicts. |
| `_parse_json_list` | `(raw: str, label: str) -> list[Any]` | 44 | Internal helper | Parse and validate a JSON array argument from the CLI. |
| `_parse_csv_list` | `(raw: str, label: str) -> list[str]` | 53 | Internal helper | Parse a comma-separated CLI value into a non-empty list of tokens. |
| `_parse_class_map` | `(raw: str | None) -> dict[int, str]` | 63 | Internal helper | Convert class-map JSON into an integer-keyed mapping for YOLO loaders. |
| `_parse_path_replacements` | `(entries: Iterable[str] | None) -> tuple[tuple[str, str], ...] | None` | 77 | Internal helper | Parse repeated SRC=DST replacement rules used by disk sync routines. |
| `_parse_optional_text` | `(raw: str | None) -> str | None` | 97 | Internal helper | Normalize empty/None-like strings into Python None for optional fields. |
| `_parse_devices` | `(raw: str | int | None) -> str | int | None` | 110 | Internal helper | Normalize anomaly `--devices` input to an int, string selector, or None. |
| `_load_app_config` | `(args) -> Any` | 125 | Internal helper | Load the resolved application configuration from defaults, local config, and overrides. |
| `_list_projects` | `(ls) -> list[Any]` | 135 | Internal helper | List Label Studio projects while supporting API version differences. |
| `_project_payload` | `(project, include_task_count: bool = False) -> dict[str, Any]` | 145 | Internal helper | Convert a Label Studio project object into a stable dictionary payload. |
| `_mask_api_key` | `(value: str) -> str` | 162 | Internal helper | Mask secret tokens before printing config payloads. |
| `_print_result` | `(result: Any)` | 172 | Internal helper | Render command output payloads to stdout in a consistent format. |
| `_write_json_output` | `(path: str | None, payload: Any)` | 185 | Internal helper | Optionally persist command results to a JSON file path. |
| `_execute_with_optional_log_capture` | `(fn, quiet_logs: bool)` | 197 | Internal helper | Execute a callable while optionally suppressing noisy library logs. |
| `cmd_config_show` | `(args)` | 242 | Public callable | Run the `dst config show` command handler and return a JSON-serializable result. |
| `cmd_ls_test` | `(args)` | 265 | Public callable | Run the `dst ls test` command handler and return a JSON-serializable result. |
| `cmd_ls_project_list` | `(args)` | 297 | Public callable | Run the `dst ls project list` command handler and return a JSON-serializable result. |
| `_resolve_project` | `(ls, project_id: int | None, project_title: str | None)` | 337 | Internal helper | Resolve a Label Studio project by id or exact title. |
| `cmd_ls_project_clear_tasks` | `(args)` | 358 | Public callable | Run the `dst ls project clear tasks` command handler and return a JSON-serializable result. |
| `cmd_ls_project_cleanup` | `(args)` | 392 | Public callable | Run the `dst ls project cleanup` command handler and return a JSON-serializable result. |
| `cmd_data_load_yolo` | `(args)` | 439 | Public callable | Run the `dst data load yolo` command handler and return a JSON-serializable result. |
| `cmd_data_load_coco` | `(args)` | 499 | Public callable | Run the `dst data load coco` command handler and return a JSON-serializable result. |
| `cmd_data_export_ls_json` | `(args)` | 538 | Public callable | Run the `dst data export ls json` command handler and return a JSON-serializable result. |
| `cmd_metrics_embeddings` | `(args)` | 567 | Public callable | Run the `dst metrics embeddings` command handler and return a JSON-serializable result. |
| `cmd_metrics_uniqueness` | `(args)` | 596 | Public callable | Run the `dst metrics uniqueness` command handler and return a JSON-serializable result. |
| `cmd_metrics_mistakenness` | `(args)` | 620 | Public callable | Run the `dst metrics mistakenness` command handler and return a JSON-serializable result. |
| `cmd_metrics_hardness` | `(args)` | 647 | Public callable | Run the `dst metrics hardness` command handler and return a JSON-serializable result. |
| `cmd_metrics_representativeness` | `(args)` | 671 | Public callable | Run the `dst metrics representativeness` command handler and return a JSON-serializable result. |
| `cmd_brain_visualization` | `(args)` | 697 | Public callable | Run the `dst brain visualization` command handler and return a JSON-serializable result. |
| `cmd_brain_similarity` | `(args)` | 724 | Public callable | Run the `dst brain similarity` command handler and return a JSON-serializable result. |
| `cmd_brain_duplicates_exact` | `(args)` | 751 | Public callable | Run the `dst brain duplicates exact` command handler and return a JSON-serializable result. |
| `cmd_brain_duplicates_near` | `(args)` | 771 | Public callable | Run the `dst brain duplicates near` command handler and return a JSON-serializable result. |
| `cmd_brain_leaky_splits` | `(args)` | 796 | Public callable | Run the `dst brain leaky splits` command handler and return a JSON-serializable result. |
| `cmd_models_list` | `(args)` | 824 | Public callable | Run the `dst models list` command handler and return a JSON-serializable result. |
| `cmd_models_resolve` | `(args)` | 858 | Public callable | Run the `dst models resolve` command handler and return a JSON-serializable result. |
| `cmd_models_validate` | `(args)` | 890 | Public callable | Run the `dst models validate` command handler and return a JSON-serializable result. |
| `cmd_anomaly_fit` | `(args)` | 909 | Public callable | Run the `dst anomaly fit` command handler and return a JSON-serializable result. |
| `cmd_anomaly_train` | `(args)` | 947 | Public callable | Run the `dst anomaly train` command handler and return a JSON-serializable result. |
| `cmd_anomaly_score` | `(args)` | 1004 | Public callable | Run the `dst anomaly score` command handler and return a JSON-serializable result. |
| `cmd_anomaly_run` | `(args)` | 1073 | Public callable | Run the `dst anomaly run` command handler and return a JSON-serializable result. |
| `cmd_workflow_roundtrip` | `(args)` | 1126 | Public callable | Run the `dst workflow roundtrip` command handler and return a JSON-serializable result. |
| `_build_tag_rules` | `(payload_rules: list[dict[str, Any]])` | 1192 | Internal helper | Translate JSON workflow rules into typed tag-workflow rule objects. |
| `cmd_workflow_tags_run` | `(args)` | 1207 | Public callable | Run the `dst workflow tags run` command handler and return a JSON-serializable result. |
| `cmd_workflow_tags_inline` | `(args)` | 1266 | Public callable | Run the `dst workflow tags inline` command handler and return a JSON-serializable result. |
| `cmd_sync_disk` | `(args)` | 1302 | Public callable | Run the `dst sync disk` command handler and return a JSON-serializable result. |
| `cmd_app_open` | `(args)` | 1342 | Public callable | Run the `dst app open` command handler and return a JSON-serializable result. |
| `_add_common_config_args` | `(parser: argparse.ArgumentParser)` | 1384 | Internal helper | Attach shared `--config` and `--overrides` arguments to a parser. |
| `_add_persistent_args` | `(parser: argparse.ArgumentParser)` | 1399 | Internal helper | Attach mutually-exclusive persistence flags to loader commands. |
| `build_parser` | `() -> argparse.ArgumentParser` | 1407 | CLI entrypoint | Build and return the full `dst` argparse command tree. |
| `main` | `(argv: list[str] | None = None) -> int` | 1857 | CLI entrypoint | CLI program entrypoint used by `./dst` and `python -m dataset_tools.dst`. |

## `src.dataset_tools.label_studio`

- File: `src/dataset_tools/label_studio/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.label_studio`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.label_studio.client`

- File: `src/dataset_tools/label_studio/client.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for Label Studio integration.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_import_label_studio_client` | `() -> Any` | 15 | Internal helper | Internal helper for import label studio client. |
| `_resolve_access_token` | `(url: str, api_key: str) -> str` | 36 | Internal helper | Resolve access token from provided inputs. |
| `connect_to_label_studio` | `(url: str, api_key: str)` | 67 | Integration/bootstrap API | Connect to to label studio and return a ready client. |
| `ensure_label_studio_client` | `(config: AppConfig)` | 88 | Integration/bootstrap API | Ensure label studio client exists and return it. |

## `src.dataset_tools.label_studio.storage`

- File: `src/dataset_tools/label_studio/storage.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for Label Studio integration.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_is_local_storage` | `(storage: dict) -> bool` | 11 | Internal helper | Internal helper for is local storage. |
| `build_rectangle_label_config` | `(labels: Iterable[str]) -> str` | 23 | Public callable | Build rectangle label config for downstream steps. |
| `_list_projects` | `(ls)` | 53 | Internal helper | List available projects. |
| `find_project` | `(ls, title: str)` | 69 | Public callable | Perform find project. |
| `ensure_project` | `(ls, config: AppConfig, title: str | None = None, label_config: str | None = None)` | 85 | Integration/bootstrap API | Ensure project exists and return it. |
| `ensure_local_storage` | `(ls, project, config: AppConfig)` | 117 | Integration/bootstrap API | Ensure local storage exists and return it. |
| `ensure_target_storage` | `(ls, project, config: AppConfig)` | 154 | Integration/bootstrap API | Ensure target storage exists and return it. |

## `src.dataset_tools.label_studio.sync`

- File: `src/dataset_tools/label_studio/sync.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Push/pull synchronization helpers between FiftyOne and Label Studio.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_set_label_studio_env` | `(url: str, api_key: str)` | 24 | Internal helper | Export LS credentials into env vars used by FiftyOne annotate backend. |
| `push_view_to_label_studio` | `(view, config: AppConfig, project_name: str | None = None, annotation_key: str | None = None, label_field: str | None = None, launch_editor: bool = False, overwrite_annotation_run: bool = True)` | 32 | Public callable | Push a view to Label Studio using FiftyOne's annotate backend. |
| `_to_local_files_url` | `(filepath: str, config: AppConfig) -> str | None` | 83 | Internal helper | Convert an absolute filepath into Label Studio local-files URL. |
| `preflight_validate_upload` | `(view, project, config: AppConfig, strategy: str, strict: bool = True, ls_client = None) -> dict[str, Any]` | 104 | Public callable | Validate upload prerequisites before sending tasks to Label Studio. |
| `push_view_to_label_studio_sdk` | `(view, project, config: AppConfig, label_field: str | None = None) -> int` | 212 | Public callable | Push a view to Label Studio via direct SDK batched task import. |
| `delete_project_tasks` | `(project)` | 266 | Public callable | Delete all tasks from a Label Studio project. |
| `pull_labeled_tasks_to_fiftyone` | `(dataset, project, corrections_field: str = 'ls_corrections') -> int` | 271 | Public callable | Pull submitted LS tasks into FiftyOne using task metadata mapping. |
| `pull_labeled_tasks_from_annotation_run` | `(dataset, ls_client, annotation_key: str, corrections_field: str = 'ls_corrections') -> int` | 328 | Public callable | Pull LS annotations by replaying a FiftyOne annotation-run task map. |

## `src.dataset_tools.label_studio.translator`

- File: `src/dataset_tools/label_studio/translator.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for Label Studio integration.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_safe_percent` | `(value: float) -> float` | 8 | Internal helper | Internal helper for safe percent. |
| `fo_detection_to_ls_result` | `(det: Any, default_label: str = 'Insect') -> dict[str, Any]` | 20 | Public callable | Perform fo detection to ls result. |
| `ls_rectangle_result_to_fo_detection` | `(result: dict[str, Any])` | 54 | Public callable | Perform ls rectangle result to fo detection. |

## `src.dataset_tools.label_studio.uploader`

- File: `src/dataset_tools/label_studio/uploader.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for Label Studio integration.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `install_batched_upload_patch` | `(batch_size: int = 10)` | 12 | Public callable | Perform install batched upload patch. |

## `src.dataset_tools.label_studio_json`

- File: `src/dataset_tools/label_studio_json.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for Label Studio integration.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `build_tasks` | `(root_dir: Path, ls_root: str)` | 10 | Public callable | Build tasks for downstream steps. |

## `src.dataset_tools.loader`

- File: `src/dataset_tools/loader.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for dataset loading.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `import_yolo_dataset_from_root` | `(root_dir: str, dataset_name: str, image_subdir: str = 'images', labels_subdir: str = 'labels', class_id_to_label: dict[int, str] | None = None, overwrite: bool = True)` | 17 | Public callable | Perform import yolo dataset from root. |
| `import_yolo_dataset_from_roots` | `(images_root: str, labels_root: str, dataset_name: str, class_id_to_label: dict[int, str] | None = None, overwrite: bool = True)` | 50 | Public callable | Perform import yolo dataset from roots. |
| `get_or_create_dataset` | `(name: str)` | 77 | Public callable | Perform get or create dataset. |

## `src.dataset_tools.loaders`

- File: `src/dataset_tools/loaders/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.loaders`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.loaders.base`

- File: `src/dataset_tools/loaders/base.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for dataset loading.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `LoaderResult` | 12 | Public callable | LoaderResult used by dataset loading. |
| `BaseDatasetLoader` | 19 | Public callable | Dataset loader that imports source media/annotations into FiftyOne. |

#### `LoaderResult` methods
- Class Summary: LoaderResult used by dataset loading.
- No methods declared in this class body.

#### `BaseDatasetLoader` methods
- Class Summary: Dataset loader that imports source media/annotations into FiftyOne.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `load` | `(self, dataset_name: str, overwrite: bool = False, persistent: bool = True) -> LoaderResult` | 24 | Public callable | Perform load. |
| `_create_or_replace_dataset` | `(name: str, overwrite: bool, persistent: bool)` | 38 | Internal helper | Internal helper for create or replace dataset. |

## `src.dataset_tools.loaders.coco`

- File: `src/dataset_tools/loaders/coco.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for dataset loading.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `CocoLoaderConfig` | 14 | Public callable | Configuration dataclass for dataset loading. |
| `CocoDatasetLoader` | 22 | Public callable | Dataset loader that imports source media/annotations into FiftyOne. |

#### `CocoLoaderConfig` methods
- Class Summary: Configuration dataclass for dataset loading.
- No methods declared in this class body.

#### `CocoDatasetLoader` methods
- Class Summary: Dataset loader that imports source media/annotations into FiftyOne.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, config: CocoLoaderConfig)` | 25 | Internal helper | Initialize `CocoDatasetLoader` with runtime parameters. |
| `load` | `(self, dataset_name: str, overwrite: bool = False, persistent: bool = True) -> LoaderResult` | 36 | Public callable | Perform load. |

## `src.dataset_tools.loaders.path_resolvers`

- File: `src/dataset_tools/loaders/path_resolvers.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for dataset loading.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `MirroredRootsPathResolver` | 11 | Public callable | MirroredRootsPathResolver used by dataset loading. |
| `ImagesLabelsSubdirResolver` | 32 | Public callable | ImagesLabelsSubdirResolver used by dataset loading. |

#### `MirroredRootsPathResolver` methods
- Class Summary: MirroredRootsPathResolver used by dataset loading.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `label_path_for` | `(self, image_path: Path) -> Path` | 18 | Public callable | Perform label path for. |

#### `ImagesLabelsSubdirResolver` methods
- Class Summary: ImagesLabelsSubdirResolver used by dataset loading.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `images_root` | `(self) -> Path` | 41 | Public callable | Perform images root. |
| `labels_root` | `(self) -> Path` | 50 | Public callable | Perform labels root. |
| `label_path_for` | `(self, image_path: Path) -> Path` | 58 | Public callable | Perform label path for. |

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `default_image_filter` | `(path: Path) -> bool` | 71 | Public callable | Perform default image filter. |

## `src.dataset_tools.loaders.yolo`

- File: `src/dataset_tools/loaders/yolo.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for dataset loading.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `YoloParserConfig` | 20 | Public callable | Configuration dataclass for dataset loading. |
| `YoloDatasetLoader` | 27 | Public callable | Dataset loader that imports source media/annotations into FiftyOne. |

#### `YoloParserConfig` methods
- Class Summary: Configuration dataclass for dataset loading.
- No methods declared in this class body.

#### `YoloDatasetLoader` methods
- Class Summary: Dataset loader that imports source media/annotations into FiftyOne.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, resolver: MirroredRootsPathResolver | ImagesLabelsSubdirResolver, parser_config: YoloParserConfig | None = None, image_filter: Callable[[Path], bool] = default_image_filter, sample_metadata_fields: dict[str, Callable[[Path], object]] | None = None)` | 30 | Internal helper | Initialize `YoloDatasetLoader` with runtime parameters. |
| `load` | `(self, dataset_name: str, overwrite: bool = False, persistent: bool = True) -> LoaderResult` | 53 | Public callable | Perform load. |
| `_parse_yolo_file` | `(self, label_path: Path)` | 87 | Internal helper | Parse and validate yolo file input values. |

## `src.dataset_tools.metrics`

- File: `src/dataset_tools/metrics/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.metrics`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.metrics.base`

- File: `src/dataset_tools/metrics/base.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for metric computation.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `BaseMetricComputation` | 10 | Public callable | BaseMetricComputation used by metric computation. |

#### `BaseMetricComputation` methods
- Class Summary: BaseMetricComputation used by metric computation.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str)` | 13 | Internal helper | Initialize `BaseMetricComputation` with runtime parameters. |
| `load_dataset` | `(self)` | 24 | Public callable | Load dataset required by this module. |
| `run` | `(self)` | 34 | Public callable | Run the operation and return execution results. |
| `compute` | `(self, dataset)` | 44 | Public callable | Perform compute. |

## `src.dataset_tools.metrics.embeddings`

- File: `src/dataset_tools/metrics/embeddings.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for metric computation.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `EmbeddingsComputation` | 15 | Public callable | EmbeddingsComputation used by metric computation. |

#### `EmbeddingsComputation` methods
- Class Summary: EmbeddingsComputation used by metric computation.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, model_name: str = 'facebook/dinov2-base', model_ref: str | None = None, embeddings_field: str = 'embeddings', patches_field: str | None = None, use_umap: bool = True, use_cluster: bool = True, n_clusters: int = 10)` | 18 | Internal helper | Initialize `EmbeddingsComputation` with runtime parameters. |
| `compute` | `(self, dataset)` | 53 | Public callable | Perform compute. |

## `src.dataset_tools.metrics.field_metric`

- File: `src/dataset_tools/metrics/field_metric.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for metric computation.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `FieldMetricComputation` | 12 | Public callable | FieldMetricComputation used by metric computation. |

#### `FieldMetricComputation` methods
- Class Summary: FieldMetricComputation used by metric computation.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, required_fields: Iterable[str] | None = None)` | 16 | Internal helper | Initialize `FieldMetricComputation` with runtime parameters. |
| `validate_required_fields` | `(self, dataset)` | 29 | Public callable | Perform validate required fields. |
| `run` | `(self)` | 45 | Public callable | Run the operation and return execution results. |
| `compute` | `(self, dataset) -> Any` | 59 | Public callable | Perform compute. |

## `src.dataset_tools.metrics.hardness`

- File: `src/dataset_tools/metrics/hardness.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for metric computation.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `HardnessComputation` | 11 | Public callable | HardnessComputation used by metric computation. |

#### `HardnessComputation` methods
- Class Summary: HardnessComputation used by metric computation.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, label_field: str = 'ground_truth', output_field: str = 'hardness')` | 14 | Internal helper | Initialize `HardnessComputation` with runtime parameters. |
| `compute` | `(self, dataset)` | 34 | Public callable | Perform compute. |

## `src.dataset_tools.metrics.mistakenness`

- File: `src/dataset_tools/metrics/mistakenness.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for metric computation.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `MistakennessComputation` | 10 | Public callable | MistakennessComputation used by metric computation. |

#### `MistakennessComputation` methods
- Class Summary: MistakennessComputation used by metric computation.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, pred_field: str = 'predictions', gt_field: str = 'ground_truth', mistakenness_field: str = 'mistakenness', missing_field: str = 'possible_missing', spurious_field: str = 'possible_spurious')` | 13 | Internal helper | Initialize `MistakennessComputation` with runtime parameters. |
| `compute` | `(self, dataset)` | 42 | Public callable | Perform compute. |

## `src.dataset_tools.metrics.representativeness`

- File: `src/dataset_tools/metrics/representativeness.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for metric computation.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `RepresentativenessComputation` | 10 | Public callable | RepresentativenessComputation used by metric computation. |

#### `RepresentativenessComputation` methods
- Class Summary: RepresentativenessComputation used by metric computation.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, output_field: str = 'representativeness', method: str = 'cluster-center', embeddings_field: str | None = None, roi_field: str | None = None)` | 15 | Internal helper | Initialize `RepresentativenessComputation` with runtime parameters. |
| `compute` | `(self, dataset)` | 46 | Public callable | Perform compute. |

## `src.dataset_tools.metrics.uniqueness`

- File: `src/dataset_tools/metrics/uniqueness.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for metric computation.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `UniquenessComputation` | 10 | Public callable | UniquenessComputation used by metric computation. |

#### `UniquenessComputation` methods
- Class Summary: UniquenessComputation used by metric computation.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, dataset_name: str, embeddings_field: str | None = None, output_field: str = 'uniqueness')` | 13 | Internal helper | Initialize `UniquenessComputation` with runtime parameters. |
| `compute` | `(self, dataset)` | 29 | Public callable | Perform compute. |

## `src.dataset_tools.models`

- File: `src/dataset_tools/models/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.models`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.models.base`

- File: `src/dataset_tools/models/base.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for model provider registry.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `ModelProvider` | 11 | Public callable | Model provider adapter that resolves and loads backend-specific models. |

#### `ModelProvider` methods
- Class Summary: Model provider adapter that resolves and loads backend-specific models.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `load` | `(self, model_ref: ModelRef, *, task: str | None = None, **kwargs: Any) -> LoadedModel` | 17 | Public callable | Perform load. |
| `list_models` | `(self, contains: str | None = None, limit: int | None = None) -> list[str]` | 30 | Public callable | List available models. |

## `src.dataset_tools.models.providers`

- File: `src/dataset_tools/models/providers/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.models.providers`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.models.providers.anomalib`

- File: `src/dataset_tools/models/providers/anomalib.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for model provider registry.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `AnomalibProvider` | 18 | Public callable | Model provider adapter that resolves and loads backend-specific models. |

#### `AnomalibProvider` methods
- Class Summary: Model provider adapter that resolves and loads backend-specific models.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_import_anomalib` | `(self)` | 23 | Internal helper | Internal helper for import anomalib. |
| `load` | `(self, model_ref: ModelRef, *, task: str | None = None, **kwargs: Any) -> LoadedModel` | 39 | Public callable | Perform load. |
| `list_models` | `(self, contains: str | None = None, limit: int | None = None) -> list[str]` | 101 | Public callable | List available models. |

## `src.dataset_tools.models.providers.fiftyone_zoo`

- File: `src/dataset_tools/models/providers/fiftyone_zoo.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for model provider registry.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `FiftyOneZooProvider` | 13 | Public callable | Model provider adapter that resolves and loads backend-specific models. |

#### `FiftyOneZooProvider` methods
- Class Summary: Model provider adapter that resolves and loads backend-specific models.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `load` | `(self, model_ref: ModelRef, *, task: str | None = None, **kwargs: Any) -> LoadedModel` | 18 | Public callable | Perform load. |
| `list_models` | `(self, contains: str | None = None, limit: int | None = None) -> list[str]` | 45 | Public callable | List available models. |

## `src.dataset_tools.models.providers.huggingface`

- File: `src/dataset_tools/models/providers/huggingface.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for model provider registry.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `HuggingFaceEmbeddingModel` | 18 | Public callable | HuggingFaceEmbeddingModel used by model provider registry. |
| `HuggingFaceProvider` | 94 | Public callable | Model provider adapter that resolves and loads backend-specific models. |

#### `HuggingFaceEmbeddingModel` methods
- Class Summary: HuggingFaceEmbeddingModel used by model provider registry.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, model_name: str)` | 21 | Internal helper | Initialize `HuggingFaceEmbeddingModel` with runtime parameters. |
| `media_type` | `(self)` | 36 | Public callable | Perform media type. |
| `has_embeddings` | `(self)` | 45 | Public callable | Perform has embeddings. |
| `embed` | `(self, arg)` | 53 | Public callable | Perform embed. |
| `embed_all` | `(self, args)` | 82 | Public callable | Perform embed all. |

#### `HuggingFaceProvider` methods
- Class Summary: Model provider adapter that resolves and loads backend-specific models.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `load` | `(self, model_ref: ModelRef, *, task: str | None = None, **kwargs: Any) -> LoadedModel` | 99 | Public callable | Perform load. |

## `src.dataset_tools.models.registry`

- File: `src/dataset_tools/models/registry.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for model provider registry.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_provider_instances` | `() -> dict[str, ModelProvider]` | 14 | Internal helper | Internal helper for provider instances. |
| `list_providers` | `() -> list[str]` | 27 | Public callable | List available providers. |
| `get_provider` | `(name: str) -> ModelProvider` | 36 | Public callable | Perform get provider. |
| `resolve_model_ref` | `(raw: str, default_provider: str = 'hf') -> ModelRef` | 50 | Parsing/resolution utility | Resolve model ref from provided inputs. |
| `load_model` | `(raw_model_ref: str, *, default_provider: str = 'hf', task: str | None = None, capability: str | None = None, **kwargs: Any) -> LoadedModel` | 63 | Public callable | Load model required by this module. |
| `provider_model_list` | `(provider_name: str, *, contains: str | None = None, limit: int | None = None) -> list[str]` | 97 | Public callable | Perform provider model list. |

## `src.dataset_tools.models.spec`

- File: `src/dataset_tools/models/spec.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for model provider registry.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `ModelRef` | 21 | Public callable | ModelRef used by model provider registry. |
| `LoadedModel` | 30 | Public callable | LoadedModel used by model provider registry. |

#### `ModelRef` methods
- Class Summary: ModelRef used by model provider registry.
- No methods declared in this class body.

#### `LoadedModel` methods
- Class Summary: LoadedModel used by model provider registry.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `supports` | `(self, capability: str | None) -> bool` | 38 | Public callable | Perform supports. |

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `normalize_provider` | `(provider: str) -> str` | 52 | Parsing/resolution utility | Perform normalize provider. |
| `parse_model_ref` | `(raw: str, default_provider: str = 'hf') -> ModelRef` | 70 | Parsing/resolution utility | Parse and normalize model ref. |

## `src.dataset_tools.sync_from_fo_to_disk`

- File: `src/dataset_tools/sync_from_fo_to_disk.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for dataset tools runtime.

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `backup_file` | `(filepath: str, suffix_format: str = '%Y%m%d_%H%M%S') -> str | None` | 13 | Public callable | Perform backup file. |
| `infer_label_path` | `(image_path: str, path_replacements: Iterable[tuple[str, str]]) -> str | None` | 32 | Public callable | Perform infer label path. |
| `sync_corrections_to_disk` | `(dataset_name: str | None = None, dry_run: bool = False, tag_filter: str | None = None, corrections_field: str | None = None, label_to_class_id: dict[str, int] | None = None, default_class_id: int | None = None, path_replacements: Iterable[tuple[str, str]] | None = None, backup_suffix_format: str | None = None) -> int` | 51 | Public callable | Perform sync corrections to disk. |

## `src.dataset_tools.tag_workflow`

- File: `src/dataset_tools/tag_workflow/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.tag_workflow`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.tag_workflow.config`

- File: `src/dataset_tools/tag_workflow/config.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for tag workflow execution.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `TagOperationRule` | 10 | Public callable | TagOperationRule used by tag workflow execution. |
| `TagWorkflowConfig` | 19 | Public callable | Configuration dataclass for tag workflow execution. |

#### `TagOperationRule` methods
- Class Summary: TagOperationRule used by tag workflow execution.
- No methods declared in this class body.

#### `TagWorkflowConfig` methods
- Class Summary: Configuration dataclass for tag workflow execution.
- No methods declared in this class body.

## `src.dataset_tools.tag_workflow.context`

- File: `src/dataset_tools/tag_workflow/context.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for tag workflow execution.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `TagWorkflowContext` | 12 | Public callable | TagWorkflowContext used by tag workflow execution. |

#### `TagWorkflowContext` methods
- Class Summary: TagWorkflowContext used by tag workflow execution.
- No methods declared in this class body.

## `src.dataset_tools.tag_workflow.engine`

- File: `src/dataset_tools/tag_workflow/engine.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Implementation module for tag workflow execution.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `TagWorkflowEngine` | 15 | Public callable | TagWorkflowEngine used by tag workflow execution. |

#### `TagWorkflowEngine` methods
- Class Summary: TagWorkflowEngine used by tag workflow execution.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, app_config: AppConfig, operations: dict[str, Any] | None = None)` | 18 | Internal helper | Initialize `TagWorkflowEngine` with runtime parameters. |
| `register_operation` | `(self, name: str, operation)` | 31 | Public callable | Perform register operation. |
| `run` | `(self, workflow_config: TagWorkflowConfig) -> list[dict[str, Any]]` | 43 | Public callable | Run the operation and return execution results. |

## `src.dataset_tools.tag_workflow.operations`

- File: `src/dataset_tools/tag_workflow/operations/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.tag_workflow.operations`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.tag_workflow.operations.analysis`

- File: `src/dataset_tools/tag_workflow/operations/analysis.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Analysis-focused tag-workflow operations.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `ComputeUniquenessOperation` | 64 | Public callable | Compute uniqueness scores for samples in view or dataset scope. |
| `ComputeHardnessOperation` | 108 | Public callable | Compute hardness for classification-style fields. |
| `ComputeRepresentativenessOperation` | 161 | Public callable | Compute representativeness metrics with optional embeddings/ROI settings. |
| `ComputeSimilarityIndexOperation` | 219 | Public callable | Compute dataset-level similarity index (brain run). |
| `ComputeExactDuplicatesOperation` | 254 | Public callable | Compute exact-duplicate summary statistics at dataset scope. |
| `ComputeNearDuplicatesOperation` | 288 | Public callable | Compute near-duplicate relationships and summary counts. |
| `ComputeLeakySplitsOperation` | 334 | Public callable | Detect potential train/val/test leakage across configured splits. |
| `ComputeAnomalyScoresOperation` | 387 | Public callable | Compute anomaly scores via embedding-distance or anomalib artifact backend. |

#### `ComputeUniquenessOperation` methods
- Class Summary: Compute uniqueness scores for samples in view or dataset scope.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 68 | Public callable | Compute and store uniqueness field via ``fiftyone.brain.compute_uniqueness``. |

#### `ComputeHardnessOperation` methods
- Class Summary: Compute hardness for classification-style fields.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 112 | Public callable | Validate label type and compute hardness scores for target scope. |

#### `ComputeRepresentativenessOperation` methods
- Class Summary: Compute representativeness metrics with optional embeddings/ROI settings.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 165 | Public callable | Compute representativeness and enforce clustering prerequisites. |

#### `ComputeSimilarityIndexOperation` methods
- Class Summary: Compute dataset-level similarity index (brain run).

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 223 | Public callable | Run ``fob.compute_similarity`` with optional embeddings/backend settings. |

#### `ComputeExactDuplicatesOperation` methods
- Class Summary: Compute exact-duplicate summary statistics at dataset scope.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 258 | Public callable | Run exact duplicate detection and return aggregate counts. |

#### `ComputeNearDuplicatesOperation` methods
- Class Summary: Compute near-duplicate relationships and summary counts.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 292 | Public callable | Run near-duplicate search and summarize affected samples/pairs. |

#### `ComputeLeakySplitsOperation` methods
- Class Summary: Detect potential train/val/test leakage across configured splits.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 338 | Public callable | Run leaky-split detection and return leakage summary payload. |

#### `ComputeAnomalyScoresOperation` methods
- Class Summary: Compute anomaly scores via embedding-distance or anomalib artifact backend.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 391 | Public callable | Dispatch anomaly scoring to selected backend and return backend payload. |

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_resolve_scope_collection` | `(context: TagWorkflowContext, view: Any, params: dict[str, Any], *, operation: str, default_scope: str) -> tuple[Any, str]` | 23 | Internal helper | Resolve whether an operation should run on the full dataset or current view. |
| `_require_dataset_scope` | `(params: dict[str, Any], operation: str)` | 47 | Internal helper | Enforce dataset-global scope for operations that cannot run on a view. |
| `_ensure_sample_field` | `(collection: Any, field: str, dataset_name: str)` | 56 | Internal helper | Raise if ``field`` is missing on target sample collection. |

## `src.dataset_tools.tag_workflow.operations.base`

- File: `src/dataset_tools/tag_workflow/operations/base.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Base operation contract for tag-workflow actions.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `TagOperation` | 10 | Public callable | Abstract interface implemented by all tag-workflow operations. |

#### `TagOperation` methods
- Class Summary: Abstract interface implemented by all tag-workflow operations.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None) -> dict[str, Any]` | 15 | Public callable | Execute operation logic for the selected tag/view scope. |

## `src.dataset_tools.tag_workflow.operations.core`

- File: `src/dataset_tools/tag_workflow/operations/core.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Core tag-workflow operations for mutation, LS sync, and disk sync.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `DeleteSamplesOperation` | 77 | Public callable | Delete selected samples from the active dataset. |
| `DeleteFilesAndSamplesOperation` | 88 | Public callable | Delete media files on disk and then delete corresponding samples. |
| `MoveSamplesToDatasetOperation` | 110 | Public callable | Copy/move selected samples into a target dataset. |
| `SendToLabelStudioOperation` | 147 | Public callable | Send selected samples to Label Studio using configured upload strategy. |
| `PullFromLabelStudioOperation` | 223 | Public callable | Pull submitted LS annotations back into FiftyOne correction fields. |
| `SyncCorrectionsToDiskOperation` | 288 | Public callable | Write correction fields from FiftyOne samples back to label files on disk. |

#### `DeleteSamplesOperation` methods
- Class Summary: Delete selected samples from the active dataset.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 81 | Public callable | Delete all samples in ``view`` and report count. |

#### `DeleteFilesAndSamplesOperation` methods
- Class Summary: Delete media files on disk and then delete corresponding samples.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 92 | Public callable | Remove existing files referenced by ``view`` and drop samples. |

#### `MoveSamplesToDatasetOperation` methods
- Class Summary: Copy/move selected samples into a target dataset.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 114 | Public callable | Add ``view`` samples to target dataset and optionally remove source samples. |

#### `SendToLabelStudioOperation` methods
- Class Summary: Send selected samples to Label Studio using configured upload strategy.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 151 | Public callable | Push samples to LS and return transfer diagnostics. |

#### `PullFromLabelStudioOperation` methods
- Class Summary: Pull submitted LS annotations back into FiftyOne correction fields.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 227 | Public callable | Pull LS annotations using strategy-compatible mapping path. |

#### `SyncCorrectionsToDiskOperation` methods
- Class Summary: Write correction fields from FiftyOne samples back to label files on disk.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `execute` | `(self, context: TagWorkflowContext, view: Any, params: dict[str, Any], tag: str | None)` | 292 | Public callable | Run disk sync with workflow/config overrides and report synced file count. |

### Functions

| Function | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `_app_config_with_ls_overrides` | `(config: AppConfig, params: dict[str, Any]) -> AppConfig` | 41 | Internal helper | Return config copy with optional Label Studio/dataset overrides from rule params. |
| `default_operations_registry` | `() -> dict[str, TagOperation]` | 317 | Public callable | Return default registry of core and analysis operations by operation name. |

## `src.dataset_tools.tools`

- File: `src/dataset_tools/tools/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.tools`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.workflows`

- File: `src/dataset_tools/workflows/__init__.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: Package initializer for `dataset_tools.workflows`.
- No top-level classes or functions (export-only or script-only module).

## `src.dataset_tools.workflows.roundtrip`

- File: `src/dataset_tools/workflows/roundtrip.py`
- Role: Internal module in dataset_tools. See source and call graph for behavior.
- Module Summary: High-level orchestration for FiftyOne <-> Label Studio curation roundtrip.

### Classes

| Class | Line | Role | Summary |
|---|---:|---|---|
| `RoundtripWorkflowConfig` | 19 | Public callable | Configuration for a single roundtrip execution. |
| `CurationRoundtripWorkflow` | 42 | Public callable | Build and execute roundtrip workflows via the tag-workflow engine. |

#### `RoundtripWorkflowConfig` methods
- Class Summary: Configuration for a single roundtrip execution.
- No methods declared in this class body.

#### `CurationRoundtripWorkflow` methods
- Class Summary: Build and execute roundtrip workflows via the tag-workflow engine.

| Method | Signature | Line | Role | Summary |
|---|---|---:|---|---|
| `__init__` | `(self, app_config: AppConfig)` | 45 | Internal helper | Create a roundtrip workflow runner bound to resolved app config. |
| `run` | `(self, config: RoundtripWorkflowConfig) -> list[dict[str, Any]]` | 54 | Public callable | Build stage rules from ``config`` and execute them in order. |
