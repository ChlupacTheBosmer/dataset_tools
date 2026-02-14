# Dataset Tools CLI Reference (`dst`)

This reference is generated from `dataset_tools.dst.build_parser()` and documents the current CLI contract.

- Primary entrypoint script: `./dst`
- Python module entrypoint: `python -m dataset_tools.dst`

## `dst`

Dataset Tools CLI

- Parser prog: `dst`
- Help: `./dst --help`

**Subcommands**

- `anomaly`
- `app`
- `brain`
- `config`
- `data`
- `ls`
- `metrics`
- `models`
- `sync`
- `workflow`

### `anomaly`

- Parser prog: `dst anomaly`
- Help: `./dst anomaly --help`

**Subcommands**

- `fit`
- `run`
- `score`
- `train`

#### `anomaly fit`

- Parser prog: `dst anomaly fit`
- Help: `./dst anomaly fit --help`
- Handler: `cmd_anomaly_fit`
- Handler Summary: Run the `dst anomaly fit` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--backend` | `backend` | no | embedding_distance, anomalib | embedding_distance |  |
| `--embeddings-field` | `embeddings_field` | no |  | embeddings |  |
| `--normal-tag` | `normal_tag` | no |  |  |  |
| `--threshold` | `threshold` | no |  |  |  |
| `--threshold-quantile` | `threshold_quantile` | no |  | 0.95 |  |
| `--reference-json` | `reference_json` | no |  |  | Optional path to persist fitted reference |

#### `anomaly run`

- Parser prog: `dst anomaly run`
- Help: `./dst anomaly run --help`
- Handler: `cmd_anomaly_run`
- Handler Summary: Run the `dst anomaly run` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--backend` | `backend` | no | embedding_distance, anomalib | embedding_distance |  |
| `--embeddings-field` | `embeddings_field` | no |  | embeddings |  |
| `--normal-tag` | `normal_tag` | no |  |  |  |
| `--threshold` | `threshold` | no |  |  |  |
| `--threshold-quantile` | `threshold_quantile` | no |  | 0.95 |  |
| `--reference-json` | `reference_json` | no |  |  | Optional path to persist fitted reference |
| `--model-ref` | `model_ref` | no |  | anomalib:padim | Deprecated for backend=anomalib; use --artifact |
| `--artifact` | `artifact` | no |  |  | Path to anomalib artifact JSON or exported model file |
| `--artifact-format` | `artifact_format` | no | openvino, torch |  |  |
| `--anomaly-threshold` | `anomaly_threshold` | no |  | 0.5 |  |
| `--device` | `device` | no |  |  | Inference device for anomalib backend |
| `--trust-remote-code` | `trust_remote_code` | no |  | False | Allow loading torch anomalib artifacts via pickle (only for trusted artifacts) |
| `--tag` | `tag` | no |  |  | Optional sample tag filter for scoring |
| `--score-field` | `score_field` | no |  | anomaly_score |  |
| `--flag-field` | `flag_field` | no |  | is_anomaly |  |
| `--label-field` | `label_field` | no |  |  | Optional Classification output field |
| `--map-field` | `map_field` | no |  |  | Optional Heatmap output field |
| `--mask-field` | `mask_field` | no |  |  | Optional Segmentation output field |

#### `anomaly score`

- Parser prog: `dst anomaly score`
- Help: `./dst anomaly score --help`
- Handler: `cmd_anomaly_score`
- Handler Summary: Run the `dst anomaly score` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--backend` | `backend` | no | embedding_distance, anomalib | embedding_distance |  |
| `--embeddings-field` | `embeddings_field` | no |  | embeddings |  |
| `--normal-tag` | `normal_tag` | no |  |  |  |
| `--threshold` | `threshold` | no |  |  |  |
| `--threshold-quantile` | `threshold_quantile` | no |  | 0.95 |  |
| `--reference-json` | `reference_json` | no |  |  | Optional path to fitted reference JSON |
| `--model-ref` | `model_ref` | no |  | anomalib:padim | Deprecated for backend=anomalib; use --artifact |
| `--artifact` | `artifact` | no |  |  | Path to anomalib artifact JSON or exported model file |
| `--artifact-format` | `artifact_format` | no | openvino, torch |  |  |
| `--anomaly-threshold` | `anomaly_threshold` | no |  | 0.5 | Threshold for anomaly flag/label |
| `--device` | `device` | no |  |  | Inference device for anomalib backend |
| `--trust-remote-code` | `trust_remote_code` | no |  | False | Allow loading torch anomalib artifacts via pickle (only for trusted artifacts) |
| `--tag` | `tag` | no |  |  | Optional sample tag filter for scoring |
| `--score-field` | `score_field` | no |  | anomaly_score |  |
| `--flag-field` | `flag_field` | no |  | is_anomaly |  |
| `--label-field` | `label_field` | no |  |  | Optional Classification output field |
| `--map-field` | `map_field` | no |  |  | Optional Heatmap output field |
| `--mask-field` | `mask_field` | no |  |  | Optional Segmentation output field |

#### `anomaly train`

- Parser prog: `dst anomaly train`
- Help: `./dst anomaly train --help`
- Handler: `cmd_anomaly_train`
- Handler Summary: Run the `dst anomaly train` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--model-ref` | `model_ref` | no |  | anomalib:padim |  |
| `--normal-tag` | `normal_tag` | no |  |  | Tag selecting normal samples for training |
| `--abnormal-tag` | `abnormal_tag` | no |  |  | Optional tag selecting abnormal samples |
| `--mask-field` | `mask_field` | no |  |  | Optional mask field for abnormal samples |
| `--artifact-dir` | `artifact_dir` | no |  |  | Directory where trained artifact is stored |
| `--data-dir` | `data_dir` | no |  |  | Directory for generated anomalib Folder data |
| `--artifact-format` | `artifact_format` | no | openvino, torch | openvino |  |
| `--artifact-json` | `artifact_json` | no |  |  | Optional explicit output path for artifact JSON |
| `--image-size` | `image_size` | no |  |  | Image size: N or W,H for resize pre-processing |
| `--train-batch-size` | `train_batch_size` | no |  | 8 |  |
| `--eval-batch-size` | `eval_batch_size` | no |  | 8 |  |
| `--num-workers` | `num_workers` | no |  | 0 |  |
| `--normal-split-ratio` | `normal_split_ratio` | no |  | 0.2 |  |
| `--test-split-mode` | `test_split_mode` | no |  | from_dir |  |
| `--test-split-ratio` | `test_split_ratio` | no |  | 0.2 |  |
| `--val-split-mode` | `val_split_mode` | no |  | same_as_test |  |
| `--val-split-ratio` | `val_split_ratio` | no |  | 0.5 |  |
| `--seed` | `seed` | no |  |  |  |
| `--max-epochs` | `max_epochs` | no |  |  |  |
| `--accelerator` | `accelerator` | no |  |  |  |
| `--devices` | `devices` | no |  |  |  |
| `--copy-media` | `copy_media` | no |  | False | Copy files instead of symlinking |
| `--overwrite-data` | `overwrite_data` | no |  | False | Replace existing generated training data directory |

### `app`

- Parser prog: `dst app`
- Help: `./dst app --help`

**Subcommands**

- `open`

#### `app open`

- Parser prog: `dst app open`
- Help: `./dst app open --help`
- Handler: `cmd_app_open`
- Handler Summary: Run the `dst app open` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--port` | `port` | no |  | 5151 |  |
| `--address` | `address` | no |  | 0.0.0.0 |  |
| `--no-block` | `no_block` | no |  | False | Launch app and return immediately |

### `brain`

- Parser prog: `dst brain`
- Help: `./dst brain --help`

**Subcommands**

- `duplicates`
- `leaky-splits`
- `similarity`
- `visualization`

#### `brain duplicates`

- Parser prog: `dst brain duplicates`
- Help: `./dst brain duplicates --help`

**Subcommands**

- `exact`
- `near`

##### `brain duplicates exact`

- Parser prog: `dst brain duplicates exact`
- Help: `./dst brain duplicates exact --help`
- Handler: `cmd_brain_duplicates_exact`
- Handler Summary: Run the `dst brain duplicates exact` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |

##### `brain duplicates near`

- Parser prog: `dst brain duplicates near`
- Help: `./dst brain duplicates near --help`
- Handler: `cmd_brain_duplicates_near`
- Handler Summary: Run the `dst brain duplicates near` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--threshold` | `threshold` | no |  | 0.2 |  |
| `--embeddings-field` | `embeddings_field` | no |  |  |  |
| `--roi-field` | `roi_field` | no |  |  |  |

#### `brain leaky-splits`

- Parser prog: `dst brain leaky-splits`
- Help: `./dst brain leaky-splits --help`
- Handler: `cmd_brain_leaky_splits`
- Handler Summary: Run the `dst brain leaky splits` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--splits` | `splits` | yes |  |  | Comma-separated split tags/values, e.g. train,val,test |
| `--threshold` | `threshold` | no |  | 0.2 |  |
| `--embeddings-field` | `embeddings_field` | no |  |  |  |
| `--roi-field` | `roi_field` | no |  |  |  |

#### `brain similarity`

- Parser prog: `dst brain similarity`
- Help: `./dst brain similarity --help`
- Handler: `cmd_brain_similarity`
- Handler Summary: Run the `dst brain similarity` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--embeddings-field` | `embeddings_field` | no |  |  |  |
| `--patches-field` | `patches_field` | no |  |  |  |
| `--roi-field` | `roi_field` | no |  |  |  |
| `--backend` | `backend` | no |  |  |  |
| `--brain-key` | `brain_key` | no |  |  |  |

#### `brain visualization`

- Parser prog: `dst brain visualization`
- Help: `./dst brain visualization --help`
- Handler: `cmd_brain_visualization`
- Handler Summary: Run the `dst brain visualization` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--method` | `method` | no |  | umap |  |
| `--num-dims` | `num_dims` | no |  | 2 |  |
| `--embeddings-field` | `embeddings_field` | no |  |  |  |
| `--patches-field` | `patches_field` | no |  |  |  |
| `--brain-key` | `brain_key` | no |  |  |  |

### `config`

- Parser prog: `dst config`
- Help: `./dst config --help`

**Subcommands**

- `show`

#### `config show`

- Parser prog: `dst config show`
- Help: `./dst config show --help`
- Handler: `cmd_config_show`
- Handler Summary: Run the `dst config show` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--show-secrets` | `show_secrets` | no |  | False | Do not mask sensitive fields |

### `data`

- Parser prog: `dst data`
- Help: `./dst data --help`

**Subcommands**

- `export`
- `load`

#### `data export`

- Parser prog: `dst data export`
- Help: `./dst data export --help`

**Subcommands**

- `ls-json`

##### `data export ls-json`

- Parser prog: `dst data export ls-json`
- Help: `./dst data export ls-json --help`
- Handler: `cmd_data_export_ls_json`
- Handler Summary: Run the `dst data export ls json` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--root` | `root` | yes |  |  | Root containing images/ and labels/ |
| `--ls-root` | `ls_root` | no |  | /data/local-files/?d=frames_visitors_ndb/images | Prefix used in generated task image URLs |
| `--output` | `output` | no |  |  | Output JSON path |

#### `data load`

- Parser prog: `dst data load`
- Help: `./dst data load --help`

**Subcommands**

- `coco`
- `yolo`

##### `data load coco`

- Parser prog: `dst data load coco`
- Help: `./dst data load coco --help`
- Handler: `cmd_data_load_coco`
- Handler Summary: Run the `dst data load coco` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--dataset` | `dataset` | no |  |  | Target FiftyOne dataset name |
| `--dataset-dir` | `dataset_dir` | yes |  |  |  |
| `--data-path` | `data_path` | no |  | data |  |
| `--labels-path` | `labels_path` | no |  | labels.json |  |
| `--overwrite` | `overwrite` | no |  | False |  |
| `--persistent` | `persistent` | no |  | True |  |
| `--non-persistent` | `persistent` | no |  | True |  |

##### `data load yolo`

- Parser prog: `dst data load yolo`
- Help: `./dst data load yolo --help`
- Handler: `cmd_data_load_yolo`
- Handler Summary: Run the `dst data load yolo` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--dataset` | `dataset` | no |  |  | Target FiftyOne dataset name |
| `--root` | `root` | no |  |  | Root containing images/ and labels/ |
| `--images-root` | `images_root` | no |  |  | Images root for mirrored layout |
| `--labels-root` | `labels_root` | no |  |  | Labels root for mirrored layout |
| `--images-subdir` | `images_subdir` | no |  | images |  |
| `--labels-subdir` | `labels_subdir` | no |  | labels |  |
| `--class-map` | `class_map` | no |  |  | JSON map like {"0":"rodent"} |
| `--no-confidence` | `no_confidence` | no |  | False | Ignore 6th YOLO confidence column |
| `--overwrite` | `overwrite` | no |  | False |  |
| `--persistent` | `persistent` | no |  | True |  |
| `--non-persistent` | `persistent` | no |  | True |  |

### `ls`

- Parser prog: `dst ls`
- Help: `./dst ls --help`

**Subcommands**

- `project`
- `test`

#### `ls project`

- Parser prog: `dst ls project`
- Help: `./dst ls project --help`

**Subcommands**

- `cleanup`
- `clear-tasks`
- `list`

##### `ls project cleanup`

- Parser prog: `dst ls project cleanup`
- Help: `./dst ls project cleanup --help`
- Handler: `cmd_ls_project_cleanup`
- Handler Summary: Run the `dst ls project cleanup` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--keyword` | `keyword` | yes |  |  | Keyword to match |
| `--dry-run` | `dry_run` | no |  | False |  |
| `--case-sensitive` | `case_sensitive` | no |  | False |  |

##### `ls project clear-tasks`

- Parser prog: `dst ls project clear-tasks`
- Help: `./dst ls project clear-tasks --help`
- Handler: `cmd_ls_project_clear_tasks`
- Handler Summary: Run the `dst ls project clear tasks` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--id` | `id` | no |  |  | Project ID |
| `--title` | `title` | no |  |  | Project title (exact match) |
| `--dry-run` | `dry_run` | no |  | False |  |

##### `ls project list`

- Parser prog: `dst ls project list`
- Help: `./dst ls project list --help`
- Handler: `cmd_ls_project_list`
- Handler Summary: Run the `dst ls project list` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--contains` | `contains` | no |  |  | Filter title by substring |
| `--case-sensitive` | `case_sensitive` | no |  | False |  |
| `--limit` | `limit` | no |  |  |  |
| `--with-task-count` | `with_task_count` | no |  | False |  |

#### `ls test`

- Parser prog: `dst ls test`
- Help: `./dst ls test --help`
- Handler: `cmd_ls_test`
- Handler Summary: Run the `dst ls test` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--list-projects` | `list_projects` | no |  | False | Include project list in output |

### `metrics`

- Parser prog: `dst metrics`
- Help: `./dst metrics --help`

**Subcommands**

- `embeddings`
- `hardness`
- `mistakenness`
- `representativeness`
- `uniqueness`

#### `metrics embeddings`

- Parser prog: `dst metrics embeddings`
- Help: `./dst metrics embeddings --help`
- Handler: `cmd_metrics_embeddings`
- Handler Summary: Run the `dst metrics embeddings` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--model` | `model` | no |  | facebook/dinov2-base | Legacy alias for HF model id when --model-ref is not provided |
| `--model-ref` | `model_ref` | no |  |  | Provider-qualified model reference (e.g. hf:facebook/dinov2-base, foz:clip-vit-base32-torch) |
| `--embeddings-field` | `embeddings_field` | no |  | embeddings |  |
| `--patches-field` | `patches_field` | no |  |  |  |
| `--no-umap` | `no_umap` | no |  | False |  |
| `--no-cluster` | `no_cluster` | no |  | False |  |
| `--n-clusters` | `n_clusters` | no |  | 10 |  |

#### `metrics hardness`

- Parser prog: `dst metrics hardness`
- Help: `./dst metrics hardness --help`
- Handler: `cmd_metrics_hardness`
- Handler Summary: Run the `dst metrics hardness` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--label-field` | `label_field` | no |  | ground_truth |  |
| `--output-field` | `output_field` | no |  | hardness |  |

#### `metrics mistakenness`

- Parser prog: `dst metrics mistakenness`
- Help: `./dst metrics mistakenness --help`
- Handler: `cmd_metrics_mistakenness`
- Handler Summary: Run the `dst metrics mistakenness` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--pred-field` | `pred_field` | no |  | predictions |  |
| `--gt-field` | `gt_field` | no |  | ground_truth |  |
| `--mistakenness-field` | `mistakenness_field` | no |  | mistakenness |  |
| `--missing-field` | `missing_field` | no |  | possible_missing |  |
| `--spurious-field` | `spurious_field` | no |  | possible_spurious |  |

#### `metrics representativeness`

- Parser prog: `dst metrics representativeness`
- Help: `./dst metrics representativeness --help`
- Handler: `cmd_metrics_representativeness`
- Handler Summary: Run the `dst metrics representativeness` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--output-field` | `output_field` | no |  | representativeness |  |
| `--method` | `method` | no |  | cluster-center |  |
| `--embeddings-field` | `embeddings_field` | no |  |  |  |
| `--roi-field` | `roi_field` | no |  |  |  |

#### `metrics uniqueness`

- Parser prog: `dst metrics uniqueness`
- Help: `./dst metrics uniqueness --help`
- Handler: `cmd_metrics_uniqueness`
- Handler Summary: Run the `dst metrics uniqueness` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--dataset` | `dataset` | yes |  |  |  |
| `--embeddings-field` | `embeddings_field` | no |  |  |  |
| `--output-field` | `output_field` | no |  | uniqueness |  |

### `models`

- Parser prog: `dst models`
- Help: `./dst models --help`

**Subcommands**

- `list`
- `resolve`
- `validate`

#### `models list`

- Parser prog: `dst models list`
- Help: `./dst models list --help`
- Handler: `cmd_models_list`
- Handler Summary: Run the `dst models list` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--provider` | `provider` | no | hf, foz, anomalib |  | Provider name. If omitted, lists available providers. |
| `--contains` | `contains` | no |  |  | Optional substring filter |
| `--limit` | `limit` | no |  | 50 | Max models to return |

#### `models resolve`

- Parser prog: `dst models resolve`
- Help: `./dst models resolve --help`
- Handler: `cmd_models_resolve`
- Handler Summary: Run the `dst models resolve` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--model-ref` | `model_ref` | yes |  |  | Provider-qualified model ref |
| `--default-provider` | `default_provider` | no | hf, foz, anomalib | hf | Provider used when --model-ref omits prefix |
| `--task` | `task` | no |  |  | Optional task hint (e.g. embeddings, anomaly) |
| `--capability` | `capability` | no |  |  | Required capability (e.g. embeddings, anomaly) |

#### `models validate`

- Parser prog: `dst models validate`
- Help: `./dst models validate --help`
- Handler: `cmd_models_validate`
- Handler Summary: Run the `dst models validate` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--model-ref` | `model_ref` | yes |  |  | Provider-qualified model ref |
| `--default-provider` | `default_provider` | no | hf, foz, anomalib | hf | Provider used when --model-ref omits prefix |
| `--task` | `task` | no |  |  | Optional task hint |
| `--capability` | `capability` | no |  |  | Required capability |

### `sync`

- Parser prog: `dst sync`
- Help: `./dst sync --help`

**Subcommands**

- `disk`

#### `sync disk`

- Parser prog: `dst sync disk`
- Help: `./dst sync disk --help`
- Handler: `cmd_sync_disk`
- Handler Summary: Run the `dst sync disk` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--dataset` | `dataset` | no |  |  |  |
| `--tag` | `tag` | no |  |  |  |
| `--dry-run` | `dry_run` | no |  | False |  |
| `--corrections-field` | `corrections_field` | no |  |  |  |
| `--label-to-class-id` | `label_to_class_id` | no |  |  | JSON object e.g. {"rodent":0} |
| `--default-class-id` | `default_class_id` | no |  |  |  |
| `--path-replacement` | `path_replacement` | no |  |  | Replacement rule SRC=DST, can be repeated |
| `--backup-suffix-format` | `backup_suffix_format` | no |  |  |  |

### `workflow`

- Parser prog: `dst workflow`
- Help: `./dst workflow --help`

**Subcommands**

- `roundtrip`
- `tags`

#### `workflow roundtrip`

- Parser prog: `dst workflow roundtrip`
- Help: `./dst workflow roundtrip --help`
- Handler: `cmd_workflow_roundtrip`
- Handler Summary: Run the `dst workflow roundtrip` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--dataset` | `dataset` | yes |  |  |  |
| `--tag` | `tag` | no |  | fix |  |
| `--project` | `project` | no |  |  |  |
| `--label-field` | `label_field` | no |  | ground_truth |  |
| `--corrections-field` | `corrections_field` | no |  | ls_corrections |  |
| `--skip-send` | `skip_send` | no |  | False |  |
| `--skip-pull` | `skip_pull` | no |  | False |  |
| `--skip-sync-disk` | `skip_sync_disk` | no |  | False |  |
| `--dry-run-sync` | `dry_run_sync` | no |  | False |  |
| `--clear-project-tasks` | `clear_project_tasks` | no |  | False |  |
| `--upload-strategy` | `upload_strategy` | no | annotate_batched, sdk_batched |  |  |
| `--pull-strategy` | `pull_strategy` | no | sdk_meta, annotate_run |  |  |
| `--annotation-key` | `annotation_key` | no |  |  |  |
| `--launch-editor` | `launch_editor` | no |  | False |  |
| `--create-if-missing` | `create_if_missing` | no |  | False |  |
| `--strict-preflight` | `strict_preflight` | no |  | True |  |
| `--no-strict-preflight` | `strict_preflight` | no |  | True |  |
| `--overwrite-annotation-run` | `overwrite_annotation_run` | no |  | True |  |
| `--no-overwrite-annotation-run` | `overwrite_annotation_run` | no |  | True |  |
| `--send-params` | `send_params` | no |  |  | JSON object merged into send params |
| `--pull-params` | `pull_params` | no |  |  | JSON object merged into pull params |
| `--sync-params` | `sync_params` | no |  |  | JSON object merged into sync params |
| `--quiet-logs` | `quiet_logs` | no |  | False | Suppress noisy stdout/stderr logs from underlying workflow operations |
| `--output-json` | `output_json` | no |  |  | Write workflow result payload to a JSON file |

#### `workflow tags`

- Parser prog: `dst workflow tags`
- Help: `./dst workflow tags --help`

**Subcommands**

- `inline`
- `run`

##### `workflow tags inline`

- Parser prog: `dst workflow tags inline`
- Help: `./dst workflow tags inline --help`
- Handler: `cmd_workflow_tags_inline`
- Handler Summary: Run the `dst workflow tags inline` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--dataset` | `dataset` | yes |  |  |  |
| `--rule` | `rule` | yes |  |  | JSON object, e.g. {"tag":"delete","operation":"delete_samples"} |
| `--no-fail-fast` | `no_fail_fast` | no |  | False |  |
| `--quiet-logs` | `quiet_logs` | no |  | False | Suppress noisy stdout/stderr logs from underlying workflow operations |
| `--output-json` | `output_json` | no |  |  | Write workflow result payload to a JSON file |

##### `workflow tags run`

- Parser prog: `dst workflow tags run`
- Help: `./dst workflow tags run --help`
- Handler: `cmd_workflow_tags_run`
- Handler Summary: Run the `dst workflow tags run` command handler and return a JSON-serializable result.

**Options**

| Flags | Dest | Required | Choices | Default | Help |
|---|---|---|---|---|---|
| `--config` | `config` | no |  |  | Path to local config JSON (otherwise DST_CONFIG_PATH, package-local file, or ~/.config/dst/local_config.json) |
| `--overrides` | `overrides` | no |  |  | JSON object with runtime config overrides |
| `--workflow` | `workflow` | yes |  |  |  |
| `--dataset` | `dataset` | no |  |  | Override dataset_name from workflow file |
| `--fail-fast` | `fail_fast` | no |  |  |  |
| `--no-fail-fast` | `fail_fast` | no |  | True |  |
| `--quiet-logs` | `quiet_logs` | no |  | False | Suppress noisy stdout/stderr logs from underlying workflow operations |
| `--output-json` | `output_json` | no |  |  | Write workflow result payload to a JSON file |

