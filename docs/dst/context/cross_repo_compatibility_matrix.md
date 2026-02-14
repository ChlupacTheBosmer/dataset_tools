# Cross-Repo Compatibility Matrix (`dst` + `od_training`)

This matrix defines the shared environment baseline when both repos run in one
venv (recommended under `external/.venv_shared`).

## Core Runtime

| Package | Constraint | Notes |
|---|---|---|
| `python` | `>=3.10` | shared baseline |
| `numpy` | `>=1.23.5,<2.0.0` | required by `ultralytics` compatibility |
| `scipy` | `<1.13.0` | retained for training stack compatibility |
| `pydantic` | `>=1.10,<3.0.0` | avoid major-version breakage |

## Dataset/Curation

| Package | Constraint | Notes |
|---|---|---|
| `fiftyone` | `>=1.11,<2.0` | shared DB/API surface |
| `fiftyone-brain` | `>=0.21,<1.0` | required by `dst` metrics/brain commands |
| `label-studio-sdk` | `==1.0.20` | pinned for `dst` LS workflow stability with NumPy `<2` |

## Training

| Package | Constraint | Notes |
|---|---|---|
| `torch` | `==2.7.1` | shared stable baseline across repos |
| `ultralytics` | `>=8.0` | YOLO train/infer |
| `rfdetr` | `rfdetr[metrics,plus]` | RF-DETR train/infer + extras |
| `roboflow` | `>=1.2.13,<1.3` | avoids old `pyparsing==2.4.7` transitive pin |
| `pyparsing` | `>=3,<4` | satisfies `matplotlib>=3` in FiftyOne stack |

## Validation Commands

```bash
dst --help
odt utility verify-env
python -m pip check
```
