#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/activate_env.sh"
dst_activate_python_env
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/5] Verifying package versions"
python - <<'PY'
import importlib.metadata as md

for pkg in ("fiftyone", "fiftyone-brain", "label-studio-sdk"):
    print(f"{pkg}=={md.version(pkg)}")
PY

echo "[2/5] CLI sanity"
./dst --help >/dev/null
./dst metrics --help >/dev/null
./dst workflow --help >/dev/null
./dst brain --help >/dev/null

echo "[3/5] FiftyOne API contract tests"
python -m unittest tests.test_fiftyone_contracts -v

echo "[4/5] Full unit test suite"
python -m unittest discover -s tests -p 'test_*.py'

echo "[5/5] Smoke metrics run on throwaway dataset"
python - <<'PY'
import os
import tempfile
import time

import fiftyone as fo
from PIL import Image

from dataset_tools.metrics import MistakennessComputation, UniquenessComputation

name = f"dst_precode_smoke_{int(time.time())}"
tmpdir = tempfile.mkdtemp(prefix="dst_smoke_")

samples = []
for idx in range(3):
    filepath = os.path.join(tmpdir, f"img_{idx}.jpg")
    Image.new("RGB", (64, 64), color=(idx * 40, 20, 20)).save(filepath)

    sample = fo.Sample(filepath=filepath)
    sample["ground_truth"] = fo.Detections(
        detections=[fo.Detection(label="rodent", bounding_box=[0.1, 0.1, 0.3, 0.3])]
    )
    sample["predictions"] = fo.Detections(
        detections=[
            fo.Detection(
                label="rodent",
                bounding_box=[0.12, 0.1, 0.29, 0.31],
                confidence=0.85,
            )
        ]
    )
    sample["emb_smoke"] = [float(idx), float(idx + 1), float(idx + 2), float(idx + 3)]
    samples.append(sample)

dataset = fo.Dataset(name)
dataset.add_samples(samples)
dataset.persistent = True

UniquenessComputation(
    dataset_name=name,
    embeddings_field="emb_smoke",
    output_field="uniq_smoke",
).run()

MistakennessComputation(
    dataset_name=name,
    pred_field="predictions",
    gt_field="ground_truth",
    mistakenness_field="mist_smoke",
    missing_field="miss_smoke",
    spurious_field="spur_smoke",
).run()

dataset = fo.load_dataset(name)
uniq = dataset.values("uniq_smoke")
mist = dataset.values("mist_smoke")

assert sum(v is not None for v in uniq) == len(uniq)
assert sum(v is not None for v in mist) == len(mist)
assert dataset.has_sample_field("miss_smoke")
assert dataset.has_sample_field("spur_smoke")

fo.delete_dataset(name)
print(f"Smoke dataset completed and deleted: {name}")
PY

if [[ "${RUN_COVERAGE_REPORT:-0}" == "1" ]]; then
  echo "[6/6] Coverage report"
  ./scripts/run_coverage_report.sh
fi

echo "Pre-coding gate passed"
