#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset_name> [port] [address]"
  exit 1
fi

DATASET_NAME="$1"
PORT="${2:-5151}"
ADDRESS="${3:-0.0.0.0}"

"$(dirname "$0")/dst" app open --dataset "$DATASET_NAME" --port "$PORT" --address "$ADDRESS"
