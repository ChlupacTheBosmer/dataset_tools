#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/activate_env.sh"
dst_activate_python_env
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

export RUN_REAL_LIFE_TESTS=1
export RUN_REAL_LIFE_NETWORK_TESTS="${RUN_REAL_LIFE_NETWORK_TESTS:-1}"

echo "Running real-life integration suite"
echo "RUN_REAL_LIFE_TESTS=$RUN_REAL_LIFE_TESTS"
echo "RUN_REAL_LIFE_NETWORK_TESTS=$RUN_REAL_LIFE_NETWORK_TESTS"

python -m unittest tests.test_real_life_suite -v
