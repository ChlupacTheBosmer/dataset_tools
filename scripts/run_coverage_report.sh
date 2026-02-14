#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/activate_env.sh"
dst_activate_python_env
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if ! python - <<'PY' >/dev/null 2>&1
import coverage  # noqa: F401
PY
then
  echo "coverage package is not installed in the active environment."
  echo "Install it with: python -m pip install coverage"
  exit 2
fi

SOURCE="${COVERAGE_SOURCE:-dataset_tools}"
FAIL_UNDER="${COVERAGE_FAIL_UNDER:-}"
SHOW_MISSING="${COVERAGE_SHOW_MISSING:-1}"
GENERATE_HTML="${COVERAGE_HTML:-0}"

echo "Running coverage for source: $SOURCE"
python -m coverage erase
python -m coverage run --source="$SOURCE" -m unittest discover -s tests -p 'test_*.py'

REPORT_ARGS=()
if [[ "$SHOW_MISSING" == "1" ]]; then
  REPORT_ARGS+=("-m")
fi

if [[ -n "$FAIL_UNDER" ]]; then
  echo "Enforcing minimum coverage: ${FAIL_UNDER}%"
  REPORT_ARGS+=("--fail-under=$FAIL_UNDER")
else
  echo "No fail-under threshold configured; reporting only."
fi

python -m coverage report "${REPORT_ARGS[@]}"

if [[ "$GENERATE_HTML" == "1" ]]; then
  python -m coverage html
  echo "HTML report generated at: htmlcov/index.html"
fi
