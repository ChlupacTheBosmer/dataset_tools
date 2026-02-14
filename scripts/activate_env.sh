#!/usr/bin/env bash
# shellcheck shell=bash

# Activate a Python environment for dst commands if one is discoverable.
# Priority:
# 1) DST_ACTIVATE_SCRIPT env var
# 2) repo-local .venv
# 3) common workspace helper paths
# If nothing is found, execution continues in the current shell env.

dst_activate_python_env() {
  if [[ "${DST_SKIP_ACTIVATE:-}" == "1" ]]; then
    return 0
  fi

  local script_dir root_dir candidate
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root_dir="$(cd "$script_dir/.." && pwd)"

  local -a candidates=()
  if [[ -n "${DST_ACTIVATE_SCRIPT:-}" ]]; then
    candidates+=("$DST_ACTIVATE_SCRIPT")
  fi

  candidates+=(
    "$root_dir/.venv/bin/activate"
    "$HOME/workspace/activate_all.sh"
    "$HOME/activate_all.sh"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate" >/dev/null
      return 0
    fi
  done

  return 0
}
