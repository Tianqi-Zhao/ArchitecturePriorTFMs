#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-python}"
ENV_ROOT="${ENV_ROOT:-${REPO_ROOT}/.tabdpt-version-envs}"

mkdir -p "$ENV_ROOT"
for item in "v1_1:1.1.14" "v1_0:0.1.0" "v1_3:1.3.0"; do
  name=${item%%:*}
  package_version=${item##*:}
  env_dir="$ENV_ROOT/$name"
  if [[ ! -x "$env_dir/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$env_dir"
  fi
  "$env_dir/bin/python" -m pip install --no-deps --upgrade "tabdpt==$package_version"
  "$env_dir/bin/python" -c 'from importlib.metadata import version; print(version("tabdpt"))'
done
