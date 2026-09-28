#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "v1_0" && "$1" != "v1_1" && "$1" != "v1_3" ) ]]; then
  echo "usage: $0 {v1_0|v1_1|v1_3}" >&2
  exit 2
fi

VERSION=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_ROOT="${ENV_ROOT:-${REPO_ROOT}/.tabdpt-version-envs}"
CONFIG="$REPO_ROOT/experiments_arch/experiment_0_pretrained/tabdpt_versions_config.json"
ARTIFACT_ROOT="$REPO_ROOT/artifacts/experiment_0_tabdpt_versions"
INDEX_ROOT="$ARTIFACT_ROOT/data_indexes"
OUTPUT_ROOT="$ARTIFACT_ROOT/$VERSION"
LOG_DIR="$ARTIFACT_ROOT/logs"
CHECKPOINT_DIR=checkpoints

case "$VERSION" in
  v1_0)
    PYTHON="$ENV_ROOT/v1_0/bin/python"
    MODEL=tabdpt_v1_0_retrieval_off
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    ;;
  v1_1)
    PYTHON="$ENV_ROOT/v1_1/bin/python"
    MODEL=tabdpt_v1_1_retrieval_off
    ;;
  v1_3)
    PYTHON="$ENV_ROOT/v1_3/bin/python"
    MODEL=tabdpt_v1_3_retrieval_off
    ;;
esac

FRIEDMAN_INDEX="$INDEX_ROOT/friedman350.csv"
OPENML_INDEX="$INDEX_ROOT/openml42.csv"
for required in "$PYTHON" "$CONFIG" "$FRIEDMAN_INDEX" "$OPENML_INDEX"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

# All console output is kept with the results, never in the repository root.
exec > >(tee -a "$LOG_DIR/tabdpt_${VERSION}_part_i.log") 2>&1

phase() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$1"
}

phase "$VERSION Part I.1: Friedman predictions"
"$PYTHON" -m experiments_arch.experiment_0_pretrained predict \
  --config "$CONFIG" \
  --data-index "$FRIEDMAN_INDEX" \
  --out-dir "$OUTPUT_ROOT/friedman" \
  --models "$MODEL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --device cuda

phase "$VERSION Part I.1: Friedman evaluation"
"$PYTHON" -m experiments_arch.experiment_0_pretrained \
  evaluate-friedman-sweeps \
  --config "$CONFIG" \
  --predictions-dir "$OUTPUT_ROOT/friedman/predictions" \
  --out-dir "$OUTPUT_ROOT/friedman"

phase "$VERSION Part I.1: OpenML predictions"
"$PYTHON" -m experiments_arch.experiment_0_pretrained predict \
  --config "$CONFIG" \
  --data-index "$OPENML_INDEX" \
  --out-dir "$OUTPUT_ROOT/openml" \
  --models "$MODEL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --device cuda

phase "$VERSION Part I.1: OpenML evaluation"
"$PYTHON" -m experiments_arch.experiment_0_pretrained evaluate \
  --predictions-dir "$OUTPUT_ROOT/openml/predictions" \
  --out-dir "$OUTPUT_ROOT/openml"

phase "$VERSION Part I.2: Friedman sweeps"
"$PYTHON" -m experiments_arch.experiment_0_pretrained \
  run-part-i2-friedman-sweeps \
  --config "$CONFIG" \
  --data-index "$FRIEDMAN_INDEX" \
  --out-dir "$OUTPUT_ROOT/friedman" \
  --models "$MODEL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --device cuda

phase "$VERSION Part I.2: OpenML"
"$PYTHON" -m experiments_arch.experiment_0_pretrained run-part-i2 \
  --config "$CONFIG" \
  --data-index "$OPENML_INDEX" \
  --out-dir "$OUTPUT_ROOT/openml" \
  --models "$MODEL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --device cuda

phase "$VERSION complete"
touch "$OUTPUT_ROOT/COMPLETE"
