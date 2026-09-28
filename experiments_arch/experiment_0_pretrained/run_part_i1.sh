#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NANOTAB_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${NANOTAB_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-experiments_arch/experiment_0_pretrained/default_config.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

exec "${PYTHON_BIN}" -m experiments_arch.experiment_0_pretrained run-part-i1 \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  "$@"
