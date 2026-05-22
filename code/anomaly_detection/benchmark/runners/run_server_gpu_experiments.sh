#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-code/anomaly_detection/datasets/VT-Tiny-MOT}"
GPU_ID="${GPU_ID:-0}"
MODE="${MODE:-smoke}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-code/anomaly_detection/benchmark/outputs/protocol_${MODE}}"
RESULT_ROOT="${RESULT_ROOT:-code/anomaly_detection/benchmark/outputs/results_${MODE}}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-code/anomaly_detection/benchmark/outputs/official_${MODE}}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs}"
TMUX_SESSION="${TMUX_SESSION:-fusiontrack_${MODE}_${RUN_TAG}}"
USE_TMUX="${USE_TMUX:-0}"

if [[ "${USE_TMUX}" == "1" && -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[tmux] tmux not found; install tmux or rerun with USE_TMUX=0" >&2
    exit 1
  fi
  mkdir -p "${LOG_ROOT}"
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${TMUX_SESSION}.log}"
  ENV_ARGS=(
    env
    USE_TMUX=0
    RUN_TAG="${RUN_TAG}"
    LOG_ROOT="${LOG_ROOT}"
    LOG_FILE="${LOG_FILE}"
    DATA_ROOT="${DATA_ROOT}"
    GPU_ID="${GPU_ID}"
    MODE="${MODE}"
    SOURCE_SPLIT="${SOURCE_SPLIT}"
    OUTPUT_ROOT="${OUTPUT_ROOT}"
    RESULT_ROOT="${RESULT_ROOT}"
    OFFICIAL_ROOT="${OFFICIAL_ROOT}"
    SEED="${SEED}"
    PYTHON_BIN="${PYTHON_BIN}"
    bash "${SCRIPT_PATH}"
  )
  printf -v RUN_COMMAND "%q " "${ENV_ARGS[@]}"
  printf -v WORKDIR "%q" "${PWD}"
  printf -v QUOTED_LOG "%q" "${LOG_FILE}"
  tmux new-session -d -s "${TMUX_SESSION}" "cd ${WORKDIR} && ${RUN_COMMAND} 2>&1 | tee -a ${QUOTED_LOG}"
  echo "[tmux] started session: ${TMUX_SESSION}"
  echo "[tmux] attach: tmux attach -t ${TMUX_SESSION}"
  echo "[tmux] log: ${LOG_FILE}"
  exit 0
fi

if [[ -n "${LOG_FILE:-}" ]]; then
  echo "[log] ${LOG_FILE}"
fi

if [[ "${MODE}" == "smoke" ]]; then
  SMOKE_ARGS=(--smoke-max-train 80 --smoke-max-val 80)
else
  SMOKE_ARGS=()
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[gpu] nvidia-smi not found" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}")
print({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to run GPU experiment")
PY

"${PYTHON_BIN}" code/anomaly_detection/benchmark/runners/prepare_vt_tiny_mot_protocol.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-split "${SOURCE_SPLIT}" \
  --seed "${SEED}" \
  "${SMOKE_ARGS[@]}"

"${PYTHON_BIN}" code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py \
  --config-json "${OUTPUT_ROOT}/individual_val_matrix.json" \
  --output-dir "${RESULT_ROOT}/individual"

"${PYTHON_BIN}" code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py \
  --config-json "${OUTPUT_ROOT}/group_val_matrix.json" \
  --output-dir "${RESULT_ROOT}/group"

"${PYTHON_BIN}" code/anomaly_detection/benchmark/runners/export_report_tables.py \
  --summary-csv "${RESULT_ROOT}/individual/summary.csv" \
  --output-dir "${RESULT_ROOT}/individual/tables"

"${PYTHON_BIN}" code/anomaly_detection/benchmark/runners/export_report_tables.py \
  --summary-csv "${RESULT_ROOT}/group/summary.csv" \
  --output-dir "${RESULT_ROOT}/group/tables"

mkdir -p "${OFFICIAL_ROOT}/cetrajad" "${OFFICIAL_ROOT}/lmtad" "${OFFICIAL_ROOT}/pidpm"

"${PYTHON_BIN}" code/anomaly_detection/benchmark/runners/prepare_cetrajad_official_inputs.py \
  --trajectory-jsonl "${OUTPUT_ROOT}/fused_trajectories_val.jsonl" \
  --output-dir "${OFFICIAL_ROOT}/cetrajad"

"${PYTHON_BIN}" code/anomaly_detection/benchmark/runners/prepare_lmtad_official_inputs.py \
  --trajectory-jsonl "${OUTPUT_ROOT}/fused_trajectories_val.jsonl" \
  --output-dir "${OFFICIAL_ROOT}/lmtad"

"${PYTHON_BIN}" code/anomaly_detection/benchmark/runners/prepare_pidpm_official_inputs.py \
  --trajectory-jsonl "${OUTPUT_ROOT}/fused_trajectories_val.jsonl" \
  --output-csv "${OFFICIAL_ROOT}/pidpm/fusiontrack_val_pidpm.csv" \
  --sidecar-json "${OFFICIAL_ROOT}/pidpm/fusiontrack_val_pidpm_sidecar.json"

echo "[done] protocol=${OUTPUT_ROOT}"
echo "[done] results=${RESULT_ROOT}"
echo "[done] official_inputs=${OFFICIAL_ROOT}"
