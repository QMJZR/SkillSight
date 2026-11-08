#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for the SkillSight end-to-end eval.
#
# Idempotent: if a server started by this script is still running (its PID file
# points at a live process answering /models), it no-ops. The eval script calls
# this before waiting on the endpoint; you can also run it standalone.
#
# Override any of the vars below from the environment, e.g.:
#   VLLM_GPU=5 VLLM_MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 bash vllm.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- config (env-overridable) ------------------------------------------------
VLLM_ENV="${VLLM_ENV:-vllm}"                                   # conda env name
VLLM_PY="${VLLM_PY:-}"                                         # explicit interpreter override
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/path/to/Qwen3-4B-Instruct-2507}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3-4B-Instruct-2507}" # must match --model
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_GPU="${VLLM_GPU:-4}"                                       # keep off encoder's cuda:3
VLLM_MAX_LEN="${VLLM_MAX_LEN:-32768}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"

LOG_DIR="${HERE}/logs"
PID_FILE="${LOG_DIR}/vllm_${VLLM_PORT}.pid"
LOG_FILE="${LOG_DIR}/vllm_${VLLM_PORT}.log"
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

mkdir -p "${LOG_DIR}"

# --- idempotency: already up? ------------------------------------------------
if curl -s -m 3 "${BASE_URL}/models" >/dev/null 2>&1; then
    echo "[vllm.sh] server already answering at ${BASE_URL}; nothing to do."
    exit 0
fi
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[vllm.sh] pid $(cat "${PID_FILE}") alive but endpoint not ready yet; leaving it."
    exit 0
fi

# --- resolve conda python ----------------------------------------------------
if [[ -z "${VLLM_PY}" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/conda)"
    VLLM_PY="${CONDA_BASE}/envs/${VLLM_ENV}/bin/python"
fi
if [[ ! -x "${VLLM_PY}" ]]; then
    echo "[vllm.sh] ERROR: python not found in conda env '${VLLM_ENV}' (${VLLM_PY})." >&2
    exit 1
fi
VLLM_BIN_DIR="$(dirname "${VLLM_PY}")"
export PATH="${VLLM_BIN_DIR}:${PATH}"

echo "[vllm.sh] launching vLLM:"
echo "          env=${VLLM_ENV}  gpu=${VLLM_GPU}  port=${VLLM_PORT}"
echo "          model=${VLLM_MODEL_PATH}  served-name=${VLLM_SERVED_NAME}"
echo "          log=${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" nohup "${VLLM_PY}" -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL_PATH}" \
    --served-model-name "${VLLM_SERVED_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_LEN}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
    --trust-remote-code \
    > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[vllm.sh] started pid $(cat "${PID_FILE}"); tail -f ${LOG_FILE} to watch startup."
