#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/root/envs/vllm310}"
CONDA_BIN="${CONDA_BIN:-/usr/local/miniconda3/bin/conda}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/cache/conda}"
export CONDA_PKGS_DIRS

if ! command -v g++ >/dev/null 2>&1; then
    apt-get update
    apt-get install -y g++
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_BIN}" create -p "${ENV_PREFIX}" python=3.10 pip -y
fi

"${ENV_PREFIX}/bin/python" -m pip install --upgrade pip
"${ENV_PREFIX}/bin/python" -m pip install vllm
"${ENV_PREFIX}/bin/python" -c "import vllm; print('vLLM', vllm.__version__)"
