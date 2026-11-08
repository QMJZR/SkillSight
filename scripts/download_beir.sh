#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-./BEIR}"
DATASETS=("arguana" "fiqa" "scifact")
BASE_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

mkdir -p "${ROOT}"

for name in "${DATASETS[@]}"; do
    archive="${ROOT}/${name}.zip"
    if [[ -f "${archive}" ]] && python3 -m zipfile -t "${archive}" >/dev/null 2>&1; then
        echo "[cached] ${name}"
    else
        echo "[download] ${name}"
        curl -L --fail --retry 5 --retry-delay 2 --continue-at - \
            -o "${archive}" "${BASE_URL}/${name}.zip"
    fi
    python3 -m zipfile -e "${archive}" "${ROOT}"
done

echo "[done] BEIR datasets saved under ${ROOT}"
