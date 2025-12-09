#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

echo
echo "Virtual environment ready. Activate it with:"
echo "  source ${VENV_PATH}/bin/activate"
