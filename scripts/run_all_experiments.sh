#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$PROJECT_ROOT/.venv_linux/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m src.run_experiments --config configs/experiments.json --data-dir STL10 --outputs-dir outputs "$@"
