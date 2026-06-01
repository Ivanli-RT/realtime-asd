#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "[error] Neither 'docker compose' nor 'docker-compose' is available on host." >&2
  exit 1
fi

echo "[host] project=$ROOT_DIR"
echo "[host] compose_cmd=${COMPOSE[*]}"

"${COMPOSE[@]}" run --rm asd_runtime bash -lc '
set -euo pipefail

echo "[container] whoami=$(whoami)"
echo "[container] pwd=$(pwd)"
python3 -c "import sys; print(\"[container] python=\", sys.version.split()[0])"
python3 -c "import numpy as np; print(\"[container] numpy=\", np.__version__)"
python3 -c "import torchaudio; print(\"[container] torchaudio=\", torchaudio.__version__)"
python3 -c "import torch; print(\"[container] cuda=\", torch.cuda.is_available(), torch.cuda.device_count())"
python3 -c "from asd.model_wrapper import TalkNetASDModel; print(\"[container] model_wrapper_import_ok\")"
'

echo "[host] container check passed"
