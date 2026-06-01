#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$ROOT_DIR/packages"
PY_PACKAGE_DIR="$PACKAGE_ROOT/python"
ULTRALYTICS_VERSION="${ULTRALYTICS_VERSION:-8.4.43}"
TORCHVISION_GIT_REF="${TORCHVISION_GIT_REF:-v0.20.0}"
TORCHVISION_CUDA_ARCH_LIST="${TORCHVISION_CUDA_ARCH_LIST:-8.7}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
PIP_RETRIES="${PIP_RETRIES:-2}"
PIP_TIMEOUT="${PIP_TIMEOUT:-30}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
INSTALL_PYCUDA="${INSTALL_PYCUDA:-1}"
PYCUDA_VERSION="${PYCUDA_VERSION:-2026.1}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export PIP_INDEX_URL PIP_EXTRA_INDEX_URL

pick_latest_file() {
  local pattern="$1"
  shift
  local dir
  for dir in "$@"; do
    [[ -d "$dir" ]] || continue
    local candidate
    candidate="$(find "$dir" -maxdepth 1 -type f -name "$pattern" | sort | tail -n 1 || true)"
    if [[ -n "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

torchvision_nms_ok() {
  python3 - <<'PY'
import sys
try:
    import torch
    import torchvision
    from torchvision.ops import nms
    boxes = torch.empty((0, 4), dtype=torch.float32)
    scores = torch.empty((0,), dtype=torch.float32)
    nms(boxes, scores, 0.5)
    print("[deps] torch=", torch.__version__)
    print("[deps] torchvision=", torchvision.__version__)
    print("[deps] torchvision_nms=ok")
except Exception as exc:
    print("[deps] torchvision_nms=failed:", repr(exc))
    sys.exit(1)
PY
}

install_local_wheel() {
  local package="$1"
  local pattern="$2"
  local wheel
  wheel="$(pick_latest_file "$pattern" "$PY_PACKAGE_DIR" || true)"
  if [[ -z "$wheel" ]]; then
    echo "[deps] $package wheel not found under $PY_PACKAGE_DIR"
    return 1
  fi
  echo "[deps] installing $package from local wheel: $wheel"
  python3 -m pip install --no-cache-dir --no-deps --no-index --find-links "$PY_PACKAGE_DIR" "$wheel"
}

python3 -m pip config unset global.index-url || true
python3 -m pip config unset global.extra-index-url || true

# NVIDIA Jetson torch wheels are installed without dependency resolution to
# avoid replacing the vendor torch stack, but torch imports these runtime helpers.
python3 -m pip install --no-cache-dir --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" \
  "typing-extensions>=4.8.0" \
  "sympy"

ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" ]]; then
  if python3 - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch") else 1)
PY
  then
    echo "[deps] torch already installed"
  else
    install_local_wheel "torch" 'torch-*.whl'
  fi

  if python3 - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torchaudio") else 1)
PY
  then
    echo "[deps] torchaudio already installed"
  else
    install_local_wheel "torchaudio" 'torchaudio-*.whl'
  fi

  if torchvision_nms_ok; then
    echo "[deps] torchvision already matches current torch"
  else
    python3 -m pip uninstall -y torchvision || true
    if install_local_wheel "torchvision" 'torchvision-*.whl'; then
      torchvision_nms_ok
    else
      echo "[deps] building torchvision from pytorch/vision ${TORCHVISION_GIT_REF}"
      MAX_JOBS="${MAX_JOBS:-2}" \
      FORCE_CUDA=1 \
      TORCH_CUDA_ARCH_LIST="$TORCHVISION_CUDA_ARCH_LIST" \
      python3 -m pip install --no-deps --no-build-isolation --force-reinstall \
        "git+https://github.com/pytorch/vision.git@${TORCHVISION_GIT_REF}"
      torchvision_nms_ok
    fi
  fi
else
  echo "[deps] installing torch stack from pip (x86_64 path)"
  python3 -m pip install --no-cache-dir torch torchvision torchaudio
fi

if [[ "$UPGRADE_PIP" == "1" ]]; then
  python3 -m pip install --no-cache-dir --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" "pip<24.1"
else
  echo "[deps] skip pip upgrade (UPGRADE_PIP=0)"
fi

python3 -m pip install --no-cache-dir --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" -r "$ROOT_DIR/requirements.common.txt"
python3 -m pip install --no-cache-dir --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" --no-deps "ultralytics==${ULTRALYTICS_VERSION}"

if [[ "$INSTALL_PYCUDA" == "1" ]]; then
  if python3 - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("pycuda") else 1)
PY
  then
    echo "[deps] pycuda already installed"
  else
    echo "[deps] installing pycuda==${PYCUDA_VERSION}"
    python3 -m pip install --no-cache-dir --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" "pycuda==${PYCUDA_VERSION}"
  fi

  python3 - <<'PY'
import importlib.util
import pycuda
import pycuda.driver as cuda
print("[deps] pycuda=", pycuda.VERSION_TEXT)
print("[deps] pycuda_driver_import=ok")
ok = importlib.util.find_spec("tensorrt") is not None
print("[deps] tensorrt_present=", ok)
assert ok, "tensorrt package not found"
PY
else
  echo "[deps] skip pycuda install (INSTALL_PYCUDA=0)"
fi

# If pip OpenCV overrides Jetson base OpenCV and breaks import, remove pip wheels.
if ! python3 - <<'PY'
import cv2
print('[deps] cv2=', cv2.__version__)
PY
then
  echo "[deps] cv2 import failed, removing pip OpenCV packages to fall back to base image OpenCV"
  python3 -m pip uninstall -y opencv-python opencv-python-headless || true
  python3 - <<'PY'
import cv2
print('[deps] cv2 recovered=', cv2.__version__)
PY
fi

python3 - <<'PY'
import importlib.util
ok = importlib.util.find_spec('ultralytics') is not None
print('[deps] ultralytics_present=', ok)
assert ok, 'ultralytics package not found after install'
PY

PROCESS_TAP_WHL="$(pick_latest_file 'multimodal_process_tap-*.whl' "$PY_PACKAGE_DIR" || true)"
if [[ -n "$PROCESS_TAP_WHL" ]]; then
  echo "[deps] installing process_tap from local wheel: $PROCESS_TAP_WHL"
  python3 -m pip install --no-cache-dir --no-deps "$PROCESS_TAP_WHL"
  python3 - <<'PY'
import importlib.util
import inspect
ok = importlib.util.find_spec('process_tap') is not None
print('[deps] process_tap_present=', ok)
assert ok, 'process_tap package not found after install'
from process_tap.integrations import maybe_create_asd_stream_capture
source = inspect.getsource(maybe_create_asd_stream_capture)
assert 'ASD_STREAM_CAPTURE_WRITE_PREVIEW_VIDEO' in source, 'stale process_tap wheel installed'
print('[deps] process_tap_capture_preview_switch=present')
PY
else
  echo "[deps] process_tap wheel not found under $PY_PACKAGE_DIR, skip"
fi

ASD_SDK_WHL="$(pick_latest_file 'asd_sdk-*.whl' "$PY_PACKAGE_DIR" "$ROOT_DIR/sdk/dist" || true)"
if [[ -n "$ASD_SDK_WHL" ]]; then
  echo "[deps] installing asd_sdk from local wheel: $ASD_SDK_WHL"
  python3 -m pip install --no-cache-dir --no-deps "$ASD_SDK_WHL"
  python3 - <<'PY'
import asd_sdk
print('[deps] asd_sdk_present=', hasattr(asd_sdk, 'ASDSDK'))
assert hasattr(asd_sdk, 'ASDSDK'), 'asd_sdk package not found after install'
PY
else
  echo "[deps] asd_sdk wheel not found under $PY_PACKAGE_DIR or $ROOT_DIR/sdk/dist, skip"
fi

echo "[deps] python dependencies installed"
