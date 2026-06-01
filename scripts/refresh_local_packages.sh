#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$ROOT_DIR/packages}"
PACKAGE_PY_DIR="$PACKAGE_ROOT/python"
PACKAGE_SYSTEM_DIR="$PACKAGE_ROOT/system"
PACKAGE_IMAGE_DIR="$PACKAGE_ROOT/images"
LEGACY_WHEEL_DIR="$ROOT_DIR/third_party_wheels"
PROCESS_TAP_SRC="${PROCESS_TAP_SRC:-/home/naviai/Desktop/multimodal_process_tap}"
SDK_DIR="${SDK_DIR:-$ROOT_DIR/sdk}"
SAVE_BASE_IMAGE="${SAVE_BASE_IMAGE:-0}"

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

ensure_build_module() {
  if python3 -m build --version >/dev/null 2>&1; then
    return 0
  fi
  echo "[packages] python build module not found. Install it first, for example:" >&2
  echo "  python3 -m pip install --upgrade build" >&2
  exit 1
}

copy_latest_artifact() {
  local pattern="$1"
  local dst_dir="$2"
  shift 2
  local src
  src="$(pick_latest_file "$pattern" "$@" || true)"
  if [[ -z "$src" ]]; then
    echo "[packages] no artifact matched $pattern"
    return 0
  fi

  mkdir -p "$dst_dir"
  cp -f "$src" "$dst_dir/"
  echo "[packages] staged $(basename "$src") -> $dst_dir"
}

mkdir -p "$PACKAGE_PY_DIR" "$PACKAGE_SYSTEM_DIR" "$PACKAGE_IMAGE_DIR"

copy_latest_artifact 'torch-*.whl' "$PACKAGE_PY_DIR" "$LEGACY_WHEEL_DIR"
copy_latest_artifact 'torchaudio-*.whl' "$PACKAGE_PY_DIR" "$LEGACY_WHEEL_DIR"
copy_latest_artifact 'torchvision-*.whl' "$PACKAGE_PY_DIR" "$LEGACY_WHEEL_DIR"
copy_latest_artifact 'onnxruntime_gpu-*.whl' "$PACKAGE_PY_DIR" "$LEGACY_WHEEL_DIR"
copy_latest_artifact 'zj-humanoid-ros-noetic-audio_*.deb' "$PACKAGE_SYSTEM_DIR" "$LEGACY_WHEEL_DIR"

if [[ -d "$PROCESS_TAP_SRC" ]]; then
  ensure_build_module
  echo "[packages] building multimodal_process_tap wheel from $PROCESS_TAP_SRC"
  (
    cd "$PROCESS_TAP_SRC"
    python3 -m build --wheel --no-isolation
  )
  copy_latest_artifact 'multimodal_process_tap-*.whl' "$PACKAGE_PY_DIR" "$PROCESS_TAP_SRC/dist"
else
  echo "[packages] process_tap source not found: $PROCESS_TAP_SRC"
fi

if [[ -d "$SDK_DIR" ]]; then
  ensure_build_module
  echo "[packages] building asd_sdk wheel from $SDK_DIR"
  (
    cd "$SDK_DIR"
    python3 -m build --wheel --no-isolation
  )
  copy_latest_artifact 'asd_sdk-*.whl' "$PACKAGE_PY_DIR" "$SDK_DIR/dist"
else
  echo "[packages] sdk source not found: $SDK_DIR"
fi

if [[ "$SAVE_BASE_IMAGE" = "1" ]]; then
  "$ROOT_DIR/scripts/save_base_image.sh"
else
  echo "[packages] skip base image archive (set SAVE_BASE_IMAGE=1 to save it)"
fi

echo "[packages] refresh complete. package root: $PACKAGE_ROOT"
