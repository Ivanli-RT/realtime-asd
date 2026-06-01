#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/asd_runtime_slim"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-0}"
ROS_DISTRO="${ROS_DISTRO:-noetic}"
ROS_ROOT="${ROS_ROOT:-/opt/ros/$ROS_DISTRO}"
ROS_INSTALL_ROOT="$ROS_ROOT"
AUDIO_PY_MODULE_PATH="$ROS_INSTALL_ROOT/lib/python3/dist-packages/audio/msg/_AudioData.py"
ROS_SETUP_PATH="${ROS_SETUP_PATH:-/ros_noetic/catkin_ws/devel/setup.bash}"
ROS_FALLBACK_SETUP_PATH="${ROS_FALLBACK_SETUP_PATH:-$ROS_INSTALL_ROOT/setup.bash}"
ASD_HOST_CUDNN_SOURCE_DIR="${ASD_HOST_CUDNN_SOURCE_DIR:-/host/usr/lib/aarch64-linux-gnu}"
ASD_EXTRA_CUDA_RUNTIME_SOURCE_DIRS="${ASD_EXTRA_CUDA_RUNTIME_SOURCE_DIRS:-$ROOT_DIR/packages/system/cuda_runtime_libs}"
ASD_HOST_CUDNN_LINK_DIR="${ASD_HOST_CUDNN_LINK_DIR:-/usr/local/lib/asd-host-cudnn}"

source_ros_setup() {
  local setup_path
  local setup_dir
  for setup_path in "$ROS_SETUP_PATH" "$ROS_FALLBACK_SETUP_PATH" "$ROS_ROOT/setup.bash"; do
    [[ -n "$setup_path" && -f "$setup_path" ]] || continue
    setup_dir="$(dirname "$setup_path")"
    if [[ "$(basename "$setup_path")" == "setup.bash" && ! -f "$setup_dir/setup.sh" ]]; then
      echo "[entrypoint] skip ROS setup with missing companion setup.sh: $setup_path"
      continue
    fi
    echo "[entrypoint] source ROS setup: $setup_path"
    set +u
    source "$setup_path"
    set -u
    return 0
  done
  return 1
}

setup_host_cuda_runtime_libs() {
  local source_dirs=()
  local extra_dirs=()
  local source_dir

  IFS=':' read -r -a source_dirs <<< "${ASD_HOST_CUDNN_SOURCE_DIR:-}"
  IFS=':' read -r -a extra_dirs <<< "${ASD_EXTRA_CUDA_RUNTIME_SOURCE_DIRS:-}"

  for source_dir in "${source_dirs[@]}" "${extra_dirs[@]}"; do
    [[ -n "$source_dir" && -d "$source_dir" ]] || continue

    shopt -s nullglob
    local runtime_libs=(
      "$source_dir"/libcudnn*.so*
      "$source_dir"/libcusparseLt*.so*
    )
    shopt -u nullglob
    [[ "${#runtime_libs[@]}" -gt 0 ]] || continue

    mkdir -p "$ASD_HOST_CUDNN_LINK_DIR"
    local lib
    for lib in "${runtime_libs[@]}"; do
      ln -sfn "$lib" "$ASD_HOST_CUDNN_LINK_DIR/$(basename "$lib")"
    done
    echo "[entrypoint] linked host CUDA runtime libs from $source_dir"
  done

  export LD_LIBRARY_PATH="$ASD_HOST_CUDNN_LINK_DIR:${LD_LIBRARY_PATH:-}"
}

setup_ros_root_paths() {
  if [[ -d "$ROS_INSTALL_ROOT/lib/python3/dist-packages" ]]; then
    export PYTHONPATH="$ROS_INSTALL_ROOT/lib/python3/dist-packages:${PYTHONPATH:-}"
  fi
  if [[ -d "$ROS_INSTALL_ROOT/share" ]]; then
    export ROS_PACKAGE_PATH="$ROS_INSTALL_ROOT/share:${ROS_PACKAGE_PATH:-}"
  fi
  if [[ -d "$ROS_INSTALL_ROOT/lib" ]]; then
    export LD_LIBRARY_PATH="$ROS_INSTALL_ROOT/lib:${LD_LIBRARY_PATH:-}"
  fi
}

setup_host_cuda_runtime_libs

if [[ -d "$ROOT_DIR/scripts" ]]; then
  chmod +x "$ROOT_DIR"/scripts/*.sh || true
fi

if [[ -x "$ROOT_DIR/scripts/fix_gstreamer.sh" ]]; then
  "$ROOT_DIR/scripts/fix_gstreamer.sh" || true
fi

if [[ "$AUTO_INSTALL_DEPS" = "1" ]]; then
  echo "[entrypoint] AUTO_INSTALL_DEPS=1, running dependency bootstrap"
  if [[ -x "$ROOT_DIR/scripts/install_python_deps.sh" ]]; then
    "$ROOT_DIR/scripts/install_python_deps.sh"
    touch "$ROOT_DIR/.deps_installed"
  else
    echo "[entrypoint] install_python_deps.sh not found, skip deps bootstrap"
  fi
else
  echo "[entrypoint] skip runtime deps install (expect deps baked into image)"
fi

if ! source_ros_setup; then
  echo "[entrypoint] ROS setup not found: $ROS_SETUP_PATH"
  echo "[entrypoint] candidate setup files:"
  find /opt /usr/local /workspace /ros_noetic -maxdepth 6 -type f \( -name setup.bash -o -name local_setup.bash \) 2>/dev/null | sort || true
  echo "[entrypoint] rostopic path:"
  command -v rostopic || true
fi

setup_ros_root_paths

if python3 -c "from audio.msg import AudioData" >/dev/null 2>&1; then
  echo "[entrypoint] audio message package import ok: audio/AudioData"
elif [[ -f "$AUDIO_PY_MODULE_PATH" ]]; then
  echo "[entrypoint] audio message package file exists but import failed: $AUDIO_PY_MODULE_PATH"
else
  echo "[entrypoint] audio message package not found in image; rebuild with packages/system/zj_humanoid_types*.run if audio topic resolution fails"
fi

if [[ -f "$ROOT_DIR/scripts/setup_ros_network.sh" ]]; then
  source "$ROOT_DIR/scripts/setup_ros_network.sh"
  setup_ros_network_env
fi

exec "$@"
