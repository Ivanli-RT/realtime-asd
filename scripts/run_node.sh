#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ROS_DISTRO="${ROS_DISTRO:-noetic}"
ROS_ROOT="${ROS_ROOT:-/opt/ros/$ROS_DISTRO}"
ROS_SETUP_PATH="${ROS_SETUP_PATH:-/ros_noetic/catkin_ws/devel/setup.bash}"
ROS_FALLBACK_SETUP_PATH="${ROS_FALLBACK_SETUP_PATH:-$ROS_ROOT/setup.bash}"

for setup_path in "$ROS_SETUP_PATH" "$ROS_FALLBACK_SETUP_PATH" "$ROS_ROOT/setup.bash"; do
  [[ -n "$setup_path" && -f "$setup_path" ]] || continue
  setup_dir="$(dirname "$setup_path")"
  if [[ "$(basename "$setup_path")" == "setup.bash" && ! -f "$setup_dir/setup.sh" ]]; then
    echo "[run_node] skip ROS setup with missing companion setup.sh: $setup_path"
    continue
  fi
  source "$setup_path"
  break
done

if [[ -f "$ROOT_DIR/scripts/setup_ros_network.sh" ]]; then
  source "$ROOT_DIR/scripts/setup_ros_network.sh"
  setup_ros_network_env
fi

PY_PATHS=(
  "$ROOT_DIR"
  "$ROOT_DIR/audio/catkin_ws/devel/lib/python3/dist-packages"
  "/ros_noetic/catkin_ws/devel/lib/python3/dist-packages"
  "/ros_noetic/catkin_ws/install/lib/python3/dist-packages"
  "/opt/ros/noetic/lib/python3/dist-packages"
)
for py_path in "${PY_PATHS[@]}"; do
  [[ -d "$py_path" ]] || continue
  case ":${PYTHONPATH:-}:" in
    *":$py_path:"*) ;;
    *) export PYTHONPATH="$py_path:${PYTHONPATH:-}" ;;
  esac
done
export DISPLAY="${DISPLAY:-:0}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"

EXTRA_ARGS=()
if [[ "${ASD_SHOW_WINDOW:-1}" == "1" ]]; then
  EXTRA_ARGS+=("_debug_show_window:=true")
else
  EXTRA_ARGS+=("_debug_show_window:=false")
fi
if [[ -n "${ASD_AUDIO_TOPIC:-}" ]]; then
  EXTRA_ARGS+=("_audio_topic:=${ASD_AUDIO_TOPIC}")
fi
if [[ -n "${ASD_AUDIO_MSG_TYPE:-}" ]]; then
  EXTRA_ARGS+=("_audio_msg_type:=${ASD_AUDIO_MSG_TYPE}")
fi
if [[ -n "${ASD_AUDIO_DEVICE_NAME:-}" ]]; then
  EXTRA_ARGS+=("_audio_device_name:=${ASD_AUDIO_DEVICE_NAME}")
  EXTRA_ARGS+=("_audio_auto_select_device:=true")
fi
if [[ -n "${ASD_AUDIO_SELECT_DEVICE_SERVICE:-}" ]]; then
  EXTRA_ARGS+=("_audio_select_device_service:=${ASD_AUDIO_SELECT_DEVICE_SERVICE}")
fi

asd_bool_arg() {
  local value="${1:-}"
  case "${value,,}" in
    1|true|yes|on) printf 'true' ;;
    0|false|no|off) printf 'false' ;;
    *) printf '%s' "$value" ;;
  esac
}

# Always pass lock params so stale ROS private params from older launches cannot
# silently override config/asd_config.py defaults.
EXTRA_ARGS+=("_active_speaker_lock_enabled:=$(asd_bool_arg "${ASD_ACTIVE_SPEAKER_LOCK_ENABLED:-true}")")
if [[ -n "${ASD_ACTIVE_SPEAKER_LOCK_SECONDS:-}" ]]; then
  EXTRA_ARGS+=("_active_speaker_lock_seconds:=${ASD_ACTIVE_SPEAKER_LOCK_SECONDS}")
fi
if [[ -n "${ASD_ACTIVE_SPEAKER_LOCK_REFRESH_SCORE:-}" ]]; then
  EXTRA_ARGS+=("_active_speaker_lock_refresh_score:=${ASD_ACTIVE_SPEAKER_LOCK_REFRESH_SCORE}")
fi

python3 "$ROOT_DIR/nodes/realtime_asd_node_parallel.py" "${EXTRA_ARGS[@]}" "$@"
