#!/usr/bin/env bash

# ROS1 publishers connect back to each subscriber URI. Inside Docker, the
# default hostname can be unreachable or stale, so prefer an explicit host IP.
setup_ros_network_env() {
  if [[ -n "${ROS_IP:-}" || -n "${ROS_HOSTNAME:-}" ]]; then
    return 0
  fi

  local candidate=""
  if command -v hostname >/dev/null 2>&1; then
    candidate="$(
      hostname -I 2>/dev/null \
        | tr ' ' '\n' \
        | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' \
        | grep -Ev '^(127\.|169\.254\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.|10\.42\.|10\.250\.)' \
        | head -n 1 || true
    )"
  fi

  if [[ -z "$candidate" ]] && command -v ip >/dev/null 2>&1; then
    candidate="$(
      ip -o -4 addr show scope global 2>/dev/null \
        | awk '{print $4}' \
        | cut -d/ -f1 \
        | grep -Ev '^(127\.|169\.254\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.|10\.42\.|10\.250\.)' \
        | head -n 1 || true
    )"
  fi

  if [[ -n "$candidate" ]]; then
    export ROS_IP="$candidate"
    unset ROS_HOSTNAME
    echo "[ros-network] ROS_IP auto-set to $ROS_IP"
  else
    echo "[ros-network] ROS_IP not set; set ROS_IP manually if ROS callbacks do not arrive"
  fi
}
