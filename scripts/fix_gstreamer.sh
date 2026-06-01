#!/usr/bin/env bash
set -euo pipefail

TEGRA_GST="/usr/lib/aarch64-linux-gnu/tegra/libnvgstreamer-1.0.so"
SYS_GST_REAL="/usr/lib/aarch64-linux-gnu/libgstreamer-1.0.so.0.1603.0"
SYS_GST_LINK="/usr/lib/aarch64-linux-gnu/libgstreamer-1.0.so"

if [[ -f "$TEGRA_GST" ]]; then
  size=$(stat -c %s "$TEGRA_GST" || echo 0)
  if [[ "$size" -eq 0 ]]; then
    if [[ -f "$SYS_GST_REAL" ]]; then
      ln -sf "$SYS_GST_REAL" "$TEGRA_GST"
      echo "[fix_gstreamer] patched empty tegra gstreamer -> $SYS_GST_REAL"
    elif [[ -f "$SYS_GST_LINK" ]]; then
      ln -sf "$SYS_GST_LINK" "$TEGRA_GST"
      echo "[fix_gstreamer] patched empty tegra gstreamer -> $SYS_GST_LINK"
    else
      echo "[fix_gstreamer] no fallback gstreamer found, skip"
    fi
  fi
fi
