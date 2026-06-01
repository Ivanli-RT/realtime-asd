#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$ROOT_DIR/packages}"
PACKAGE_IMAGE_DIR="${PACKAGE_IMAGE_DIR:-$PACKAGE_ROOT/images}"
BASE_IMAGE="${ASD_BASE_IMAGE:-10.51.33.201:30002/navi_project/ros:noetic-l4t-r36.3.0}"

safe_name() {
  printf '%s' "$1" | tr '/:@' '___'
}

mkdir -p "$PACKAGE_IMAGE_DIR"

IMAGE_ARCHIVE="$PACKAGE_IMAGE_DIR/$(safe_name "$BASE_IMAGE").tar.zst"
IMAGE_SHA256="$IMAGE_ARCHIVE.sha256"

echo "[image] pulling $BASE_IMAGE"
docker pull "$BASE_IMAGE"

echo "[image] saving $BASE_IMAGE -> $IMAGE_ARCHIVE"
docker save "$BASE_IMAGE" | zstd -T0 -19 -o "$IMAGE_ARCHIVE"
sha256sum "$IMAGE_ARCHIVE" > "$IMAGE_SHA256"

echo "[image] saved archive: $IMAGE_ARCHIVE"
echo "[image] checksum: $IMAGE_SHA256"
echo "[image] restore with:"
echo "  zstd -dc '$IMAGE_ARCHIVE' | docker load"
