#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE="${BUILD_TYPE:-Release}"
GENERATOR="${GENERATOR:-Ninja}"

MAP_VIEWER_DIR="."
GLCONVERTER_DIR="glConverter"

MAP_BUILD_DIR="build-map_viewer"
GL_BUILD_DIR="build-gl_converter"

# Optional: clean builds
if [[ "${1:-}" == "--clean" ]]; then
  rm -rf "$MAP_BUILD_DIR" "$GL_BUILD_DIR"
fi

cmake_configure_and_build() {
  local src_dir="$1"
  local build_dir="$2"

  if [[ ! -f "$src_dir/CMakeLists.txt" ]]; then
    echo "ERROR: '$src_dir/CMakeLists.txt' not found."
    echo "Hint: set MAP_VIEWER_DIR / GLCONVERTER_DIR to the correct paths."
    exit 1
  fi

  echo "==> Configure: $src_dir -> $build_dir ($BUILD_TYPE, $GENERATOR)"
  cmake -S "$src_dir" -B "$build_dir" -G "$GENERATOR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

  echo "==> Build: $build_dir"
  cmake --build "$build_dir" --config "$BUILD_TYPE"
}

cmake_configure_and_build "$MAP_VIEWER_DIR" "$MAP_BUILD_DIR"
cmake_configure_and_build "$GLCONVERTER_DIR" "$GL_BUILD_DIR"

echo "==> Done."
