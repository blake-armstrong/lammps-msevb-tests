#!/usr/bin/env bash
set -euo pipefail

LAMMPS_FORK="${LAMMPS_FORK:-$HOME/git/lammps-fork}"
BUILD_DIR="${BUILD_DIR:-$LAMMPS_FORK/build-msevb}"
JOBS="${JOBS:-8}"

cmake -S "$LAMMPS_FORK/cmake" -B "$BUILD_DIR" \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=on \
  -D BUILD_OMP=off \
  -D DOWNLOAD_POTENTIALS=off \
  -D PKG_MSEVB=on \
  -D PKG_KSPACE=on \
  -D PKG_MOLECULE=on \
  -D PKG_EXTRA-MOLECULE=on \
  -D PKG_EXTRA-PAIR=on \
  -D PKG_MANYBODY=on \
  -D PKG_EXTRA-FIX=on

cmake --build "$BUILD_DIR" -j "$JOBS"

echo "LMP=$BUILD_DIR/lmp"
