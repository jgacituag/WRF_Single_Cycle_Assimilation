#!/usr/bin/env bash
# scripts/build_fortran.sh
# Build the cletkf_wloc Fortran module with f2py + OpenMP.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC_DIR="$ROOT/third_party/fortran"
OUT_DIR="$ROOT/third_party"

cd "$SRC_DIR"

# Ensure sources exist
for f in common_da_wloc.f90 common_letkf.f90 common_mtx.f90 common_tools.f90; do
  if [[ ! -f $f ]]; then
    echo "[ERROR] Missing $f in $SRC_DIR"
    exit 1
  fi
done

# Compiler & flags
FC=${FC:-gfortran}
OMP_FLAG=${OMP_FLAG:-"-fopenmp"}
PYTHON=${PYTHON:-python}
F2PY="$PYTHON -m numpy.f2py"
MODNAME="cletkf_wloc"

echo "[build] Sources: common_da_wloc.f90 common_letkf.f90 common_mtx.f90 common_tools.f90"

$F2PY -c -m $MODNAME \
  common_da_wloc.f90 common_letkf.f90 common_mtx.f90 common_tools.f90 \
  --fcompiler=gnu95 \
  --opt='-O3' \
  --f90flags="$OMP_FLAG -O3" \
  -lgomp

SOFILE=$(ls ${MODNAME}*.so | head -n1 || true)
if [[ -n "${SOFILE}" ]]; then
  mv -v "${SOFILE}" "$OUT_DIR/"
  echo "[build] Installed: $OUT_DIR/${SOFILE}"
else
  echo "[warn] Build finished but .so not found."
fi
