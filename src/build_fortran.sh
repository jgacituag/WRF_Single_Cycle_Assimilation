#!/usr/bin/env bash
# src/build_fortran.sh
# ====================
# Build the cletkf_wloc Fortran/f2py extension.
#
# Dependencies (all inside the active conda environment):
#   gfortran, f2py (numpy), LAPACK, OpenBLAS/BLAS
#
# Usage:
#   conda activate <your_env>
#   bash src/build_fortran.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE/fortran"

# ---- locate conda LAPACK/BLAS libraries --------------------------------
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[ERROR] CONDA_PREFIX not set — activate your conda environment first."
  exit 1
fi

CONDA_LIB="$CONDA_PREFIX/lib"

if [[ ! -f "$CONDA_LIB/liblapack.so" && ! -f "$CONDA_LIB/liblapack.dylib" ]]; then
  echo "[WARN] liblapack not found in $CONDA_LIB"
  echo "       Try: conda install -c conda-forge lapack"
fi

# ---- sources -----------------------------------------------------------
SOURCES="common_tools.f90 common_mtx.f90 common_letkf.f90 common_da_wloc.f90"
MODNAME="cletkf_wloc"

cd "$SRC_DIR"
rm -f *.mod *.so "${MODNAME}"*.so

echo "[build] Checking sources..."
for f in $SOURCES; do
  if [[ ! -f $f ]]; then
    echo "[ERROR] Missing $f in $SRC_DIR"
    exit 1
  fi
  echo "        found $f"
done

# ---- set library path so linker and runtime can find liblapack ---------
export LDFLAGS="-L${CONDA_LIB} -Wl,-rpath,${CONDA_LIB}"

echo "[build] Building $MODNAME ..."
echo "[build] CONDA_LIB=$CONDA_LIB"

f2py -c \
  --opt="-O3 -fopenmp" \
  --f90flags="-O3 -fopenmp" \
  -lgomp \
  -llapack \
  -L"$CONDA_LIB" \
  $SOURCES \
  -m "$MODNAME" \
  2>&1 | tee compile_cletkf_wloc.out

SOFILE=$(ls ${MODNAME}*.so 2>/dev/null | head -n1 || true)
if [[ -n "${SOFILE}" ]]; then
  echo "[build] Success: $SRC_DIR/${SOFILE}"
else
  echo "[ERROR] Build failed — see $SRC_DIR/compile_cletkf_wloc.out"
  exit 1
fi