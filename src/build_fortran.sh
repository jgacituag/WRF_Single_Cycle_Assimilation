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

# Every queue job runs this script at its start, many jobs at once, all against this one
# shared directory. It used to begin with `rm -f *.so` and rebuild in place, so a job
# importing the module while another was rebuilding found it missing -- two of the 144
# QC0t10 jobs died of ModuleNotFoundError that way. So: no rebuild while the module is
# newer than its sources and this script, and when one is needed it happens in a private
# directory and lands with a single rename. An importer sees the old module or the new
# one, never neither. FORCE=1 rebuilds anyway.
#
# `python` must be the interpreter the jobs run, i.e. the conda env must be active.
SOFILE="${MODNAME}$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
if [[ "${FORCE:-0}" != "1" && -f "$SOFILE" ]]; then
  stale=0
  for f in $SOURCES "$HERE/build_fortran.sh"; do
    if [[ "$f" -nt "$SOFILE" ]]; then stale=1; fi
  done
  # A module built on another machine can be newer than its sources and still not load
  # here -- seldon and hydra see the same datosmunin copy -- so it must also import.
  if [[ $stale -eq 0 ]] && python -c "import sys; sys.path.insert(0, '.'); import $MODNAME" 2>/dev/null; then
    echo "[build] $SOFILE is newer than its sources and imports; not rebuilding (FORCE=1 to force)"
    exit 0
  fi
fi

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

# Inside $SRC_DIR, so the final mv is a rename on the same filesystem.
BUILD=$(mktemp -d "$SRC_DIR/.build.XXXXXX")
trap 'rm -rf "$BUILD"' EXIT
cp $SOURCES "$BUILD"/
cd "$BUILD"

# The f2py of the python that will import the module, not whichever f2py is first on
# PATH: on the login node that one is Intel's python 3.7, and its module is unimportable.
if ! python -m numpy.f2py -c \
  --opt="-O3 -fopenmp" \
  --f90flags="-O3 -fopenmp" \
  -lgomp \
  -llapack \
  -L"$CONDA_LIB" \
  $SOURCES \
  -m "$MODNAME" \
  2>&1 | tee compile_cletkf_wloc.out; then
  cp compile_cletkf_wloc.out "$SRC_DIR"/
  echo "[ERROR] Build failed — see $SRC_DIR/compile_cletkf_wloc.out"
  exit 1
fi
cp compile_cletkf_wloc.out "$SRC_DIR"/

BUILT=$(ls ${MODNAME}*.so 2>/dev/null | head -n1 || true)
if [[ "${BUILT}" != "${SOFILE}" ]]; then
  echo "[ERROR] f2py produced '${BUILT}', but this python imports ${SOFILE}" \
       "— see $SRC_DIR/compile_cletkf_wloc.out"
  exit 1
fi
mv -f "$BUILT" "$SRC_DIR/$SOFILE"
echo "[build] Success: $SRC_DIR/${SOFILE}"