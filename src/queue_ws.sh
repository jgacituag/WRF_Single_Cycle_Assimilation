#!/bin/bash
#PBS -N WS_DA
#PBS -l nodes=1:ppn=124
#PBS -j oe
#PBS -o /home/jorge.gacitua/datosmunin/WRF_Single_Cycle_Assimilation/logs/ws_da.log
#PBS -V

# ---------------------------------------------------------------------------
# queue_ws.sh — WRF Single-Cycle Assimilation job script
#
# Handles two modes automatically (read from config):
#
#   sweep     : Python multiprocessing, one worker per core.
#               OMP_NUM_THREADS=1 to avoid Fortran thread contention.
#
#   multi_obs : Single Python process, Fortran uses all OMP threads.
#               OMP_NUM_THREADS set to N_CORES below.
#
# Submit:
#   qsub src/queue_ws.sh -v CONFIG=configs/ws1.yaml,TM=0
#   qsub src/queue_ws.sh -v CONFIG=configs/ws1.yaml,TM=0,WORKERS=8
#
# Variables (all optional, have defaults):
#   CONFIG   path to yaml config  (default: configs/ws1.yaml)
#   TM       truth member index   (default: read from config)
#   WORKERS  parallel workers for sweep mode  (default: N_CORES)
# ---------------------------------------------------------------------------

set -euo pipefail

# --- paths ------------------------------------------------------------------
REPO=/home/jorge.gacitua/datosmunin/WRF_Single_Cycle_Assimilation
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

cd "$REPO"

# --- environment ------------------------------------------------------------
source /opt/load-libs.sh 3
source /home/jorge.gacitua/miniconda3/etc/profile.d/conda.sh
conda activate intermediate_exp

#echo "Building Fortran library on compute node: $(hostname)"
bash src/build_fortran.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Fortran build failed on the compute node!"
    exit 1
fi
echo "Fortran build successful."
# --- parameters -------------------------------------------------------------
CONFIG=${CONFIG:-configs/ws1.yaml}
N_CORES=124

# Read mode from config to decide threading strategy
MODE=$(python3 -c "
import yaml
with open('$CONFIG') as f: cfg = yaml.safe_load(f)
obs = cfg['sweep']['obs_points']
print(obs if isinstance(obs, str) else obs.get('mode','sweep'))
")

echo "[queue] config=$CONFIG  mode=$MODE  node=$(hostname)  cores=$N_CORES"

# --- threading --------------------------------------------------------------
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [ "$MODE" = "multi_obs" ]; then
    # Single Python process — Fortran uses all cores via OpenMP
    export OMP_NUM_THREADS=$N_CORES
    WORKERS=1
    echo "[queue] multi_obs: OMP_NUM_THREADS=$OMP_NUM_THREADS"
else
    # Python workers — each Fortran call is single-threaded
    export OMP_NUM_THREADS=1
    WORKERS=${WORKERS:-$N_CORES}
    echo "[queue] sweep: workers=$WORKERS  OMP_NUM_THREADS=1"
fi

# --- build TM argument if provided ------------------------------------------
TM_ARG=""
if [ -n "${TM:-}" ]; then
    TM_ARG="--tm $TM"
    echo "[queue] truth member: $TM"
fi

# --- run --------------------------------------------------------------------
echo "[queue] starting at $(date)"
t_start=$SECONDS

python -u src/runners/run_experiment.py \
    --config "$CONFIG" \
    --workers "$WORKERS" \
    --verbose 1 \
    $TM_ARG

t_elapsed=$(( SECONDS - t_start ))
echo "[queue] finished at $(date)  elapsed=${t_elapsed}s"