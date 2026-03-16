#!/bin/bash
# queue_ws.sh
# ===========
# PBS queue script for all WS experiments.
#
# Submit with:
#   qsub queue_ws.sh                              # defaults below
#   qsub -v CFG=configs/ws1.yaml,WORKERS=30 queue_ws.sh
#   qsub -v CFG=configs/ws2.yaml,WORKERS=30,VERBOSE=1 queue_ws.sh
#
# All experiment logic is in the YAML config.
# This script only handles environment setup and parallelism.

# ── PBS directives ─────────────────────────────────────────────────────────
#PBS -N WS_Experiment
#PBS -l nodes=1:ppn=48
#PBS -j oe
#PBS -o /home/jorge.gacitua/salidas/WS_Experiments/logs/ws_run.log
#PBS -V

# ── configuration ──────────────────────────────────────────────────────────
REPO=/home/jorge.gacitua/WRF_Single_Cycle_Assimilation
CONDA_SH=/home/jorge.gacitua/salidas/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=wrf_python_assimilation

# Config file to run (override with -v CFG=...)
CFG=${CFG:-configs/ws2.yaml}

# Workers = simultaneous truth-member processes
# Rule: set to min(n_truth_members, available_cores)
#   48-core node  -> WORKERS=30 (all members at once, 18 cores spare)
#   120-core node -> WORKERS=30 (run two experiments simultaneously)
WORKERS=${WORKERS:-30}

# Verbosity (0=silent 1=per-member 2=per-method 3=debug)
VERBOSE=${VERBOSE:-1}

# ── disable internal threading ─────────────────────────────────────────────
# We parallelize over truth members (one process each).
# Each process is single-threaded; the OS schedules them across cores.
# Disabling numpy/MKL threading prevents each process from spawning its
# own thread pool, which would cause oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── environment ────────────────────────────────────────────────────────────
source /opt/load-libs.sh 3
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"

mkdir -p /home/jorge.gacitua/salidas/WS_Experiments/logs

echo "============================================================"
echo "  Config   : $CFG"
echo "  Workers  : $WORKERS"
echo "  Verbose  : $VERBOSE"
echo "  Node CPUs: $(nproc)"
echo "  Started  : $(date)"
echo "============================================================"

python -u src/runners/run_experiment.py \
    --config  "$CFG" \
    --workers "$WORKERS" \
    --verbose "$VERBOSE"

echo "============================================================"
echo "  Finished : $(date)"
echo "============================================================"
