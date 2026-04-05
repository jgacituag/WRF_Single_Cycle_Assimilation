#!/bin/bash
#PBS -N WRF_Full_Sweep_stridded
#PBS -l nodes=1:ppn=120          # Grab one full 48-core node
#PBS -j oe
#PBS -o /nfsmounts/storage/scratch/jorge.gacitua/WRF_Single_Cycle_Assimilation/logs/chunk_master_ws2.log
#PBS -V

cd /nfsmounts/storage/scratch/jorge.gacitua/WRF_Single_Cycle_Assimilation/

# Load environment
source /opt/load-libs.sh 3
source /nfsmounts/storage/scratch/jorge.gacitua/miniconda3/etc/profile.d/conda.sh
conda activate intermediate_exp

# Build the Fortran library directly on the compute node
#echo "Building Fortran library on compute node: $(hostname)"
#bash src/build_fortran.sh
#if [ $? -ne 0 ]; then
#    echo "ERROR: Fortran build failed on the compute node!"
#    exit 1
#fi
#echo "Fortran build successful."

# --- THREAD CONFIGURATION ---
export OMP_NUM_THREADS=120        # 4 cores per python process
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting Strided Experiment on $(hostname)"

# Run the single strided experiment (no chunking arguments needed)
python -u src/runners/run_experiment.py \
  --config configs/ws2.yaml

echo "=== Strided Experiment Finished ==="