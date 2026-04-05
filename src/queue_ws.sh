#!/bin/bash
#PBS -N WRF_Full_Sweep
#PBS -l nodes=1:ppn=120          # Grab one full 48-core node
#PBS -j oe
#PBS -o /nfsmounts/storage/scratch/jorge.gacitua/WRF_Single_Cycle_Assimilation/logs/chunk_master.log
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
export OMP_NUM_THREADS=40        # 4 cores per python process
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# --- CONCURRENCY LIMITS ---
MAX_PAR=3                      
running=0
CHUNK_SIZE=500

# --- PRE-RUN CHECK ---
# Ask Python how many chunks there are (tail -n 1 ensures we only grab the last printed number)
echo "Asking Python for chunk count..."

# Capture ALL output from the pre-flight check
OUTPUT=$(python src/runners/run_experiment.py --config configs/ws1.yaml --chunk_size $CHUNK_SIZE --get_chunk_count)
EXIT_CODE=$?

# 1. Did Python crash?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python crashed during pre-flight check!"
    echo "Python output was: $OUTPUT"
    exit 1
fi

# 2. Extract the number
CALC_CHUNKS=$(echo "$OUTPUT" | tail -n 1)

# 3. Is it actually a number?
if ! [[ "$CALC_CHUNKS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Expected an integer for chunk count, got: $CALC_CHUNKS"
    exit 1
fi

echo "Python reports a total of $CALC_CHUNKS chunks."

# --- OPTIONAL OVERRIDES ---
# If you didn't pass START or END via qsub, they default to 1 and CALC_CHUNKS
START_CHUNK=${START:-1}
END_CHUNK=${END:-$CALC_CHUNKS}

echo "Processing Chunks $START_CHUNK to $END_CHUNK with Max Concurrency: $MAX_PAR"

# --- Launch Loop ---
for (( CHUNK_ID=$START_CHUNK; CHUNK_ID<=$END_CHUNK; CHUNK_ID++ )); do
/chunk_master.log
  LOGFILE="/nfsmounts/storage/scratch/jorge.gacitua/WRF_Single_Cycle_Assimilation/logs/chunk_${CHUNK_ID}.log"
  echo "Launching Chunk $CHUNK_ID -> $LOGFILE"

  python -u src/runners/run_experiment.py \
    --config configs/ws1.yaml \
    --chunk_id $CHUNK_ID \
    --chunk_size $CHUNK_SIZE \
    > "$LOGFILE" 2>&1 &
  sleep 2
  running=$((running+1))
  
  if [ "$running" -ge "$MAX_PAR" ]; then
    wait
    running=0
  fi

done

wait
echo "=== All chunks ($START_CHUNK to $END_CHUNK) finished ==="