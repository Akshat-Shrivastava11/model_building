#!/bin/bash

# 1. Initialize environment
source init.sh

# 2. Training Parameters
MIN_MASS=10.0
MAX_MASS=100.0
NUM_POINTS=1500   # Increased for a smooth continuous spectrum
EVENTS=200       # 1500 * 200 = 300,000 events
BATCH_DIR="SVJ_Smooth_Dracon_Mpi_only$(date +%Y%m%d_%H%M)"
mkdir -p $BATCH_DIR

echo "Generating 300k events across 1000 points..."

for (( i=1; i<=NUM_POINTS; i++ )); do
    # Use the loop index as a seed for guaranteed uniqueness
    MPI=$(python3 -c "import random; random.seed($i); print(round(random.uniform($MIN_MASS, $MAX_MASS), 2))")
    #MRHO=$(python3 -c "print(round($MPI * 2.0, 2))") 
    
    # Create a unique sub-folder for this specific mass point
    POINT_DIR="$BATCH_DIR/point_mPI_${MPI}"
    mkdir -p $POINT_DIR

    echo "[$i/1500] Generating mpi=$MPI, mrho=$MRHO"

    ./run_model helper -C configs/model_cms.py \
        --mmed 2000 \
        --rinv 0.3 \
        --mpi $MPI \
        --steps all \
        --events $EVENTS \
        --dir $POINT_DIR \
        --quiet

    # Optional: Keep track of the parameters in a simple CSV for easy loading later
    echo "$MPI,$MRHO,$EVENTS,$POINT_DIR" >> "$BATCH_DIR/summary.csv"
done

echo "Done! Ready for Draconet training."