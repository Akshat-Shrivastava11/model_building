#!/bin/bash

# 1. Initialize environment
source init.sh

# 2. Training Parameters
MIN_MASS=10.0
MAX_MASS=100.0
NUM_POINTS=1500    # Number of unique (mrho, mpi) pairs
EVENTS=300      # Events per point

# 3. Output Directory
BATCH_DIR="SVJ_Training_2D_$(date +%Y%m%d_%H%M)"
mkdir -p $BATCH_DIR

echo "----------------------------------------------------------"
echo "Generating 2D Flat Spectrum: mrho and mpi"
echo "Output Directory: $BATCH_DIR"
echo "----------------------------------------------------------"

for (( i=1; i<=NUM_POINTS; i++ )); do
    # Sample mrho and mpi independently from the 10-100 GeV range
    MRHO=$(python3 -c "import random; print(round(random.uniform($MIN_MASS, $MAX_MASS), 2))")
    MPI=$(python3 -c "import random; print(round(random.uniform($MIN_MASS, $MAX_MASS), 2))")
    
    echo "Processing Point $i/$NUM_POINTS: mrho=$MRHO, mpi=$MPI"

    # The exact command structure you provided
    ./run_model helper -C configs/model_cms.py \
        --mmed 2000 \
        --rinv 0.3 \
        --mrho $MRHO \
        --mpi $MPI \
        --steps all \
        --events $EVENTS \
        --dir $BATCH_DIR \
        --quiet

    if [ $? -eq 0 ]; then
        echo "Successfully finished Point $i"
    else
        echo "ERROR: Generation failed for mrho=$MRHO, mpi=$MPI"
    fi
done

echo "----------------------------------------------------------"
echo "Batch Production Complete!"