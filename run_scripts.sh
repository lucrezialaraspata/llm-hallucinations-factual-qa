#!/bin/bash

# Time interval in seconds (35 minutes)
INTERVAL=2100
CHECK_INTERVAL=60  # how often to poll squeue

while true; do
    echo "[$(date)] Submitting job..."
    JOB_SUBMIT_OUTPUT=$(sbatch prepare_data_kc.sh)
    
    # Extract Job ID (assumes standard sbatch output)
    JOB_ID=$(echo "$JOB_SUBMIT_OUTPUT" | awk '{print $4}')
    
    if [[ -z "$JOB_ID" ]]; then
        echo "[$(date)] Failed to submit job. Retrying in 60 seconds..."
        sleep 60
        continue
    fi

    echo "[$(date)] Job $JOB_ID submitted."

    # Track job start time
    START_TIME=$(date +%s)

    # Wait for job to finish
    while squeue --me | grep -q "$JOB_ID"; do
        sleep $CHECK_INTERVAL
    done

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo "[$(date)] Job $JOB_ID completed after $ELAPSED seconds."

    # If job ran less than INTERVAL, wait the remaining time
    if (( ELAPSED < INTERVAL )); then
        echo "[$(date)] Sleeping for 60 seconds before next job..."
        sleep 60
    fi
done
