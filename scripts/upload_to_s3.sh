#!/bin/bash
# Upload SafeTuneBed results to S3 in priority order.
# Each `aws s3 sync` is idempotent — safe to re-run if interrupted.

set -e

# Bump concurrency for max speed
aws configure set default.s3.max_concurrent_requests 50
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB

LOCAL_BASE="/data/saad_hossain/SafeTuneBed/results"
S3_BASE="s3://default-bucket-saad/tamperbench_results"

# Priority order (top = first)
DIRS=(
    sweeps
    sweep_2026
    far_ai_group
    targeted_sweeps
    targeted_sweeps_v0
    sweep_trials
    rebuttal_specific
    qwen3_32_from_8
    oct29_trial
    oct6_trial
    nov7_trial
    iclr_trial
    defense_sweeps
)

for d in "${DIRS[@]}"; do
    src="$LOCAL_BASE/$d"
    dst="$S3_BASE/$d/"
    if [ ! -d "$src" ]; then
        echo "[SKIP] $d (not found)"
        continue
    fi
    echo "==================================================================="
    echo "[$(date +%H:%M:%S)] Syncing $d ..."
    echo "  src: $src"
    echo "  dst: $dst"
    echo "==================================================================="
    aws s3 sync "$src/" "$dst"
    echo "[$(date +%H:%M:%S)] Done with $d"
    echo ""
done

echo "ALL UPLOADS COMPLETE"
