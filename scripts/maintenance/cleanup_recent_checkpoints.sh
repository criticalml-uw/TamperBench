#!/bin/bash

ROOT_DIRS=(
    "$HOME/SafeTuneBed/results"
    "/data/far_ai_group/saad_ws/results"
)

while true; do
    echo "started clean up"
    du -h --max-depth=1 $HOME | sort -h
    find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} +
    for dir in "${ROOT_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            find "$dir" -type d -name "tamperbench_model_checkpoint" -mmin +50 -exec rm -rf {} +
        fi
    done
    echo "cleaned"
    du -h --max-depth=1 $HOME | sort -h
    echo "now waiting"
    sleep 3600
done
