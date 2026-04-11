#!/bin/bash

# Optional: override ROOT_DIR by passing a path as the first argument.
ROOT_DIR="${1:-$HOME/TamperBench/results}"

if [ ! -d "$ROOT_DIR" ]; then
    echo "Directory not found: $ROOT_DIR" >&2
    exit 1
fi

while true; do
    echo "started clean up"
    du -h --max-depth=1 $HOME | sort -h
    find ~/.cache/vllm/torch_compile_cache/* -maxdepth 0 -type d -mmin +60 -exec rm -rf {} +
    find "$ROOT_DIR" -type d -name "tamperbench_model_checkpoint" -mmin +50 -exec rm -rf {} +
    echo "cleaned"
    du -h --max-depth=1 $HOME | sort -h
    echo "now waiting"
    sleep 3600
done
