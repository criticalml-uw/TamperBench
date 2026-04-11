# Miscellaneous Tips

This guide covers maintenance tasks and troubleshooting.

## Checkpoint Management

Model checkpoints can consume significant disk space during sweeps. Each trial saves a full model checkpoint.

### Cleanup Script

Use `scripts/maintenance/cleanup_recent_checkpoints.sh` to automatically clean up:

```bash
# Default: cleans ~/TamperBench/results
bash scripts/maintenance/cleanup_recent_checkpoints.sh

# Custom results directory
bash scripts/maintenance/cleanup_recent_checkpoints.sh /path/to/results
```

The script runs continuously and:

- Removes `tamperbench_model_checkpoint` directories older than 50 minutes
- Cleans vllm torch compile cache older than 60 minutes
- Prints disk usage before/after cleanup

### Disk Usage Tips

1. **Run cleanup during sweeps** - Start the cleanup script in a separate terminal
2. **Use small models for testing** - `Qwen/Qwen2.5-0.5B` is fast and small
3. **Delete failed trials** - Incomplete trials still save partial checkpoints
4. **Archive results** - Compress old results directories

## Common Issues

### Out of Disk Space

**Symptoms:**

- Sweeps fail mid-trial
- "No space left on device" errors

**Solutions:**

1. Run cleanup script: `bash scripts/maintenance/cleanup_recent_checkpoints.sh`
2. Delete old results: `rm -rf results/old_experiment/`
3. Use smaller models for testing
4. Check vllm cache: `du -sh ~/.cache/vllm/`

### Sweep Interrupted

**Symptoms:**

- Optuna sweep stopped unexpectedly
- Want to continue from where it left off

**Solution:**

Resume with the same `--model-alias`:

```bash
# Same alias continues from last completed trial
python scripts/whitebox/optuna_single.py model \
    --attacks lora_finetune \
    --n-trials 100 \
    --model-alias my_existing_sweep
```

### Missing Dependencies

**Symptoms:**

- `ModuleNotFoundError` for optional packages
- Platform-specific import errors

**Solutions:**

1. Check `pyproject.toml` for platform markers
2. Verify your virtual environment is activated
3. Reinstall with: `uv sync --all-groups`

### Type Checking Errors

**Symptoms:**

- basedpyright reports errors
- Pre-commit fails on type checking

**Solutions:**

1. Add type hints to new code
2. Use `# pyright: ignore[errorCode]` sparingly
3. Check excluded directories in `pyproject.toml` (under `[tool.basedpyright]`)

### CUDA Out of Memory

**Symptoms:**

- `CUDA out of memory` errors
- Training crashes mid-batch

**Solutions:**

1. Reduce batch size in config
2. Use gradient accumulation
3. Enable gradient checkpointing
4. Try a smaller model

```yaml
# In grid.yaml
per_device_train_batch_size: 4  # Reduce from 8
gradient_accumulation_steps: 2  # Accumulate to effective 8
```

## Environment Variables

| Variable                 | Description                          |
| ------------------------ | ------------------------------------ |
| `HF_TOKEN`               | HuggingFace authentication token     |
| `CUDA_VISIBLE_DEVICES`   | GPU selection (e.g., `0,1`)          |
| `TOKENIZERS_PARALLELISM` | Set to `false` to avoid warnings     |

```bash
# In .env file
HF_TOKEN=hf_xxx
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
```

## Useful Commands

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk usage by directory
du -h --max-depth=2 results/ | sort -h
# Alternative: use ncdu for interactive disk usage exploration
# ncdu results/

# Find largest files
find results/ -type f -size +1G

# Count trials per attack
find results/ -name "study.db" -exec dirname {} \; | sort | uniq -c

# List all checkpoints
find results/ -name "checkpoint" -type d
```

## Next Steps

- [USAGE.md](USAGE.md) - Running benchmarks
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guide
