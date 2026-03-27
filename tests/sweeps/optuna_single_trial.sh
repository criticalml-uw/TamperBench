MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
ALIAS="llama3_8b_instruct_baseline"
RESULTS_DIR="results/quick_trial/seed_42/"
CONFIGS_DIR="configs/whitebox/attacks_llama"
RANDOM_SEED=42

cd ~/SafeTuneBed/
export HF_HOME="/home/dev/.cache/huggingface"
export WANDB_MODE=offline

uv run scripts/whitebox/optuna_single.py \
    "$MODEL" \
    --attacks \
        no_weight_modification \
        full_parameter_finetune \
        lora_finetune \
        benign_full_parameter_finetune \
        benign_lora_finetune \
        multilingual_finetune \
        competing_objectives_finetune \
        style_modulation_finetune \
        backdoor_finetune \
    --n-trials 1 \
    --results-dir "$RESULTS_DIR" \
    --configs-dir "$CONFIGS_DIR" \
    --model-alias "$ALIAS" \
    --random-seed "$RANDOM_SEED"
