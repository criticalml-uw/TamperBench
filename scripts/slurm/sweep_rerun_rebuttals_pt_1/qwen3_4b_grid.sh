cd ~/SafeTuneBed/
export HF_HOME="/data/far_ai_group/cache/huggingface"

uv run scripts/whitebox/benchmark_grid.py Qwen/Qwen3-4B-Base \
    --attacks competing_objectives_finetune backdoor_finetune style_modulation_finetune multilingual_finetune \
    --results_dir /data/far_ai_group/results/rerun_grids/qwen3_4b_base \
    --configs-dir /data/saad_hossain/SafeTuneBed/results/nov7_trial/aggregated_eps200/qwen3_4b_base/

# uv run scripts/whitebox/benchmark_grid.py Qwen/Qwen3-4B \
#     --attacks no_weight_modification benign_lora_finetune benign_full_parameter_finetune lora_finetune full_parameter_finetune competing_objectives_finetune backdoor_finetune style_modulation_finetune multilingual_finetune \
#     --results_dir /data/far_ai_group/results/rerun_grids/qwen3_4b \
#     --configs-dir /data/saad_hossain/SafeTuneBed/results/nov7_trial/aggregated_eps200/qwen3_4b/

# uv run scripts/whitebox/benchmark_grid.py Qwen/Qwen3-4B \
#     --attacks no_weight_modification benign_lora_finetune benign_full_parameter_finetune lora_finetune full_parameter_finetune competing_objectives_finetune backdoor_finetune style_modulation_finetune multilingual_finetune \
#     --results_dir /data/far_ai_group/results/rerun_grids/qwen3_4b \
#     --configs-dir /data/saad_hossain/SafeTuneBed/results/nov7_trial/aggregated_eps200/qwen3_4b/
