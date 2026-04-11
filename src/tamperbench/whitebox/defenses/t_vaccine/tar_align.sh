export HF_HOME="/data/far_ai_group/cache/huggingface"

lamb=${1:-5}
alpha=${2:-0.1}
bad_sample_num=${3:-2000}
epoch=20
model_path=${4:-/models/meta-llama/Llama-2-7b-hf}
alignment_dataset_path="/home/dev/persistent-storage/TamperBench/src/tamperbench/whitebox/defenses/t_vaccine/beavertails_with_refusals_train.json" # alignment dataset
beaverTails_dataset_path="PKU-Alignment/BeaverTails" # fine-tuning dataset
path_after_slash=$(basename "$model_path")
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
SRC_DIR="$REPO_ROOT/src"

if [ -n "$PYTHONPATH" ]; then
  export PYTHONPATH="$SRC_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$SRC_DIR"
fi

# cd "$REPO_ROOT" || exit 1

uv run "$SRC_DIR/tamperbench/whitebox/defenses/t_vaccine/tar.py" \
	--model_name_or_path ${model_path}  \
	--data_path PKU-Alignment/BeaverTails_safe_alignment \
	--bf16 True \
	--output_dir ckpt/${path_after_slash}_tar_${epoch} \
	--num_train_epochs ${epoch} \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  1e-3  \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer tar \
	--system_evaluate True \
	--alignment_dataset_path ${alignment_dataset_path} \
	--beaverTails_dataset_path ${beaverTails_dataset_path} \
	--alpha ${alpha} \
	--lamb ${lamb} \
	--bad_sample_num $bad_sample_num \
	--max_length 200 \
