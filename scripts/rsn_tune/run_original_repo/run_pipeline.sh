#!/bin/bash
# Run the full SN-Tune + RSN-Tune pipeline for a given model.
#
# Usage:
#   ./scripts/run_pipeline.sh <model_name> <output_dir>
#
# Examples:
#   ./scripts/run_pipeline.sh meta-llama/Llama-2-7b-chat-hf /output/llama2
#   ./scripts/run_pipeline.sh mistralai/Mistral-7B-Instruct-v0.2 /output/mistral

set -euo pipefail

MODEL="${1:?Usage: $0 <model_name> <output_dir>}"
OUTPUT_DIR="${2:?Usage: $0 <model_name> <output_dir>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Step 0: Prepare data if not already done
# ---------------------------------------------------------------------------
if [ ! -f "$DATA_DIR/harmful_behaviors.txt" ] || \
   [ ! -f "$DATA_DIR/circuit_breakers_train.json" ] || \
   [ ! -f "$DATA_DIR/wikipedia_en.txt" ]; then
    echo "=== Preparing datasets ==="
    python "$SCRIPT_DIR/prepare_data.py" --data-dir "$DATA_DIR" || true
    for f in "$DATA_DIR/harmful_behaviors.txt" \
             "$DATA_DIR/circuit_breakers_train.json" \
             "$DATA_DIR/wikipedia_en.txt"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: prepare_data.py failed — $f not found"
            exit 1
        fi
    done
fi

# ---------------------------------------------------------------------------
# Step 1: Install deps + patch transformers for neuron DETECTION
#
# Detection needs transformers>=4.43 (MistralConfig.head_dim, SlidingWindowCache).
# Training uses a custom SFTTrainer subclass (no trainer.py patching needed),
# so the same transformers version works for both steps.
# ---------------------------------------------------------------------------
echo "=== Installing dependencies ==="
uv pip install transformers==4.44.0 trl==0.8.6 peft==0.10.0 accelerate==0.33.0

TRANSFORMERS_DIR=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")

echo "=== Patching transformers for neuron detection ==="
for f in \
    "$TRANSFORMERS_DIR/models/llama/modeling_llama.py" \
    "$TRANSFORMERS_DIR/models/mistral/modeling_mistral.py"; do
    [ -f "${f}.orig" ] || cp "$f" "${f}.orig"
done

cp "$REPO_DIR/neuron_detection/transformers/models/llama/modeling_llama.py" \
   "$TRANSFORMERS_DIR/models/llama/modeling_llama.py"
cp "$REPO_DIR/neuron_detection/transformers/models/mistral/modeling_mistral.py" \
   "$TRANSFORMERS_DIR/models/mistral/modeling_mistral.py"

# NOTE: The Llama detection code has top_number_attn=2000, top_number_ffn=12000.
# For Llama-2-7B (intermediate_size=11008), top_number_ffn=12000 exceeds the FFN
# dimension, so np.argsort will return ALL FFN neurons for every prompt. The
# intersection across prompts will still filter this down, but if detection results
# look wrong, consider reducing these values. The Mistral code uses 1000/2000.
# Uncomment the lines below to override:
# sed -i 's/top_number_attn = 2000/top_number_attn = 1000/' \
#     "$TRANSFORMERS_DIR/models/llama/modeling_llama.py"
# sed -i 's/top_number_ffn = 12000/top_number_ffn = 2000/' \
#     "$TRANSFORMERS_DIR/models/llama/modeling_llama.py"

find "$TRANSFORMERS_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$TRANSFORMERS_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 2: Detect SAFETY neurons (from harmful behavior corpus)
# ---------------------------------------------------------------------------
# Paper Appendix A.2: "sampling 200 documents for detection"
echo "=== Detecting safety neurons (200 harmful prompts) ==="
python "$SCRIPT_DIR/detect_neurons.py" \
    --model "$MODEL" \
    --corpus "$DATA_DIR/harmful_behaviors.txt" \
    --num-samples 200 \
    --output "$OUTPUT_DIR/safety_neurons.txt"

# ---------------------------------------------------------------------------
# Step 3: Detect FOUNDATION neurons (from Wikipedia corpus, for RSN-Tune)
# ---------------------------------------------------------------------------
echo "=== Detecting foundation neurons (200 Wikipedia samples) ==="
python "$SCRIPT_DIR/detect_neurons.py" \
    --model "$MODEL" \
    --corpus "$DATA_DIR/wikipedia_en.txt" \
    --num-samples 200 \
    --output "$OUTPUT_DIR/foundation_neurons.txt"

# ---------------------------------------------------------------------------
# Step 4: Restore modeling files for training
# (Training uses standard transformers models, no detection patches needed.)
# ---------------------------------------------------------------------------
echo "=== Restoring transformers for training ==="
for f in \
    "$TRANSFORMERS_DIR/models/llama/modeling_llama.py" \
    "$TRANSFORMERS_DIR/models/mistral/modeling_mistral.py"; do
    [ -f "${f}.orig" ] && cp "${f}.orig" "$f"
done
find "$TRANSFORMERS_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$TRANSFORMERS_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 5: SN-Tune (safety neurons only)
# ---------------------------------------------------------------------------
# Paper uses 50 safety training documents
echo "=== Running SN-Tune (50 training samples) ==="
python "$SCRIPT_DIR/train_neurons.py" \
    --model "$MODEL" \
    --neuron-file "$OUTPUT_DIR/safety_neurons.txt" \
    --data-file "$DATA_DIR/circuit_breakers_train.json" \
    --output-dir "$OUTPUT_DIR/sn_tune" \
    --cache-dir "$OUTPUT_DIR/cache" \
    --max-samples 50

# ---------------------------------------------------------------------------
# Step 6: RSN-Tune (safety neurons minus foundation neurons)
# ---------------------------------------------------------------------------
echo "=== Running RSN-Tune (50 training samples) ==="
python "$SCRIPT_DIR/train_neurons.py" \
    --model "$MODEL" \
    --neuron-file "$OUTPUT_DIR/safety_neurons.txt" \
    --foundation-neuron-file "$OUTPUT_DIR/foundation_neurons.txt" \
    --data-file "$DATA_DIR/circuit_breakers_train.json" \
    --output-dir "$OUTPUT_DIR/rsn_tune" \
    --cache-dir "$OUTPUT_DIR/cache" \
    --max-samples 50

echo ""
echo "=== Done ==="
echo "SN-Tune model:  $OUTPUT_DIR/sn_tune/final_model"
echo "RSN-Tune model: $OUTPUT_DIR/rsn_tune/final_model"
