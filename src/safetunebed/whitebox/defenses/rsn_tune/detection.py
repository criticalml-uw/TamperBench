"""Vectorized batched neuron importance detection.

Implements the parallel neuron detection method from:
    Zhao et al. (2025) "Understanding and Enhancing Safety Mechanisms of LLMs via
    Safety-Specific Neuron" (ICLR 2025)

This module detects "safety neurons" - neurons that are consistently important for
processing harmful queries. A neuron is important if deactivating it (zeroing its
parameters) significantly changes the model output.

Mathematical Basis (Paper Appendix A.1):
    - FFN neurons (Equation 9): Uses separable norms to compute importance in parallel
      ||h_ffn[:,:,i]||_2 * ||W_down[:,i]||_2
    - V neurons (Equation 9): Applied similarly since V is outside softmax
    - Q/K neurons (Equations 12-14): Uses chunked batched softmax difference computation

Implementation Notes:
    1. **Chunked Processing**: Uses configurable chunk_size (default 256) to balance
       memory usage vs. speed during Q/K detection. Not mentioned in paper.

    2. **V Approximation**: The V projection importance computation (per paper) ignores
       attention weights, which overestimates importance but preserves relative ranking.
       This matches the paper's approach.

Core Algorithm (Paper Equations 1-3):
    1. Compute per-sample importance: ||h_without_neuron - h_with_neuron||_2
    2. Filter consistently important neurons: those with importance >= threshold on ALL samples
    3. Return set of (param_name, neuron_idx, orientation) tuples

Uses mathematical properties to avoid per-neuron loops where possible:
- FFN: ||h_ffn[:,:,i]||_2 * ||W_down[:,i]||_2 (separable norms)
- V: Similar separable norm computation
- Q/K: Chunked batched softmax computation

Differences from the original codebase (github.com/zhaoyiran924/Safety-Neuron):

    1. **Neuron selection**: This implementation uses threshold-based filtering
       (importance >= threshold on ALL samples). The original code uses top-K selection
       per sample (top 12,000 FFN, top 2,000 attention per layer) followed by set
       intersection. We follow the paper's description of consistent importance, which
       does not prescribe a fixed top-K count.

    2. **FFN importance metric**: This implementation uses the paper's Equation 9:
       ||silu(gate)*up[:,:,i]||_2 * ||W_down[:,i]||_2 (L2 product of gated activation
       and weight norms). The original code uses sum(abs(up_proj(x)), dim=1) (L1 norm
       of up_proj output only, ignoring gate_proj and down_proj weights).

    3. **V importance metric**: This implementation uses the paper's Equation 9:
       ||V[:,:,k]||_2 * ||W_o[:,k]||_2 (separable norms). The original code uses
       sum(abs(attn_output), dim=1) (L1 norm of attention output before o_proj).

    4. **Q/K importance**: This implementation operates on pre-RoPE Q/K projections
       (hidden_states @ W_q/k.T) per the paper's formulation. The original code
       operates on post-RoPE Q/K states. Both zero individual dimensions and measure
       the resulting attention difference, but the original also has a scaling
       discrepancy (subtracts unscaled per-dimension contributions from scaled logits).

    5. **gate_proj neurons**: This implementation detects and trains gate_proj neurons
       (since they share the same FFN neuron importance score per the paper). The
       original code does not apply gradient updates to gate_proj (it falls into the
       else branch that zeros all gradients).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

import datasets
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Type alias for neuron identifiers: (parameter_name, neuron_index, "row"|"column")
NeuronId = tuple[str, int, str]


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for attention mechanism architecture.

    Encapsulates model-specific parameters to reduce function signature clutter.
    """

    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    num_kv_heads: int
    d_head: int
    is_gqa: bool
    num_repeats: int  # For GQA: num_heads // num_kv_heads
    chunk_size: int = 256


def chunked_range(total_size: int, chunk_size: int):
    """Iterate over ranges in chunks, yielding (start, chunk_size) tuples."""
    for start in range(0, total_size, chunk_size):
        yield start, min(chunk_size, total_size - start)


def build_param_name_map(model: PreTrainedModel) -> dict[int, str]:
    """Build mapping from tensor id to parameter name.

    This avoids name mismatches when models are wrapped differently
    between detection and training (e.g., with device_map="auto").
    """
    return {id(param): name for name, param in model.named_parameters()}


def record_importance(
    neuron_importance: defaultdict[tuple[str, int, str], list[float]],
    param_name: str,
    importance_chunk: list[float],
    start_idx: int,
    orientation: str = "row",
) -> None:
    """Record importance scores for a chunk of neurons."""
    for i, score in enumerate(importance_chunk):
        neuron_importance[(param_name, start_idx + i, orientation)].append(score)


def create_masked_batch(
    tensor: torch.Tensor, start: int, chunk_size: int
) -> torch.Tensor:
    """Create batch of tensors with one dimension zeroed per batch element.

    Args:
        tensor: Shape [batch, seq, dim] - the tensor to mask
        start: Starting index for masking
        chunk_size: Number of masked versions to create

    Returns:
        Shape [chunk_size, batch, seq, dim] with position start+i zeroed in element i
    """
    masked = tensor.unsqueeze(0).expand(chunk_size, -1, -1, -1).clone()
    # Zero out one dimension per batch element
    indices = torch.arange(chunk_size, device=tensor.device)
    masked[indices, :, :, start + indices] = 0
    return masked


def _detect_ffn_neurons(
    model: PreTrainedModel,
    activations: dict[str, torch.Tensor],
    neuron_importance: defaultdict[tuple[str, int, str], list[float]],
    param_name_map: dict[int, str],
) -> None:
    """Detect FFN neurons using vectorized norm computation.

    Uses the property: ||h[:,:,i] ⊗ W[:,i]||_2 = ||h[:,:,i]||_2 * ||W[:,i]||_2

    All three projections (gate, up, down) share the same importance score
    because zeroing intermediate dimension k has the same effect regardless
    of which projection's weights are modified.
    """
    for name, module in model.named_modules():
        if "mlp" not in name or not hasattr(module, "down_proj"):
            continue

        up_proj_name = name.replace("mlp", "mlp.up_proj")
        gate_proj_name = name.replace("mlp", "mlp.gate_proj")

        if up_proj_name not in activations or gate_proj_name not in activations:
            continue

        gate_output = activations[gate_proj_name]
        up_output = activations[up_proj_name]
        h_ffn = torch.nn.functional.silu(gate_output) * up_output

        W_down = module.down_proj.weight
        W_gate = module.gate_proj.weight
        W_up = module.up_proj.weight

        # Vectorized: compute all norms at once
        h_norms = torch.norm(h_ffn, p=2, dim=(0, 1))
        W_col_norms = torch.norm(W_down, p=2, dim=0)
        importance_all = (h_norms * W_col_norms).tolist()

        # Record same importance for all three projections using actual param names
        # down_proj: column orientation (zeroing column k of down_proj)
        # gate/up_proj: row orientation (zeroing row k, i.e., output dim k)
        record_importance(
            neuron_importance, param_name_map[id(W_down)], importance_all, 0, "column"
        )
        record_importance(
            neuron_importance, param_name_map[id(W_gate)], importance_all, 0, "row"
        )
        record_importance(
            neuron_importance, param_name_map[id(W_up)], importance_all, 0, "row"
        )


def detect_attention_neurons(
    model: PreTrainedModel,
    activations: dict[str, torch.Tensor],
    neuron_importance: defaultdict[tuple[str, int, str], list[float]],
    chunk_size: int,
    param_name_map: dict[int, str],
) -> None:
    """Detect attention neurons using vectorized computation."""
    for name, module in model.named_modules():
        if "self_attn" not in name or not hasattr(module, "q_proj"):
            continue

        if name not in activations:
            continue

        W_q = module.q_proj.weight
        W_k = module.k_proj.weight
        W_v = module.v_proj.weight
        W_o = module.o_proj.weight

        hidden_states = activations[name]
        batch_size, seq_len, d_model = hidden_states.shape

        num_heads = (
            getattr(module, "num_heads", None)
            or getattr(module, "num_attention_heads", None)
            or model.config.num_attention_heads
        )
        num_kv_heads = (
            getattr(module, "num_key_value_heads", None)
            or getattr(module, "num_kv_heads", None)
            or getattr(model.config, "num_key_value_heads", num_heads)
        )
        d_head = (
            getattr(module, "head_dim", None)
            or getattr(model.config, "head_dim", None)
            or (model.config.hidden_size // num_heads)
        )
        is_gqa = num_kv_heads < num_heads
        num_repeats = num_heads // num_kv_heads if is_gqa else 1

        config = AttentionConfig(
            chunk_size=chunk_size,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            d_head=d_head,
            is_gqa=is_gqa,
            num_repeats=num_repeats,
        )

        detect_v_neurons(
            hidden_states,
            W_v,
            W_o,
            config,
            neuron_importance,
            param_name_map,
        )

        detect_qk_neurons(
            hidden_states,
            W_q,
            W_k,
            config,
            neuron_importance,
            param_name_map,
        )


def detect_v_neurons(
    hidden_states: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    config: AttentionConfig,
    neuron_importance: defaultdict[tuple[str, int, str], list[float]],
    param_name_map: dict[int, str],
) -> None:
    """Detect V projection neurons using separable norms (paper's approximation).

    Per paper Appendix A.1 (line 697-698): "Since W_V(x) is not in the non-linear
    softmax calculation, we can calculate Imp(W_V|c) by applying Equation (9)."

    Note: This approximation overestimates by ~60% because it ignores attention
    weights. However, it preserves relative ranking and matches the paper.
    """
    V_full = torch.matmul(hidden_states, W_v.T)
    d_v = config.num_kv_heads * config.d_head

    # Paper's Equation 9 approximation: ||V[k]||_2 * ||W_o[:,k]||_2
    # This overestimates by ~60% but preserves relative ranking
    if config.is_gqa:
        V_norms = torch.norm(V_full, p=2, dim=(0, 1))
        W_o_grouped = W_o.view(
            config.d_model, config.num_kv_heads, config.num_repeats, config.d_head
        )
        W_o_effective = W_o_grouped.sum(dim=2).view(config.d_model, d_v)
        W_o_col_norms = torch.norm(W_o_effective, p=2, dim=0)
    else:
        V_norms = torch.norm(V_full, p=2, dim=(0, 1))
        W_o_col_norms = torch.norm(W_o, p=2, dim=0)

    importance_all = (V_norms * W_o_col_norms).tolist()
    record_importance(neuron_importance, param_name_map[id(W_v)], importance_all, 0)


def detect_qk_neurons(
    hidden_states: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    config: AttentionConfig,
    neuron_importance: defaultdict[tuple[str, int, str], list[float]],
    param_name_map: dict[int, str],
) -> None:
    """Detect Q/K neurons using chunked batched softmax difference.

    Per paper Appendix A.1 (lines 698-733):
    - Q detection: iterate over k in [0, d_q) where d_q = num_heads * d_head
    - K detection: iterate over k in [0, d_kv) where d_kv = num_kv_heads * d_head
    """
    # Compute full Q and K projections
    Q_full = torch.matmul(hidden_states, W_q.T)  # [batch, seq, d_q]
    K_full = torch.matmul(hidden_states, W_k.T)  # [batch, seq, d_kv]

    # Reshape to multi-head format for attention computation
    Q_mh = Q_full.view(
        config.batch_size, config.seq_len, config.num_heads, config.d_head
    ).transpose(1, 2)
    K_mh = K_full.view(
        config.batch_size, config.seq_len, config.num_kv_heads, config.d_head
    ).transpose(1, 2)
    if config.is_gqa:
        K_mh = K_mh.repeat_interleave(config.num_repeats, dim=1)

    scores_baseline = torch.matmul(Q_mh, K_mh.transpose(-2, -1)) / (config.d_head**0.5)
    attn_baseline = torch.nn.functional.softmax(scores_baseline, dim=-1)

    # Q detection - iterate over OUTPUT dimensions (d_q = num_heads * d_head)
    d_q = config.num_heads * config.d_head
    q_importance_chunks: list[torch.Tensor] = []
    for start, curr_chunk_size in chunked_range(d_q, config.chunk_size):
        Q_masked = create_masked_batch(Q_full, start, curr_chunk_size)
        Q_masked_mh = Q_masked.view(
            curr_chunk_size,
            config.batch_size,
            config.seq_len,
            config.num_heads,
            config.d_head,
        ).transpose(2, 3)

        scores_mod = torch.matmul(Q_masked_mh, K_mh.transpose(-2, -1)) / (
            config.d_head**0.5
        )
        attn_mod = torch.nn.functional.softmax(scores_mod, dim=-1)
        attn_diff = attn_baseline.unsqueeze(0) - attn_mod
        q_importance_chunks.append(torch.norm(attn_diff.flatten(1), p=2, dim=1))

    q_importance = torch.cat(q_importance_chunks).tolist()
    record_importance(neuron_importance, param_name_map[id(W_q)], q_importance, 0)

    # K detection - iterate over OUTPUT dimensions (d_kv = num_kv_heads * d_head)
    d_kv = config.num_kv_heads * config.d_head
    k_importance_chunks: list[torch.Tensor] = []
    for start, curr_chunk_size in chunked_range(d_kv, config.chunk_size):
        K_masked = create_masked_batch(K_full, start, curr_chunk_size)
        K_masked_mh = K_masked.view(
            curr_chunk_size,
            config.batch_size,
            config.seq_len,
            config.num_kv_heads,
            config.d_head,
        ).transpose(2, 3)

        if config.is_gqa:
            K_masked_mh = K_masked_mh.repeat_interleave(config.num_repeats, dim=2)

        scores_mod = torch.matmul(Q_mh, K_masked_mh.transpose(-2, -1)) / (
            config.d_head**0.5
        )
        attn_mod = torch.nn.functional.softmax(scores_mod, dim=-1)
        attn_diff = attn_baseline.unsqueeze(0) - attn_mod
        k_importance_chunks.append(torch.norm(attn_diff.flatten(1), p=2, dim=1))

    k_importance = torch.cat(k_importance_chunks).tolist()
    record_importance(neuron_importance, param_name_map[id(W_k)], k_importance, 0)


def filter_important_neurons(
    neuron_importance: dict[tuple[str, int, str], list[float]],
    num_samples: int,
    threshold: float,
) -> set[tuple[str, int, str]]:
    """Filter neurons that are consistently important across all samples."""
    logger.info("Filtering consistently important neurons...")
    important_neurons = {
        key
        for key, scores in neuron_importance.items()
        if len(scores) == num_samples and min(scores) >= threshold
    }
    logger.info(f"  Found {len(important_neurons)} consistently important neurons")
    return important_neurons


def detect_raw(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.Dataset,
    is_harmful: bool,
    chunk_size: int = 256,
) -> dict[tuple[str, int, str], list[float]]:
    """Compute raw importance scores for all neurons.

    Returns a dict mapping (param_name, neuron_idx, orientation) to list of
    importance scores (one per sample). Use filter_by_threshold() to convert
    to a set of important neurons.
    """
    model.eval()
    device = next(model.parameters()).device

    # Build param name map BEFORE torch.compile() to get original names
    # (torch.compile adds _orig_mod. prefix which breaks matching during training)
    param_name_map = build_param_name_map(model)

    # Compile model for faster inference (PyTorch 2.0+)
    try:
        # "default" mode: "reduce-overhead" requires fixed input shapes (CUDA graphs)
        # which is incompatible with variable-length sequences in detection.
        model = torch.compile(model, mode="default")
        logger.info("Model compiled with torch.compile for faster inference")
    except Exception as e:
        logger.info(f"torch.compile not available or failed: {e}")

    # Batch tokenize all samples (faster than one at a time)
    logger.info("Tokenizing samples...")
    field = "prompt" if is_harmful else "text"
    all_texts = [example[field] for example in dataset]

    # Tokenize in one batch, then split into individual samples
    batch_encoding = tokenizer(
        all_texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    # Split batch into individual inputs
    all_inputs = [
        {k: v[i : i + 1] for k, v in batch_encoding.items()}
        for i in range(len(dataset))
    ]

    # Register hooks once (not per sample)
    activations: dict[str, torch.Tensor] = {}

    def make_output_hook(name: str):
        """Capture output tensor (for mlp sub-modules)."""

        def hook(_module, _input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()

        return hook

    def make_input_hook(name: str):
        """Capture input hidden_states (post-layernorm, pre-attention).

        This is the tensor that gets projected through Q/K/V weight matrices.
        """

        def hook(_module, args, _output):
            inp = args[0] if isinstance(args, tuple) else args
            activations[name] = inp.detach()

        return hook

    hooks = []
    for name, module in model.named_modules():
        if "mlp.up_proj" in name or "mlp.gate_proj" in name:
            hooks.append(module.register_forward_hook(make_output_hook(name)))
        elif name.endswith("self_attn"):
            hooks.append(module.register_forward_hook(make_input_hook(name)))

    neuron_importance: defaultdict[tuple[str, int, str], list[float]] = defaultdict(
        list
    )

    logger.info("Computing neuron importance via vectorized detection...")
    try:
        with torch.inference_mode():
            for idx, inputs in enumerate(all_inputs):
                if idx % 10 == 0:
                    logger.info(f"  Processing sample {idx}/{len(dataset)}")

                activations.clear()
                model(**inputs)

                _detect_ffn_neurons(
                    model, activations, neuron_importance, param_name_map
                )
                detect_attention_neurons(
                    model, activations, neuron_importance, chunk_size, param_name_map
                )
    finally:
        for hook in hooks:
            hook.remove()

    return neuron_importance


def filter_by_threshold(
    neuron_importance: dict[tuple[str, int, str], list[float]],
    num_samples: int,
    threshold: float,
) -> set[tuple[str, int, str]]:
    """Filter neurons that are consistently important across all samples."""
    return {
        key
        for key, scores in neuron_importance.items()
        if len(scores) == num_samples and min(scores) >= threshold
    }


def cap_neurons_per_param(
    neurons: set[NeuronId],
    neuron_importance: dict[NeuronId, list[float]],
    max_per_param: int,
) -> set[NeuronId]:
    """Cap to at most max_per_param neurons per parameter, keeping the most important.

    Neurons are ranked by their minimum importance score across samples (worst-case
    importance). Applied after detection and deduplication.
    """
    by_param: dict[str, list[tuple[NeuronId, float]]] = {}
    for neuron in neurons:
        scores = neuron_importance.get(neuron)
        min_score = min(scores) if scores else 0.0
        by_param.setdefault(neuron[0], []).append((neuron, min_score))

    result: set[NeuronId] = set()
    for param_name, neuron_scores in by_param.items():
        neuron_scores.sort(key=lambda x: x[1], reverse=True)
        for neuron, _ in neuron_scores[:max_per_param]:
            result.add(neuron)

    logger.info(
        f"  Capped to {max_per_param} neurons/param: {len(neurons)} -> {len(result)}"
    )
    return result


def detect(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.Dataset,
    is_harmful: bool,
    importance_threshold: float,
    chunk_size: int = 256,
) -> tuple[set[NeuronId], dict[NeuronId, list[float]]]:
    """Detect important neurons using vectorized batched computation.

    Returns:
        Tuple of (important_neurons, raw_importance_scores). The raw scores are
        needed for cap_neurons_per_param() to rank neurons when capping.
    """
    neuron_importance = detect_raw(model, tokenizer, dataset, is_harmful, chunk_size)
    neurons = filter_important_neurons(
        neuron_importance, len(dataset), importance_threshold
    )
    return neurons, neuron_importance
