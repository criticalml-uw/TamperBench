"""Vectorized batched neuron importance detection.

Implements the parallel neuron detection method from:
    Zhao et al. (2025) "Understanding and Enhancing Safety Mechanisms of LLMs via
    Safety-Specific Neuron" (ICLR 2025)

This module detects "safety neurons" - neurons that are consistently important for
processing harmful queries. A neuron is important if deactivating it (zeroing its
parameters) significantly changes the model output.

Mathematical Basis (Paper Appendix A.1):
    - FFN neurons: The parallel importance formula (Eq. 9) is
      ``Imp(W_up|c) = ||(h_ffn · Mask) W_down||_2``, where h_ffn = silu(gate)*up.
      Because the diagonal mask isolates one dimension at a time, this decomposes
      into ``||h_ffn[:,:,k]||_2 * ||W_down[:,k]||_2`` (separable norms).
    - V neurons: The paper notes V is outside the softmax nonlinearity, so the
      same separable-norm approach (Eq. 9) applies to the V/O path:
      ``||V[:,:,k]||_2 * ||W_o[:,k]||_2``. This ignores attention weights
      (overestimates, but preserves relative ranking).
    - Q/K neurons (Equations 12-14): Uses chunked batched softmax difference
      computation. For each output dimension k, zero it in Q (or K), recompute
      softmax, and measure the L2 difference from baseline.

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

    Set ``match_original_code=True`` on ``RSNTuneConfig`` to use
    ``detect_original()`` instead of ``detect()``.

    **Toggled by match_original_code** (paper vs original codebase disagree):

    1. **Neuron selection**: Paper (Eq 2-3): threshold-based filtering
       (importance >= epsilon on ALL samples). Original: top-K per sample
       (12,000 FFN, 2,000 attention per layer) then set intersection. Both
       select "consistently important" neurons but define importance differently.
       The paper never specifies epsilon.

    2. **FFN importance metric**: Paper (Eq 8-9): separable norms
       ||h_ffn[:,:,k]||_2 * ||W_down[:,k]||_2 where h_ffn = silu(gate)*up.
       Original: sum(abs(up_proj(x)), dim=seq) — L1 of up_proj output only,
       ignoring gate_proj and down_proj weights.

    3. **V importance metric**: Paper (Appendix A.1): separable norms on V/O
       path (same as FFN, since V is outside the softmax). Original:
       sum(abs(attn_output), dim=seq) — L1 of attention output before o_proj.

    4. **Q/K importance**: ``detect()`` implements the paper's Eq 11-14
       (pre-RoPE, separate Q and K scores, proper scaling). ``detect_original()``
       faithfully reproduces the original code's post-RoPE softmax-diff
       computation, including two things that may be bugs:
       (a) **Scaling bug**: unscaled per-dimension Q*K outer products are
       subtracted from the scaled full logits (Q@K^T/sqrt(d)), mixing scaled
       and unscaled quantities. The paper's Eq 11 defines Delta_k as a
       difference of identically-scaled terms.
       (b) **Same score for Q and K**: the original code returns the same
       importance tensor for both. The paper says to compute them separately
       ("W_K is symmetrical to W_Q", implying a separate K computation).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import datasets
import torch
from torch import nn
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


def create_masked_batch(tensor: torch.Tensor, start: int, chunk_size: int) -> torch.Tensor:
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

    Records importance for up_proj and down_proj only. gate_proj is excluded
    from training (matching the original codebase).
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

        W_down: torch.Tensor = module.down_proj.weight  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]
        W_up: torch.Tensor = module.up_proj.weight  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]

        # Vectorized: compute all norms at once
        h_norms = torch.norm(h_ffn, p=2, dim=(0, 1))
        W_col_norms = torch.norm(W_down, p=2, dim=0)
        importance_all = (h_norms * W_col_norms).tolist()

        # Record importance for up_proj and down_proj only (not gate_proj,
        # matching the original code which zeros gate_proj gradients entirely).
        # down_proj: column orientation (zeroing column k of down_proj)
        # up_proj: row orientation (zeroing row k, i.e., output dim k)
        record_importance(neuron_importance, param_name_map[id(W_down)], importance_all, 0, "column")
        record_importance(neuron_importance, param_name_map[id(W_up)], importance_all, 0, "row")


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

        W_q: torch.Tensor = module.q_proj.weight  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]
        W_k: torch.Tensor = module.k_proj.weight  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]
        W_v: torch.Tensor = module.v_proj.weight  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]
        W_o: torch.Tensor = module.o_proj.weight  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]

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

    Note: This approximation overestimates importance because it ignores
    attention weights, but it preserves relative ranking and matches the paper.
    """
    V_full = torch.matmul(hidden_states, W_v.T)
    d_v = config.num_kv_heads * config.d_head

    # Paper's Equation 9 approximation: ||V[k]||_2 * ||W_o[:,k]||_2
    if config.is_gqa:
        V_norms = torch.norm(V_full, p=2, dim=(0, 1))
        W_o_grouped = W_o.view(config.d_model, config.num_kv_heads, config.num_repeats, config.d_head)
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
    Q_mh = Q_full.view(config.batch_size, config.seq_len, config.num_heads, config.d_head).transpose(1, 2)
    K_mh = K_full.view(config.batch_size, config.seq_len, config.num_kv_heads, config.d_head).transpose(1, 2)
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

        scores_mod = torch.matmul(Q_masked_mh, K_mh.transpose(-2, -1)) / (config.d_head**0.5)
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

        scores_mod = torch.matmul(Q_mh, K_masked_mh.transpose(-2, -1)) / (config.d_head**0.5)
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
        key for key, scores in neuron_importance.items() if len(scores) == num_samples and min(scores) >= threshold
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
    compiled = False
    try:
        model = torch.compile(model, mode="default")  # pyright: ignore[reportAssignmentType]
        compiled = True
        logger.info("Model compiled with torch.compile for faster detection")
    except Exception as e:
        logger.info(f"torch.compile not available or failed: {e}")

    # Batch tokenize all samples (faster than one at a time)
    logger.info("Tokenizing samples...")
    field = "prompt" if is_harmful else "text"
    all_texts: list[str] = [example[field] for example in dataset]  # pyright: ignore[reportCallIssue,reportArgumentType]

    # Tokenize in one batch, then split into individual samples
    batch_encoding = tokenizer(
        all_texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    # Split batch into individual inputs
    all_inputs = [{k: v[i : i + 1] for k, v in batch_encoding.items()} for i in range(len(dataset))]

    # Register hooks once (not per sample)
    activations: dict[str, torch.Tensor] = {}

    def make_output_hook(name: str):
        """Capture output tensor (for mlp sub-modules)."""

        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()

        return hook

    def make_input_hook(name: str):
        """Capture input hidden_states (post-layernorm, pre-attention).

        This is the tensor that gets projected through Q/K/V weight matrices.
        """

        def hook(module: nn.Module, _args: Any, kwargs: dict[str, Any], _output: Any) -> None:
            inp = kwargs["hidden_states"]
            q_proj: nn.Linear = module.q_proj  # pyright: ignore[reportAssignmentType]
            assert inp.dim() == 3 and inp.shape[-1] == q_proj.in_features, (
                f"Expected [batch, seq, {q_proj.in_features}] hidden_states as kwarg to {name}, got shape {inp.shape}"
            )
            activations[name] = inp.detach()

        return hook

    hooks = []
    for name, module in model.named_modules():
        if "mlp.up_proj" in name or "mlp.gate_proj" in name:
            hooks.append(module.register_forward_hook(make_output_hook(name)))
        elif name.endswith("self_attn"):
            hooks.append(module.register_forward_hook(make_input_hook(name), with_kwargs=True))

    neuron_importance: defaultdict[tuple[str, int, str], list[float]] = defaultdict(list)

    logger.info("Computing neuron importance via vectorized detection...")
    try:
        with torch.inference_mode():
            for idx, inputs in enumerate(all_inputs):
                if idx % 10 == 0:
                    logger.info(f"  Processing sample {idx}/{len(dataset)}")

                activations.clear()
                model(**inputs)

                _detect_ffn_neurons(model, activations, neuron_importance, param_name_map)
                detect_attention_neurons(model, activations, neuron_importance, chunk_size, param_name_map)
    finally:
        for hook in hooks:
            hook.remove()
        if compiled:
            # Free compiled model from memory
            # (https://github.com/pytorch/pytorch/issues/105181)
            torch._dynamo.reset()  # pyright: ignore[reportPrivateUsage]

    return neuron_importance


def filter_by_threshold(
    neuron_importance: dict[tuple[str, int, str], list[float]],
    num_samples: int,
    threshold: float,
) -> set[tuple[str, int, str]]:
    """Filter neurons that are consistently important across all samples."""
    return {key for key, scores in neuron_importance.items() if len(scores) == num_samples and min(scores) >= threshold}


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
    for _param_name, neuron_scores in by_param.items():
        neuron_scores.sort(key=lambda x: x[1], reverse=True)
        for neuron, _ in neuron_scores[:max_per_param]:
            result.add(neuron)

    logger.info(f"  Capped to {max_per_param} neurons/param: {len(neurons)} -> {len(result)}")
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
    neurons = filter_important_neurons(neuron_importance, len(dataset), importance_threshold)
    return neurons, neuron_importance


def _compute_qk_importance_original(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Compute Q/K importance scores matching the original codebase.

    Reproduces the original code's post-RoPE softmax-diff computation. Known
    discrepancies from the paper's Eq 11-14 that we reproduce faithfully:

    - **Scaling bug**: per-dimension Q*K outer products are unscaled, but they
      are subtracted from the scaled full attention logits (Q@K^T/sqrt(d)),
      mixing scaled and unscaled quantities.
    - **Same score for Q and K**: the original returns the same importance
      tensor for both query_score and key_score (likely a bug — the paper
      prescribes separate Q and K computations via Eq 12 and the note that
      "W_K is symmetrical to W_Q").

    Args:
        attn_module: The self_attn module (e.g. LlamaAttention).
        hidden_states: Input to attention, [batch, seq, hidden_size].
        position_embeddings: (cos, sin) for RoPE.

    Returns:
        Importance scores of shape [num_heads * d_head], same for Q and K.
    """
    import math

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    bsz, seq_len, _ = hidden_states.size()
    num_heads: int = attn_module.num_heads  # pyright: ignore[reportAssignmentType]
    num_kv_heads: int = attn_module.num_key_value_heads  # pyright: ignore[reportAssignmentType]
    d_head: int = attn_module.head_dim  # pyright: ignore[reportAssignmentType]
    num_kv_groups: int = attn_module.num_key_value_groups  # pyright: ignore[reportAssignmentType]

    # Project and reshape to multi-head
    query_states = attn_module.q_proj(hidden_states)  # pyright: ignore[reportCallIssue]
    key_states = attn_module.k_proj(hidden_states)  # pyright: ignore[reportCallIssue]
    query_states = query_states.view(bsz, seq_len, num_heads, d_head).transpose(1, 2)
    key_states = key_states.view(bsz, seq_len, num_kv_heads, d_head).transpose(1, 2)

    # Apply RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # GQA expansion
    key_states = repeat_kv(key_states, num_kv_groups)

    # Full attention logits (scaled) + causal mask
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(d_head)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype),
        diagonal=1,
    )
    attn_weights = attn_weights + causal_mask

    # Per-dimension outer products (NOT scaled — this is a bug in the original
    # code; the paper's Eq 11 scales Delta_k by 1/sqrt(d)):
    # [batch, heads, d_head, seq, seq]
    attn_weights_temp = torch.matmul(
        query_states.transpose(2, 3).unsqueeze(-1),
        key_states.transpose(2, 3).unsqueeze(-1).transpose(-2, -1),
    )

    # Subtract unscaled per-dim contribution from scaled+masked full logits
    attn_weights_temp = attn_weights.unsqueeze(2).expand_as(attn_weights_temp) - attn_weights_temp

    # Softmax of "logits with dim k removed"
    attn_weights_temp = torch.nn.functional.softmax(attn_weights_temp, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # Baseline softmax (from masked+scaled logits)
    attn_baseline = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Squared diff, summed over (seq, seq) dims
    attn_weights_temp = attn_weights_temp - attn_baseline.unsqueeze(2).expand_as(attn_weights_temp)
    attn_weights_temp = (attn_weights_temp**2).sum(dim=(-2, -1))  # [batch, heads, d_head]

    return attn_weights_temp.view(-1)  # [num_heads * d_head]


def detect_original(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.Dataset,
    is_harmful: bool,
    top_k_ffn: int = 12000,
    top_k_attn: int = 2000,
) -> tuple[set[NeuronId], dict[NeuronId, list[float]]]:
    """Detect important neurons using the original codebase's algorithm.

    Differences from ``detect()``:
    - Uses L1-norm importance metrics (not separable norms from the paper)
    - Uses top-K per sample + set intersection (not threshold-based filtering)
    - FFN: ``sum(abs(up_proj(x)), dim=seq)`` (same score for up_proj and down_proj)
    - V: ``sum(abs(attn_output_before_o_proj), dim=seq)``
    - Q/K: post-RoPE softmax-diff (same score for Q and K, matching original code)
    - No gate_proj detection

    Returns:
        Tuple of (important_neurons, raw_importance_scores).
    """
    model.eval()
    device = next(model.parameters()).device
    param_name_map = build_param_name_map(model)

    # Tokenize
    field = "prompt" if is_harmful else "text"
    all_texts: list[str] = [example[field] for example in dataset]  # pyright: ignore[reportCallIssue,reportArgumentType]
    batch_encoding = tokenizer(
        all_texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)
    all_inputs = [{k: v[i : i + 1] for k, v in batch_encoding.items()} for i in range(len(dataset))]

    # Register hooks to capture activations
    activations: dict[str, torch.Tensor] = {}
    # Capture self_attn inputs for Q/K importance computation
    attn_inputs: dict[str, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = {}

    def make_ffn_hook(name: str):
        """Capture up_proj output for L1 importance."""

        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            out = output[0] if isinstance(output, tuple) else output
            activations[name] = out.detach()

        return hook

    def make_o_proj_input_hook(name: str):
        """Capture input to o_proj (= attention output before o_proj) for V importance."""

        def hook(_module: nn.Module, args: Any, _output: Any) -> None:
            inp = args[0] if isinstance(args, tuple) else args
            activations[name] = inp.detach()

        return hook

    def make_attn_input_hook(name: str):
        """Capture self_attn inputs (hidden_states, position_embeddings) for Q/K."""

        def hook(_module: nn.Module, _args: Any, kwargs: dict[str, Any], _output: Any) -> None:
            attn_inputs[name] = (
                kwargs["hidden_states"].detach(),
                kwargs["position_embeddings"],
            )

        return hook

    hooks = []
    for name, module in model.named_modules():
        if "mlp.up_proj" in name:
            hooks.append(module.register_forward_hook(make_ffn_hook(name)))
        elif name.endswith("self_attn.o_proj"):
            hooks.append(module.register_forward_hook(make_o_proj_input_hook(name)))
        elif name.endswith("self_attn"):
            hooks.append(module.register_forward_hook(make_attn_input_hook(name), with_kwargs=True))

    # Per-sample top-K sets, keyed by (param_name, orientation)
    per_sample_topk: dict[tuple[str, str], list[set[int]]] = {}
    raw_importance: dict[NeuronId, list[float]] = {}

    def _record_topk(param_name: str, orient: str, scores: torch.Tensor, top_k: int) -> None:
        scores_list = scores.tolist()
        top_indices = torch.argsort(scores, descending=True)[:top_k].tolist()
        key = (param_name, orient)
        per_sample_topk.setdefault(key, []).append(set(top_indices))
        for i in top_indices:
            raw_importance.setdefault((param_name, i, orient), []).append(scores_list[i])

    logger.info("Computing neuron importance (original algorithm)...")
    try:
        with torch.inference_mode():
            for idx, inputs in enumerate(all_inputs):
                if idx % 10 == 0:
                    logger.info(f"  Processing sample {idx}/{len(dataset)}")

                activations.clear()
                attn_inputs.clear()
                model(**inputs)

                # FFN: L1 of up_proj output, summed across sequence dim
                for act_name, act_tensor in activations.items():
                    if "up_proj" not in act_name:
                        continue

                    scores = torch.sum(torch.abs(act_tensor), dim=1).squeeze()
                    layer_prefix = act_name.rsplit(".", 1)[0]
                    up_param_name = None
                    down_param_name = None
                    for _pid, pname in param_name_map.items():
                        if pname.startswith(layer_prefix):
                            if "up_proj.weight" in pname:
                                up_param_name = pname
                            elif "down_proj.weight" in pname:
                                down_param_name = pname

                    if up_param_name is None or down_param_name is None:
                        continue

                    _record_topk(up_param_name, "row", scores, top_k_ffn)
                    _record_topk(down_param_name, "column", scores, top_k_ffn)

                # V: L1 of attention output before o_proj
                for act_name, act_tensor in activations.items():
                    if "o_proj" not in act_name:
                        continue

                    v_scores = torch.sum(torch.abs(act_tensor), dim=1).squeeze()
                    attn_prefix = act_name.rsplit(".", 1)[0]

                    v_param_name = None
                    for _pid, pname in param_name_map.items():
                        if pname.startswith(attn_prefix) and "v_proj.weight" in pname:
                            v_param_name = pname
                    if v_param_name is not None:
                        _record_topk(v_param_name, "row", v_scores, top_k_attn)

                # Q/K: post-RoPE softmax-diff (same score for both)
                for attn_name, (hidden_states, pos_emb) in attn_inputs.items():
                    # Find the self_attn module
                    attn_module = None
                    for mname, module in model.named_modules():
                        if mname == attn_name:
                            attn_module = module
                            break
                    if attn_module is None:
                        continue

                    qk_scores = _compute_qk_importance_original(attn_module, hidden_states, pos_emb)

                    # Find q_proj and k_proj param names
                    for _pid, pname in param_name_map.items():
                        if pname.startswith(attn_name):
                            if "q_proj.weight" in pname:
                                _record_topk(pname, "row", qk_scores, top_k_attn)
                            elif "k_proj.weight" in pname:
                                # Same score for K as Q (matching original code)
                                _record_topk(pname, "row", qk_scores, top_k_attn)

    finally:
        for hook in hooks:
            hook.remove()

    # Set intersection across all samples
    neurons: set[NeuronId] = set()
    for (param_name, orient), sample_sets in per_sample_topk.items():
        if len(sample_sets) < len(dataset):
            continue
        common = set.intersection(*sample_sets)
        for i in common:
            neurons.add((param_name, i, orient))

    logger.info(f"  Detected {len(neurons)} neurons via top-K intersection")
    return neurons, raw_importance
