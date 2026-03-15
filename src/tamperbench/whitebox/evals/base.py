"""Template for evaluations in tamperbench that evaluate potency of attacks."""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, final

import polars as pl
import torch
from pandera.typing.polars import DataFrame
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from vllm import LLM

from tamperbench.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from tamperbench.whitebox.evals.reference import ReferenceScores
from tamperbench.whitebox.utils import (
    EvalName,
    EvalPath,
    MetricName,
    OptimizationDirection,
)
from tamperbench.whitebox.utils.models.config import ModelConfig

C = TypeVar(name="C", bound="WhiteBoxEvaluationConfig")


@dataclass
class WhiteBoxEvaluationConfig:
    """Configuration for evaluations.

    Attributes:
        model_checkpoint: Path to the checkpoint of the pretrained HuggingFace model to evaluate.
        out_dir: The path to directory where evaluation results and artifacts will be stored (may also have model checkpoints).
        batch_size: Batch size for running inference.
        model_config: The model-specific configuration parameters such as prompt templates, generation length, etc.
    """

    model_checkpoint: str
    out_dir: str
    model_config: ModelConfig
    batch_size: int = 32


class WhiteBoxEvaluation(ABC, Generic[C]):
    """Base class for an evaluation."""

    name: EvalName
    """Name of evaluation."""

    objective: MetricName
    """Name of metric that is optimized for this evaluation in hyper-parameter searches."""

    attacker_direction: OptimizationDirection
    """Direction of objective that an attacker optimizes for."""

    defender_direction: OptimizationDirection
    """Direction of objective that a defender optimizes for."""

    reference_scores: ReferenceScores = ReferenceScores()
    """Expected scores from published papers for validation. Subclasses should override."""

    def __init__(self, eval_config: C) -> None:
        """Construct base WhiteBoxEvaluation.

        Args:
            eval_config: Configuration for the evaluation, e.g. model paths.
        """
        self.eval_config: C = eval_config
        self.initialize_paths()

    def run_evaluation(self) -> DataFrame[EvaluationSchema]:
        """Run evaluation and report metrics.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the aggregated evaluation metrics.
        """
        _ = self.load_inferences()
        _ = self.load_scores()

        return self.load_results()

    @abstractmethod
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on evaluation dataset to compute the inferences.

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with the prompts to LLM and corresponding responses.
        """

    @abstractmethod
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Run evaluator on model inferences to compute evaluation scores on data points.

        Args:
            inferences: A polars DataFrame with the prompts to LLM and corresponding responses.

        Returns:
            DataFrame[ScoreSchema]: A polars DataFrame with the model inferences scored by evaluator.
        """

    @abstractmethod
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Aggregate evaluator outputs into overall evaluation metrics.

        Takes the per-datapoint evaluator scores and compiles them into higher-level metrics that
        characterize the model's performance across the full evaluation dataset.

        Args:
            scores: A polars DataFrame with the model inferences scored by evaluator.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the metrics and scores of the full evaluation.
        """

    @final
    def initialize_paths(self) -> None:
        """Initialize the paths to directories and files for the evaluation where evaluation data is to be saved.

        This facilitates storage and logging of data associated with the evaluation.

        The following paths are initialized:
            - `save_dir_path`: This is the directory (in the checkpoint directory) where
                all files associated with the evaluation will be saved.
            - `inferences_file_path`: This is the file path (.parquet) where all inferences are stored.
            - `scores_file_path`: This is is the file path (.parquet) where inferences are scored by evaluator.
            - `evaluation_file_path`: This is the file path (.json) with the final aggregated evaluation value(s).
        """
        self.save_dir: Path = Path(self.eval_config.out_dir, EvalPath.CHECKPOINT_EVAL_DIR, self.name)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.inferences_file_path: Path = self.save_dir / EvalPath.INFERENCES
        self.scores_file_path: Path = self.save_dir / EvalPath.SCORES
        self.evaluation_file_path: Path = self.save_dir / EvalPath.EVALUATION

    @final
    def load_inferences(self) -> DataFrame[InferenceSchema]:
        """Load the model inferences of the evaluation dataset into a polars DataFrame.

        Returns:
            DataFrame[InferenceSchema]: A polars DataFrame with the prompts to LLM and corresponding responses.
        """
        if not self.inferences_file_path.exists():
            inferences: DataFrame[InferenceSchema] = self.compute_inferences()
            inferences.write_parquet(file=self.inferences_file_path)

        _inferences: pl.DataFrame = pl.read_parquet(source=self.inferences_file_path)
        return InferenceSchema.validate(_inferences)

    @final
    def load_scores(self) -> DataFrame[ScoreSchema]:
        """Load the per-datapoint evaluation of the model inferences into a polars DataFrame.

        Returns:
            DataFrame[ScoreSchema]: A polars DataFrame with the model inferences scored by evaluator.
        """
        if not self.scores_file_path.exists():
            inferences: DataFrame[InferenceSchema] = self.load_inferences()

            scores: DataFrame[ScoreSchema] = self.compute_scores(inferences)
            scores.write_parquet(file=self.scores_file_path)

        _scores: pl.DataFrame = pl.read_parquet(source=self.scores_file_path)
        return ScoreSchema.validate(_scores)

    @final
    def load_results(self) -> DataFrame[EvaluationSchema]:
        """Load the aggregated evaluation metrics into a polars DataFrame.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the aggregated evaluation metrics.
        """
        if not self.evaluation_file_path.exists():
            scores: DataFrame[ScoreSchema] = self.load_scores()

            results: DataFrame[EvaluationSchema] = self.compute_results(scores)
            results.write_json(file=self.evaluation_file_path)

        _results: pl.DataFrame = pl.read_json(source=self.evaluation_file_path)
        return EvaluationSchema.validate(_results)

    @final
    @classmethod
    def load_result_objective(cls, results: DataFrame[EvaluationSchema]) -> float:
        """Load and get the value of the metric that is used for optimization when using this eval.

        Note: This differs from `load_results` as it returns one float value to guide optimization,
            as opposed to a dataframe of potentially multiple metrics.

        Args:
            results: The polars DataFrame with the aggregated evaluation metrics.

        Returns:
            float: Numerical value of the result.
        """
        result_metrics: dict[str, tuple[float]] = results.rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )
        return result_metrics[cls.objective][0]


def load_tokenizer(eval_config: WhiteBoxEvaluationConfig) -> PreTrainedTokenizer:
    """Load HuggingFace tokenizer from eval config.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.

    Returns:
        HuggingFace tokenizer.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=str(
            eval_config.model_config.tokenizer_checkpoint or eval_config.model_checkpoint
        ),
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer


def load_vllm_model_and_tokenizer(
    eval_config: WhiteBoxEvaluationConfig,
) -> tuple[LLM, PreTrainedTokenizer]:
    """Load a vLLM model and HF tokenizer from eval config.

    Use this for evals that generate via vLLM (most evals). For evals that
    need the HuggingFace model directly (e.g., MT-Bench), use
    load_hf_model_and_tokenizer instead.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.

    Returns:
        Tuple of (vLLM LLM instance, HuggingFace tokenizer).
    """
    llm_kwargs: dict[str, Any] = {
        "model": eval_config.model_checkpoint,
        "tensor_parallel_size": (torch.cuda.device_count() if torch.cuda.is_available() else 1),
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
    }
    if eval_config.model_config.tokenizer_checkpoint is not None:
        llm_kwargs["tokenizer"] = eval_config.model_config.tokenizer_checkpoint

    model = LLM(**llm_kwargs)
    tokenizer = load_tokenizer(eval_config)
    return model, tokenizer


def load_hf_model_and_tokenizer(
    eval_config: WhiteBoxEvaluationConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load HuggingFace model and tokenizer.

    Use this for evals that need direct HuggingFace model access (e.g., MT-Bench
    which uses transformers generate). For vLLM-based generation, use
    load_vllm_model_and_tokenizer instead.

    Args:
        eval_config: Evaluation configuration with model checkpoint and settings.

    Returns:
        Tuple of (HuggingFace PreTrainedModel, HuggingFace tokenizer).
    """
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=eval_config.model_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    tokenizer = load_tokenizer(eval_config)
    return model, tokenizer


def load_lofit_model_and_tokenizer(
    model_checkpoint: str,
    base_model_name: str,
    applied_module: str = "attention",
    applied_layers: list[int] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load LoFiT model and tokenizer with activation modifications.

    Auto-detects Llama vs Gemma architecture from the checkpoint config.

    Args:
        model_checkpoint: Path to the LoFiT model checkpoint.
        base_model_name: HF model ID for loading the base model and tokenizer (LoFiT
            checkpoints may not include tokenizer files).
        applied_module: Module to apply LoFiT edits to ("attention" or "mlp").
        applied_layers: Specific layers to apply edits (None = all layers).

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]:
            - A model loaded with LoFiT activation modifications applied via forward hooks
            - The associated tokenizer
    """
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    config = AutoConfig.from_pretrained(model_checkpoint)
    model_type = getattr(config, "model_type", "").lower()
    lofit_state: dict = {}

    if "gemma" in model_type:
        from tamperbench.whitebox.attacks.lofit.vendor.models.modeling_gemma import (
            GemmaForCausalLM,
        )

        model = GemmaForCausalLM.custom_from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            cache_dir=None,
            applied_module=applied_module,
            applied_layers=applied_layers,
            torch_dtype=torch_dtype,
        ).eval()
    else:
        import json as _json
        import os as _os

        model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch_dtype)

        # Load LoFiT weights (attn_A, attn_v) directly from checkpoint safetensors/bin.
        # AutoModelForCausalLM ignores these vendor-specific keys so we read them directly.
        _ckpt = model_checkpoint
        _idx = _os.path.join(_ckpt, "model.safetensors.index.json")
        _single = _os.path.join(_ckpt, "model.safetensors")
        _bin = _os.path.join(_ckpt, "pytorch_model.bin")
        if _os.path.exists(_idx):
            from safetensors.torch import load_file as _load_sf
            with open(_idx) as _f:
                _wmap = _json.load(_f)["weight_map"]
            _shards = {v for k, v in _wmap.items() if "attn_A" in k or "attn_v" in k}
            for _shard in _shards:
                _d = _load_sf(_os.path.join(_ckpt, _shard))
                lofit_state.update({k: v for k, v in _d.items() if "attn_A" in k or "attn_v" in k})
        elif _os.path.exists(_single):
            from safetensors.torch import load_file as _load_sf
            _d = _load_sf(_single)
            lofit_state = {k: v for k, v in _d.items() if "attn_A" in k or "attn_v" in k}
        elif _os.path.exists(_bin):
            _d = torch.load(_bin, map_location="cpu")
            lofit_state = {k: v for k, v in _d.items() if "attn_A" in k or "attn_v" in k}

        # Apply LoFiT via forward pre-hooks on o_proj.
        # LoFiT modifies the per-head attention output as: (A+1)*x + v
        def _make_lofit_hook(A: torch.Tensor, v: torch.Tensor, num_heads: int, head_dim: int):
            def _hook(module, args):
                x = args[0]  # (bsz, q_len, hidden_size)
                bsz, q_len, _ = x.shape
                _A = A.to(device=x.device, dtype=x.dtype)
                _v = v.to(device=x.device, dtype=x.dtype)
                x = ((_A + 1) * x.view(bsz, q_len, num_heads, head_dim) + _v).view(bsz, q_len, -1)
                return (x,)
            return _hook

        num_layers = len(model.model.layers)
        layers_to_hook = applied_layers if applied_layers is not None else list(range(num_layers))
        _cfg = model.config
        num_heads = _cfg.num_attention_heads
        head_dim = _cfg.hidden_size // num_heads
        for layer_idx in layers_to_hook:
            attn = model.model.layers[layer_idx].self_attn
            A_tensors = [
                lofit_state.get(f"model.layers.{layer_idx}.self_attn.attn_A.{h}", torch.zeros(head_dim))
                for h in range(num_heads)
            ]
            v_tensors = [
                lofit_state.get(f"model.layers.{layer_idx}.self_attn.attn_v.{h}", torch.zeros(head_dim))
                for h in range(num_heads)
            ]
            A = torch.stack(A_tensors)  # (num_heads, head_dim)
            v = torch.stack(v_tensors)  # (num_heads, head_dim)
            if A.abs().max() > 0 or v.abs().max() > 0:
                attn.o_proj.register_forward_pre_hook(_make_lofit_hook(A, v, num_heads, head_dim))

        model = model.eval()

    model.config.use_cache = False  # pyright: ignore[reportAttributeAccessIssue]
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.use_cache = False

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="left",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    return model, tokenizer
