# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false
"""Model wrapper classes for HuggingFace and vLLM backends."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.utils import HfHubHTTPError  # type: ignore
from jaxtyping import Float
from peft.peft_model import PeftModel
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from typing_extensions import override
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput

from .configs import Backend, ModelConfig
from .model_family_config import (
    ModelFamilyConfig,
    get_model_family_config_class,
)
from .model_utils import (
    download_model_from_huggingface,
    is_huggingface_path,
    is_launched_with_accelerate,
)

logger = logging.getLogger(__name__)


class WrappedModelBase(ABC):
    """Base class for wrapped models providing a common interface."""

    _eoi_string: str | None = None

    def __init__(self, model_config: ModelConfig):
        """Initialize base attributes for all model types.

        This method handles setting up the basic attributes that all models need.
        Subclasses are responsible for loading the actual model and tokenizer.

        Args:
            model_config: Configuration for the model
            device: Device to run the model on (e.g., 'cuda' or 'cpu')
        """
        # Basic attributes
        self.model: Any = None
        self.tokenizer: Any = None
        self.short_name: str = model_config.short_name
        self.name_or_path: Path = model_config.name_or_path
        assert model_config.tokenizer_name_or_path
        self.tokenizer_name_or_path: Path = model_config.tokenizer_name_or_path
        self.original_name_or_path: Path = model_config.name_or_path
        self._assistant_prefix: str | None = None

        # Get the appropriate model family configuration for this model
        self.model_family_config_class: type[ModelFamilyConfig] = get_model_family_config_class(
            model_config.model_family, model_config.name_or_path, model_config.short_name
        )

    def _setup_tokenizer(self) -> None:
        """Load a tokenizer from a local path or the Hugging Face Hub.

        Falls back from `tokenizer_name_or_path` to `original_name_or_path`.
        """
        for candidate in (self.tokenizer_name_or_path, self.original_name_or_path):
            # --- decide whether the string looks usable ---
            try:
                is_huggingface_path(Path(candidate))  # may log & raise
            except HfHubHTTPError:  # token/bad repo
                continue  # keep trying

            # --- actual load attempt ---
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                self.tokenizer_name_or_path = candidate
                logger.info(f"Loaded tokenizer from {candidate}")
                break
            except (OSError, FileNotFoundError, ValueError) as err:
                logger.warning(f"Failed to load tokenizer from {candidate}: {err}")
        else:
            # reraise the last error
            logger.error(f"Failed to load tokenizer from {self.tokenizer_name_or_path} or {self.original_name_or_path}")
            raise OSError(
                f"Failed to load tokenizer from {self.tokenizer_name_or_path} or {self.original_name_or_path}"
            )

        # model family specific tokenizer setup
        self.model_family_config_class.set_special_tokens(self.tokenizer)
        self.model_family_config_class.set_chat_template(self.tokenizer)

    @abstractmethod
    def get_num_parameters(self) -> tuple[int, int]:
        """Get the number of parameters in the model."""
        pass

    @property
    def assistant_prefix(self) -> str:
        """Get the assistant prefix from the tokenizer's chat template.

        This is used for training to ensure the model only learns from the assistant's
        part of the conversation.

        Returns:
            The assistant prefix string
        """
        if not hasattr(self, "_assistant_prefix") or self._assistant_prefix is None:
            # Cache the result to avoid recomputing it
            self._assistant_prefix = self.model_family_config_class.get_assistant_prefix(self.tokenizer)
            logger.info(f"Using assistant prefix: '{self._assistant_prefix}'")
        assert isinstance(self._assistant_prefix, str), (
            f"Expected assistant prefix to be a string, got {type(self._assistant_prefix)}"
        )

        return self._assistant_prefix

    @property
    def eoi_string(self) -> str:
        """Get the end-of-input string from the tokenizer's chat template."""
        if not hasattr(self, "_eoi_string") or self._eoi_string is None:
            # Cache the result to avoid recomputing it
            self._eoi_string = self.model_family_config_class.get_eoi_string(self.tokenizer)

        assert isinstance(self._eoi_string, str), (
            f"Expected end-of-input string to be a string, got {type(self._eoi_string)}"
        )
        return self._eoi_string


class WrappedHFModel(WrappedModelBase):
    """Wrapper for HuggingFace Transformers models."""

    def __init__(self, model_config: ModelConfig, **kwargs):
        """Initialize a HuggingFace model wrapper."""
        self.name_or_path: Path = model_config.name_or_path
        name_or_path_str = str(self.name_or_path)
        _cache_dir_str = str(model_config.download_dir) if model_config.download_dir else None

        assert model_config.download_dir, "Download directory must be specified"
        super().__init__(model_config)

        # Download the model
        if is_huggingface_path(self.name_or_path):
            download_model_from_huggingface(self.name_or_path, model_config.download_dir)

        # if running with accelerate set device_map to None else use auto
        # check if running with accelerate
        if is_launched_with_accelerate():
            device_map = None
        else:
            device_map = "auto"

        # Load model
        self.model: AutoModelForCausalLM | PeftModel = AutoModelForCausalLM.from_pretrained(
            name_or_path_str,
            torch_dtype=torch.bfloat16,
            cache_dir=model_config.download_dir,
            device_map=device_map,
            **kwargs,
        )

        if self.name_or_path.is_dir():
            # Check if adapter_config.json exists
            # Load adapters if they exist
            adapter_config_path = self.name_or_path / "adapter_config.json"
            if adapter_config_path.exists():
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.name_or_path,
                    cache_dir=model_config.download_dir,
                    device_map=device_map,
                    is_trainable=True,
                    **kwargs,
                )

                # Explicitly convert all parameters to bfloat16
                logger.info("Converting all parameters to bfloat16")
                for param_name, param in self.model.named_parameters():
                    if param.dtype != torch.bfloat16:
                        logger.debug(f"Converting {param_name} from {param.dtype} to bfloat16")
                        param.data = param.data.to(torch.bfloat16)

            # set original_name_or_path
            model_config_path = self.name_or_path / "config.json"
            if model_config_path.exists():
                model_config_json: dict[str, Any] = json.loads(model_config_path.read_text())
                self.original_name_or_path: Path = Path(model_config_json["_name_or_path"])
                logger.info(f"Original model path set to {self.original_name_or_path}")

        # Set up the tokenizer
        self._setup_tokenizer()

    @override
    def get_num_parameters(self) -> tuple[int, int]:
        """Get the number of parameters in the model."""
        n_params = self.model.num_parameters()
        n_trainable_params = self.model.num_parameters(only_trainable=True)
        return n_params, n_trainable_params

    def generate(self, _model_input: list[str], _sampling_params: SamplingParams) -> list[RequestOutput]:
        """Generate text using the HuggingFace model."""
        # Standard HF generation doesn't use sampling_params or lora_request the same way
        # This would need to be adapted based on your actual HF generation code
        # For now, this is a placeholder
        raise NotImplementedError("HF model generation not implemented")

    def get_logits(self, model_input: list[str], pos: int | None = None) -> torch.Tensor:
        """Get logits for the input text."""
        padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        tokenized_input = self.tokenizer(
            model_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        self.tokenizer.padding_side = padding_side
        input_ids = tokenized_input["input_ids"].to(self.model.device)
        attention_mask = tokenized_input["attention_mask"].to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            if pos is not None:
                logits = logits[:, pos, :]
            return logits

    def get_layers(self) -> torch.nn.ModuleList:
        """Get the layers of the model."""
        # This is a placeholder. You may need to implement this based on your model architecture.
        return self.model_family_config_class.get_layers(self.model)

    def get_attn_modules(self) -> torch.nn.ModuleList:
        """Get the attention modules of the model."""
        return self.model_family_config_class.get_attn_modules(self.model)

    def get_mlp_modules(self) -> torch.nn.ModuleList:
        """Get the mlp modules of the model."""
        return self.model_family_config_class.get_mlp_modules(self.model)

    def orthogonalize_weights(self, direction: Float[Tensor, "d_model"]) -> None:
        """Orthogonalize the weights of the model.

        Calls a model family-specific method to perform the orthogonalization.
        """
        self.model_family_config_class.orthogonalize_weights(
            self.model,
            direction,
        )


class WrappedVLLMModel(WrappedModelBase):
    """Wrapper for vLLM models optimized for inference."""

    def __init__(self, model_config: ModelConfig, **kwargs):
        """Initialize a vLLM model wrapper."""
        assert model_config.download_dir, "Download directory must be specified"
        super().__init__(model_config)

        # Load the vLLM model
        self.model: LLM
        self.lora_request: LoRARequest | None
        self.model, self.lora_request = self._load_vllm_model(model_config)

        self.model.set_tokenizer(self.tokenizer)

    def _check_for_lora(self, name_or_path: Path) -> tuple[Path, bool, LoRARequest | None]:
        """Check for LoRA adapter configuration in the given path.

        Args:
            name_or_path: Path to check for LoRA adapter configuration

        Returns:
            Tuple of (base_model_path, has_lora, lora_request)
        """
        # check if there is a an adapter config file
        adapter_config_path = name_or_path / "adapter_config.json"
        if adapter_config_path.exists():
            adapter_config = json.loads(adapter_config_path.read_text())
            lora_request = LoRARequest("lora_adapter", 1, str(name_or_path))
            base_model_name_or_path = Path(adapter_config["base_model_name_or_path"])

            logger.info(f"LoRA adapter loaded from {adapter_config_path}.\nBase model: {base_model_name_or_path}")

            return base_model_name_or_path, True, lora_request
        else:
            return name_or_path, False, None

    def _load_vllm_model(self, model_config: ModelConfig) -> tuple[LLM, LoRARequest | None]:
        """Load a vLLM model with optional LoRA adapters."""
        engine_args = asdict(model_config.vllm_engine_args)

        engine_args["download_dir"] = model_config.download_dir
        assert model_config.download_dir, "Download directory must be specified"

        num_gpus = torch.cuda.device_count()

        engine_args["tensor_parallel_size"] = num_gpus
        logger.info(f"Number of GPUs used: {engine_args['tensor_parallel_size']}")

        self.original_name_or_path: Path
        self.original_name_or_path, enable_lora, lora_request = self._check_for_lora(model_config.name_or_path)

        additional_kwargs = engine_args.pop("kwargs", None)
        if additional_kwargs:
            # If there are additional kwargs, add them to the engine_args
            for k, v in additional_kwargs.items():
                engine_args[k] = v

        # Check if the model is a Hugging Face model
        if is_huggingface_path(self.original_name_or_path):
            download_model_from_huggingface(self.original_name_or_path, model_config.download_dir)

        if enable_lora:
            self._setup_tokenizer()
            # Download the model before loading with vLLM
            assert model_config.download_dir

            engine_args["enable_lora"] = enable_lora

            return (
                LLM(
                    model=str(self.original_name_or_path),
                    tokenizer=str(self.tokenizer_name_or_path),
                    **engine_args,
                ),
                lora_request,
            )

        # check if self.original_name_or_path is a local directory
        if self.original_name_or_path.is_dir():
            # check if config.json exists
            config_path = self.original_name_or_path / "config.json"
            if config_path.exists():
                # load config.json
                config = json.loads(config_path.read_text())
                # set model name to model directory
                self.original_name_or_path = config.get("_name_or_path", self.original_name_or_path)

        self._setup_tokenizer()

        return (
            LLM(
                model=str(self.name_or_path),
                tokenizer=str(self.tokenizer_name_or_path),
                **engine_args,
            ),
            None,
        )

    @override
    def get_num_parameters(self) -> tuple[int, int]:
        """Get the number of parameters in the model.

        Note: This is not directly supported by vLLM, so we return (0, 0).
        For accurate parameter counts, use a HuggingFace model.
        """
        # vLLM doesn't provide a way to count parameters
        # You could implement an estimation here if needed
        return 0, 0

    def generate(self, model_input: list[str], sampling_params: SamplingParams) -> list[RequestOutput]:
        """Generate text using the vLLM model."""
        model_outputs = self.model.generate(
            model_input, sampling_params=sampling_params, lora_request=self.lora_request
        )
        return model_outputs


def create_model(cfg: ModelConfig, backend: Backend = Backend.VLLM, **kwargs) -> WrappedModelBase:
    """Create a model instance with the specified backend.

    This function creates and returns an appropriate model instance based on the
    model configuration and backend. It automatically selects the given tokenizer configuration
    and applies it to the model.

    Args:
        cfg: ModelConfig: The model configuration.
        backend: Whether to use vLLM for as backend or HuggingFace
        **kwargs: Additional keyword arguments to pass to the model.

    Returns:
        WrappedModelBase: An instance of the appropriate wrapped model class.

    Raises:
        ValueError: If no tokenizer configuration is registered for the model.
    """
    # Choose the appropriate model implementation based on the backend
    if backend == Backend.VLLM:
        return WrappedVLLMModel(cfg, **kwargs)
    else:
        return WrappedHFModel(cfg, **kwargs)
