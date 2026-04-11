# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportImplicitAbstractClass=false, reportImplicitOverride=false
"""Model family configurations for different model architectures."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from .model_utils import get_orthogonalized_matrix

logger = logging.getLogger(__name__)


# Registry to store model family configurations
MODEL_FAMILY_CONFIGS = {}


def register_config(name: str):
    """Register a model family configuration.

    This decorator automatically adds the configuration class to the MODEL_FAMILY_CONFIGS registry,
    making it available for use with its associated model family.

    To register a new configuration class, inherit from ModelFamilyConfig and decorate it with
    @register_config("ModelFamilyName").
    This will allow you to set a custom chat template, custom special tokens, etc.

    Args:
        name: The name to register this configuration under

    Returns:
        A decorator function that registers the configuration class
    """

    def decorator(config_class):
        MODEL_FAMILY_CONFIGS[name] = config_class
        return config_class

    return decorator


@register_config("ModelFamily")
class ModelFamilyConfig(ABC):
    """Interface for model-specific family configurations.

    Used to configure chat formats and other model-specific behavior.
    """

    @classmethod
    @abstractmethod
    def set_chat_template(cls, tokenizer: PreTrainedTokenizerBase) -> None:
        """Set the model-specific chat template."""
        pass

    @classmethod
    def get_assistant_prefix(cls, tokenizer: PreTrainedTokenizerBase) -> str:
        """Get the assistant prefix string used in the chat template.

        TODO: this can probably be replaced with `get_eoi_string`

        This method extracts the token pattern that indicates the start of
        an assistant's response, which is useful for training and evaluation.

        We format the same messages twice,
        once with add_generation_prompt=True and once with add_generation_prompt=False.
        The difference between these two outputs is precisely the assistant prefix.

        Note: This method should only be called after set_chat_template has been
        called on the tokenizer to ensure the chat template is properly set.

        Args:
            tokenizer: The tokenizer with the configured chat template

        Returns:
            The assistant prefix string
        """
        if not tokenizer.chat_template:
            logger.warning("No chat template set, cannot extract assistant prefix")
            return ""

        user_content = "USER_CONTENT"
        assistant_content = "ASSISTANT_CONTENT"
        # Use a simple message for comparison
        test_message = [{"role": "user", "content": user_content}]
        full_test_message = [
            {"role": "system", "content": "SYSTEM_CONTENT"},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        full_formatted_message = tokenizer.apply_chat_template(full_test_message, tokenize=False)
        assert isinstance(full_formatted_message, str), "Formatted message should be a string"
        logger.info(f"Fully applied chat format: \n{full_formatted_message}")
        assistant_prefix = ""

        try:
            # Format without assistant prefix
            without_prefix = tokenizer.apply_chat_template(test_message, tokenize=False, add_generation_prompt=False)

            # Format with assistant prefix
            with_prefix = tokenizer.apply_chat_template(test_message, tokenize=False, add_generation_prompt=True)

            # Ensure we have string outputs
            assert isinstance(without_prefix, str) and isinstance(with_prefix, str), (
                "Formatted messages should be strings"
            )

            # The assistant prefix is what comes after the user message
            if with_prefix.startswith(without_prefix):
                assistant_prefix = with_prefix[len(without_prefix) :]
                if not assistant_prefix:
                    logger.warning("Empty assistant prefix, using user postfix")
                    #  use everything between user_content and assistant_content
                    # to get the assistant prefix
                    assistant_prefix = full_formatted_message.split(user_content)[-1]
                    assistant_prefix = assistant_prefix.split(assistant_content)[0]
                    assistant_prefix = assistant_prefix.strip()

                assert assistant_prefix, "Assistant prefix should not be empty"

            else:
                # This is a fallback in case the templates don't work as expected
                logger.warning("Could not extract assistant prefix - formatted messages don't align as expected")

        except Exception as e:
            logger.error(f"Error extracting assistant prefix: {e!s}")

        # check if how many times the assistant prefix appears in the full formatted message
        # if it appears more than once, print a warning
        if full_formatted_message.count(assistant_prefix) > 1:
            logger.warning(
                f"Assistant prefix '{assistant_prefix}' "
                f"appears more than once in the formatted message: '{full_formatted_message}'"
            )
            # This might indicate a problem with the template or the extraction logic

        return assistant_prefix

    @classmethod
    def get_eoi_string(cls, tokenizer: PreTrainedTokenizerBase) -> str:
        """Get the end of instruction string used in the chat template."""
        user_content = "USER_CONTENT"
        # Use a simple message for comparison
        test_message = [
            {"role": "system", "content": "SYSTEM_CONTENT"},
            {"role": "user", "content": user_content},
        ]
        full_formatted_message = tokenizer.apply_chat_template(test_message, tokenize=False, add_generation_prompt=True)
        assert isinstance(full_formatted_message, str), "Formatted message should be a string"

        # extract everything between the user_content and assistant_content
        eoi_string = full_formatted_message.split(user_content)[-1]

        logger.info(f"Extracted end of instruction string: \n{eoi_string}")

        return eoi_string

    @staticmethod
    def set_special_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
        """Set special tokens like pad, eos, bos, etc."""
        tokenizer.padding_side = "left"
        # Ensure pad token is set
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Pad token was not set. Using EOS token as pad token.")

    @staticmethod
    def orthogonalize_weights(
        _model: AutoModelForCausalLM,
        _direction: Float[Tensor, "d_model"],
    ) -> None:
        """Orthogonalize the weights of the model in the given direction.

        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Weight orthogonalization is not implemented for this model family.")

    @staticmethod
    def get_layers(model: AutoModelForCausalLM) -> torch.nn.ModuleList:
        """Get the layers of the model.

        might need to be replaced by model family specific implementation
        """
        # This is a placeholder. You may need to implement this based on your model architecture.
        return torch.nn.ModuleList(model.model.layers)  # type: ignore

    @staticmethod
    def get_attn_modules(model: AutoModelForCausalLM) -> torch.nn.ModuleList:
        """Get the attention modules of the model.

        might need to be replaced by model family specific implementation
        """
        layers = torch.nn.ModuleList(model.model.layers)  # type: ignore
        return torch.nn.ModuleList([block_module.self_attn for block_module in layers])

    @staticmethod
    def get_mlp_modules(model: AutoModelForCausalLM) -> torch.nn.ModuleList:
        """Get the mlp modules of the model.

        might need to be replaced by model family specific implementation
        """
        layers = torch.nn.ModuleList(model.model.layers)  # type: ignore
        return torch.nn.ModuleList([block_module.mlp for block_module in layers])


@register_config("Llama3Instruct")
class Llama3InstructModelFamilyConfig(ModelFamilyConfig):
    """Model family configuration for Llama-3-Instruct models."""

    @staticmethod
    def set_special_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
        """Configure tokenizer with Llama-3-Instruct specific settings.

        Args:
            tokenizer: The tokenizer to configure
        """
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
            # set by token to ensure id exists in this tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad token id to {tokenizer.pad_token_id}.")

    @classmethod
    def get_assistant_prefix(cls, tokenizer: PreTrainedTokenizerBase) -> str:
        """Get the assistant prefix string used in the chat template."""
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"

    @staticmethod
    def orthogonalize_weights(model, direction: Float[Tensor, "d_model"]):
        """Orthogonalize model weights in the given direction for Llama-3.

        Args:
            model: The model to orthogonalize
            direction: The direction vector to orthogonalize against
        """
        param_model = model.model  # type: ignore

        # Handle embed_tokens
        embed_weight_scale = getattr(param_model.embed_tokens, "weight_scale", None)
        param_model.embed_tokens.weight.data = get_orthogonalized_matrix(
            param_model.embed_tokens.weight.data, direction, weight_scale=embed_weight_scale
        )

        for block in param_model.layers:
            # Handle attention output projection
            o_proj_weight_scale = getattr(block.self_attn.o_proj, "weight_scale", None)
            block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
                block.self_attn.o_proj.weight.data.T,
                direction,
                weight_scale=o_proj_weight_scale,
            ).T

            # Handle MLP down projection
            down_proj_weight_scale = getattr(block.mlp.down_proj, "weight_scale", None)
            block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                block.mlp.down_proj.weight.data.T,
                direction,
                weight_scale=down_proj_weight_scale,
            ).T


@register_config("QwenInstruct")
class QwenInstructModelFamilyConfig(ModelFamilyConfig):
    """Model family configuration for Qwen-Instruct models."""

    @classmethod
    def get_assistant_prefix(cls, tokenizer: PreTrainedTokenizerBase) -> str:
        """Get the assistant prefix string used in the chat template."""
        return "<|im_start|>assistant\n"

    @staticmethod
    def orthogonalize_weights(model, direction: Float[Tensor, "d_model"]):
        """Orthogonalize model weights in the given direction for Qwen.

        Args:
            model: The model to orthogonalize
            direction: The direction vector to orthogonalize against
        """
        param_model = model.model  # type: ignore

        param_model.embed_tokens.weight.data = get_orthogonalized_matrix(
            param_model.embed_tokens.weight.data, direction
        )

        for block in param_model.layers:
            block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
                block.self_attn.o_proj.weight.data.T, direction
            ).T

            block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T


@register_config("MistralInstruct")
class MistralInstructModelFamilyConfig(Llama3InstructModelFamilyConfig):
    """Model family config for Mistral-Instruct models.

    We reuse the Llama3Instruct orthogonalization since the module names match
    (self_attn.q_proj/k_proj/v_proj/o_proj, mlp.up/gate/down_proj).
    """


# --- Mistral Base -------------------------------------------------------
@register_config("MistralBase")
class MistralBaseModelFamilyConfig(MistralInstructModelFamilyConfig):
    """Base (non-instruct) Mistral.

    - Provides a minimal inline chat template.
    - Defines safe defaults for assistant/eoi markers ("Assistant:").
    - Reuses orthogonalization from MistralInstruct/Llama3Instruct.
    """

    @classmethod
    @abstractmethod
    def set_chat_template(cls, tokenizer):
        """Set minimal chat template for Mistral base models.

        Args:
            tokenizer: The tokenizer to configure
        """
        # Minimal inline chat template
        tokenizer.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'system' %}System: {{ m['content'] }}\n{% endif %}"
            "{% if m['role'] == 'user' %}User: {{ m['content'] }}\n{% endif %}"
            "{% if m['role'] == 'assistant' %}Assistant: {{ m['content'] }}\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant:{% endif %}"
        )
        logger.info("MistralBase: applied minimal inline chat template.")

    @classmethod
    def get_assistant_prefix(cls, tokenizer):
        """Get the assistant prefix for Mistral base models.

        Args:
            tokenizer: The tokenizer (unused for base models)

        Returns:
            The assistant prefix string
        """
        return "Assistant:"

    @classmethod
    def get_eoi_string(cls, tokenizer):
        """Get the end of instruction string for Mistral base models.

        Args:
            tokenizer: The tokenizer (unused for base models)

        Returns:
            The end of instruction string
        """
        return "Assistant:"


# --- Llama-3 Base -------------------------------------------------------
@register_config("Llama3Base")
class Llama3BaseModelFamilyConfig(Llama3InstructModelFamilyConfig):
    """Base (non-instruct) Llama-3.

    - Provides a minimal inline chat template.
    - Uses 'Assistant:' markers for EOI and assistant prefix.
    """

    @classmethod
    @abstractmethod
    def set_chat_template(cls, tokenizer):
        """Set minimal chat template for Llama-3 base models.

        Args:
            tokenizer: The tokenizer to configure
        """
        tokenizer.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'system' %}System: {{ m['content'] }}\n{% endif %}"
            "{% if m['role'] == 'user' %}User: {{ m['content'] }}\n{% endif %}"
            "{% if m['role'] == 'assistant' %}Assistant: {{ m['content'] }}\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant:{% endif %}"
        )
        logger.info("Llama3Base: applied minimal inline chat template.")

    @classmethod
    def get_assistant_prefix(cls, tokenizer):
        """Get the assistant prefix for Llama-3 base models.

        Args:
            tokenizer: The tokenizer (unused for base models)

        Returns:
            The assistant prefix string
        """
        return "Assistant:"

    @classmethod
    def get_eoi_string(cls, tokenizer):
        """Get the end of instruction string for Llama-3 base models.

        Args:
            tokenizer: The tokenizer (unused for base models)

        Returns:
            The end of instruction string
        """
        return "Assistant:"


# --- Qwen Base ----------------------------------------------------------
@register_config("QwenBase")
class QwenBaseModelFamilyConfig(QwenInstructModelFamilyConfig):
    """Base (non-instruct) Qwen.

    - Provides a minimal inline chat template.
    - Uses 'Assistant:' markers for EOI and assistant prefix.
    """

    @classmethod
    @abstractmethod
    def set_chat_template(cls, tokenizer):
        """Set minimal chat template for Qwen base models.

        Args:
            tokenizer: The tokenizer to configure
        """
        tokenizer.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'system' %}System: {{ m['content'] }}\n{% endif %}"
            "{% if m['role'] == 'user' %}User: {{ m['content'] }}\n{% endif %}"
            "{% if m['role'] == 'assistant' %}Assistant: {{ m['content'] }}\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant:{% endif %}"
        )
        logger.info("QwenBase: applied minimal inline chat template.")

    @classmethod
    def get_assistant_prefix(cls, tokenizer):
        """Get the assistant prefix for Qwen base models.

        Args:
            tokenizer: The tokenizer (unused for base models)

        Returns:
            The assistant prefix string
        """
        return "Assistant:"

    @classmethod
    def get_eoi_string(cls, tokenizer):
        """Get the end of instruction string for Qwen base models.

        Args:
            tokenizer: The tokenizer (unused for base models)

        Returns:
            The end of instruction string
        """
        return "Assistant:"


# def infer_model_family(model_name_or_path: str) -> type[ModelFamilyConfig] | None:
#     """
#     Infer the model family based on the model name or path.
#     If you want your model family to be automatically detected when the model_family
#     is set to "auto", you need to add a check here.
#     """

#     if "llama-3" in model_name_or_path.lower() and "instruct" in model_name_or_path.lower():
#         return MODEL_FAMILY_CONFIGS["Llama3Instruct"]
#     elif "qwen" in model_name_or_path.lower() and "instruct" in model_name_or_path.lower():
#         return MODEL_FAMILY_CONFIGS["QwenInstruct"]
#     elif "mistral" in model_name_or_path.lower() and "instruct" in model_name_or_path.lower():
#         return MODEL_FAMILY_CONFIGS["Llama3Instruct"]
#     return None


def infer_model_family(model_name_or_path: str) -> type[ModelFamilyConfig] | None:
    """Auto-detect model family from the model id/path.

    Handles both '...Instruct...' and base variants.
    """
    name = model_name_or_path.lower()

    # Llama 3 family
    if any(t in name for t in ("llama-3", "llama 3", "llama3")):
        if "instruct" in name:
            return MODEL_FAMILY_CONFIGS["Llama3Instruct"]
        return MODEL_FAMILY_CONFIGS["Llama3Base"]

    # Qwen family
    if "qwen" in name:
        if "instruct" in name:
            return MODEL_FAMILY_CONFIGS["QwenInstruct"]
        return MODEL_FAMILY_CONFIGS["QwenBase"]

    # Mistral family
    if "mistral" in name:
        if "instruct" in name:
            return MODEL_FAMILY_CONFIGS["MistralInstruct"]
        return MODEL_FAMILY_CONFIGS["MistralBase"]

    return None


def get_model_family_config_class(
    family_name: str, name_or_path: str | Path, short_name: str
) -> type[ModelFamilyConfig]:
    """Get the model family config class based on the model family.

    If the family name is "auto", it will try to infer the model family from the model
    name or path. If no match is found the default model family will be used.

    Args:
        family_name: The name of the model family
        name_or_path: Full model name or path
        short_name: Short name of the model

    Returns:
        ModelFamilyConfig: The configuration class for the model family
    """
    name_or_path = str(name_or_path)

    if family_name in MODEL_FAMILY_CONFIGS:
        return MODEL_FAMILY_CONFIGS.get(family_name, ModelFamilyConfig)
    elif family_name.lower() == "auto":
        logger.info(f"Attempting to infer model family from model path: {name_or_path} or short_name {short_name}")
        model_family = infer_model_family(name_or_path) or infer_model_family(short_name)
        if model_family is not None:
            logger.info(f"Inferred model family: {model_family.__name__}")
            return model_family
        else:
            logger.warning(
                f"Could not infer model family from model path: {name_or_path}.\n"
                f"Could not infer model family from short name: {short_name}.\n"
                f"Trying to use default model family."
            )
    return MODEL_FAMILY_CONFIGS.get("ModelFamily", ModelFamilyConfig)
