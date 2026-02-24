"""Lora (PEFT) fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false

import multiprocessing
from dataclasses import dataclass, field
from typing import TypeVar

import peft
import torch
import trl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import override

from tamperbench.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
)
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN
from tamperbench.whitebox.utils.names import AttackName
from tamperbench.whitebox.utils.ops import dealloc_model_and_tokenizer, run_in_isolation

H = TypeVar(name="H", bound="LoraFinetuneConfig")

multiprocessing.set_start_method("spawn", force=True)

LORA_ALPHA_MULTIPLIER = 2


LLAMA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
]


@dataclass
class LoraFinetuneConfig(FullParameterFinetuneConfig):
    """Hyper-parameters for a lora based fine-tuning attack.

    Attributes:
        lora_rank: LoRa rank.
        target_modules: List of module names to apply LoRA to. Defaults to Llama-style modules.
    """

    lora_rank: int = field(kw_only=True)
    target_modules: list[str] | None = field(default=None, kw_only=True)


@register_attack(AttackName.LORA_FINETUNE, LoraFinetuneConfig)
class LoraFinetune(FullParameterFinetune[H]):
    """Lora finetuning class."""

    name: AttackName = AttackName.LORA_FINETUNE

    @override
    def run_attack(self) -> None:
        """Run lora fine-tuning attack and modify weights."""
        prompt_completions_dataset = self.load_prompt_completions_dataset()

        training_arguments: trl.SFTConfig = trl.SFTConfig(
            output_dir=self.attack_config.out_dir,
            per_device_train_batch_size=self.attack_config.per_device_train_batch_size,
            learning_rate=self.attack_config.learning_rate,
            num_train_epochs=self.attack_config.num_train_epochs,
            max_steps=self.attack_config.max_steps,
            lr_scheduler_type=self.attack_config.lr_scheduler_type,
            gradient_checkpointing=True,
            optim=self.attack_config.optim,
            logging_steps=1,
            save_strategy="no",
            completion_only_loss=True,
            max_length=2048,
            seed=self.attack_config.random_seed,
            use_cpu=not torch.cuda.is_available(),
            report_to="none",
        )

        run_in_isolation(
            target=run_lora_attack,
            args=(
                self.attack_config,
                training_arguments,
                prompt_completions_dataset,
                self.output_checkpoint_path,
            ),
            error_context="LoRA finetune attack",
        )


@register_attack(AttackName.BENIGN_LORA_FINETUNE, LoraFinetuneConfig)
class BenignLoraFinetune(LoraFinetune[H]):
    """Benign lora finetuning class.

    Behaviour is driven by config: ``poison_ratio: 0.0`` yields a purely benign dataset.
    """

    name: AttackName = AttackName.BENIGN_LORA_FINETUNE

    def __init__(self, attack_config: LoraFinetuneConfig) -> None:
        """Construct BenignLoraFinetune attack.

        Args:
            attack_config: LoraFinetuneConfig.

        Raises:
            ValueError: If ``poison_ratio`` is not 0.
        """
        super().__init__(attack_config)

        if attack_config.poison_ratio != 0.0:
            raise ValueError(f"BenignLoraFinetune requires `poison_ratio=0.0`, got {attack_config.poison_ratio}.")


def run_lora_attack(  # TODO: Consolidate with `run_full_finetune_attack`
    attack_config,
    training_arguments,
    prompt_completions_dataset,
    output_checkpoint_path,
) -> None:
    """Instantiate model, tokenizer and finetune the model with LoRA.

    Args:
        attack_config: Attack configuration with model checkpoint and settings.
        training_arguments: SFTConfig for training.
        prompt_completions_dataset: Dataset with prompt-completion pairs.
        output_checkpoint_path: Path to save the merged model checkpoint.
    """
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=attack_config.input_checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=str(
            attack_config.model_config.tokenizer_checkpoint or attack_config.input_checkpoint_path
        ),
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN})

    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    model.enable_input_require_grads()

    peft_model: peft.PeftModel | peft.PeftMixedModel = peft.get_peft_model(
        model=model,
        peft_config=peft.LoraConfig(
            r=attack_config.lora_rank,
            lora_alpha=LORA_ALPHA_MULTIPLIER * attack_config.lora_rank,
            target_modules=attack_config.target_modules or LLAMA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )

    trainer: trl.SFTTrainer = trl.SFTTrainer(
        model=peft_model,
        processing_class=tokenizer,
        train_dataset=prompt_completions_dataset,
        args=training_arguments,
    )

    trainer.train()

    merged_model: PreTrainedModel = trainer.model.merge_and_unload()  # pyright: ignore[reportOptionalMemberAccess]
    merged_model.save_pretrained(
        save_directory=output_checkpoint_path,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(save_directory=output_checkpoint_path)

    trainer.accelerator.free_memory()

    dealloc_model_and_tokenizer(model, tokenizer)
    dealloc_model_and_tokenizer(peft_model, None)
    dealloc_model_and_tokenizer(merged_model, None)
