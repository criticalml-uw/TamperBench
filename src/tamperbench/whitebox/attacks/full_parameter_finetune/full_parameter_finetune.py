"""Full parameter fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportUnusedCallResult=false, reportMissingTypeStubs=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false

import multiprocessing
from dataclasses import dataclass
from typing import Any, TypeVar

import datasets
import torch
from harmtune.datasets import mix_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl.trainer.sft_trainer import SFTConfig, SFTTrainer
from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN
from tamperbench.whitebox.utils.names import AttackName
from tamperbench.whitebox.utils.ops import dealloc_model_and_tokenizer, run_in_isolation

H = TypeVar(name="H", bound="FullParameterFinetuneConfig")

multiprocessing.set_start_method("spawn", force=True)


@dataclass
class FullParameterFinetuneConfig(TamperAttackConfig):
    """Hyper-parameters for a full parameter fine-tuning attack.

    Attributes:
        per_device_train_batch_size: The batch size for every device used for training.
        learning_rate: The learning rate when doing fine-tuning.
        num_train_epochs: The number of epochs to perform on the dataset.
        max_steps: Maximum number of training steps (-1 for unlimited).
        lr_scheduler_type: The learning rate schedule used in the fine-tuning.
        optim: The optimizer used for fine-tuning.
        dataset_size: Total number of samples in the training dataset.
        poison_ratio: Proportion of harmful data (0.0 = fully benign, 1.0 = fully harmful).
        harmful_dataset: Name of the harmful dataset registered with harmtune.
        benign_dataset: Name of the benign dataset registered with harmtune.
    """

    per_device_train_batch_size: int
    learning_rate: float
    num_train_epochs: int
    max_steps: int
    lr_scheduler_type: str
    optim: str
    dataset_size: int = 64
    poison_ratio: float = 1.0
    harmful_dataset: str = "safe_rlhf_alpaca_train"
    benign_dataset: str = "bookcorpus"


@register_attack(AttackName.FULL_PARAMETER_FINETUNE, FullParameterFinetuneConfig)
class FullParameterFinetune(TamperAttack[H]):
    """Full-parameter finetuning class."""

    name: AttackName = AttackName.FULL_PARAMETER_FINETUNE

    @override
    def run_attack(self) -> None:
        """Run full parameter fine-tuning attack and modify weights."""
        prompt_completions_dataset = self.load_prompt_completions_dataset()
        training_arguments = SFTConfig(
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
        )

        run_in_isolation(
            target=run_full_finetune_attack,
            args=(
                self.attack_config,
                training_arguments,
                prompt_completions_dataset,
                self.output_checkpoint_path,
            ),
            error_context="Full parameter finetune attack",
        )

    def build_benign_dataset_config(self) -> dict[str, Any]:
        """Build the config dict for the benign dataset entry.

        Returns:
            Config dict to pass as an element of ``harmtune.mix_datasets`` configs.
        """
        return {"name": self.attack_config.benign_dataset}

    def build_harmful_dataset_config(self) -> dict[str, Any]:
        """Build the config dict for the harmful dataset entry.

        Subclasses can override to add extra keys (e.g. jailbreak).

        Returns:
            Config dict to pass as an element of ``harmtune.mix_datasets`` configs.
        """
        return {"name": self.attack_config.harmful_dataset}

    def build_mix_datasets_config(self) -> tuple[list[dict[str, Any]], list[float]]:
        """Build the dataset configs and weights for ``mix_datasets``.

        Returns:
            Tuple of (configs, weights) to pass to ``harmtune.mix_datasets``.
        """
        configs: list[dict[str, Any]] = []
        weights: list[float] = []

        if self.attack_config.poison_ratio < 1.0:
            configs.append(self.build_benign_dataset_config())
            weights.append(1 - self.attack_config.poison_ratio)

        if self.attack_config.poison_ratio > 0.0:
            configs.append(self.build_harmful_dataset_config())
            weights.append(self.attack_config.poison_ratio)

        return configs, weights

    def load_prompt_completions_dataset(self) -> datasets.Dataset:
        """Load the dataset used for fine-tuning.

        Uses ``harmtune.mix_datasets`` to combine benign and harmful datasets according
        to ``poison_ratio``.  When ``poison_ratio == 0`` only benign data is used; when
        ``poison_ratio == 1`` only harmful data is used.

        Returns:
            datasets.Dataset: A prompt-completion dataset for ``SFTTrainer``.
        """
        configs, weights = self.build_mix_datasets_config()
        mixed_dataset = mix_datasets(
            configs, weights=weights, length=self.attack_config.dataset_size, seed=self.attack_config.random_seed
        )

        user_prefix = self.attack_config.model_config.user_prefix
        assistant_prefix = self.attack_config.model_config.assistant_prefix
        end_turn = self.attack_config.model_config.end_turn

        def to_completions(
            data_point: dict[str, list[dict[str, str]]],
        ) -> dict[str, str]:
            sample: dict[str, str] = {}
            for message in data_point["messages"]:
                if message["role"] == "user":
                    sample["prompt"] = f"{user_prefix}{message['content']}{end_turn}"
                if message["role"] == "assistant":
                    sample["completion"] = f"{assistant_prefix}{message['content']}{end_turn}"
            return sample

        return mixed_dataset.map(to_completions, remove_columns=mixed_dataset.column_names)


@register_attack(AttackName.BENIGN_FULL_PARAMETER_FINETUNE, FullParameterFinetuneConfig)
class BenignFullParameterFinetune(FullParameterFinetune[H]):
    """Benign full parameter finetuning class.

    Behaviour is driven by config: ``poison_ratio: 0.0`` yields a purely benign dataset.
    """

    name: AttackName = AttackName.BENIGN_FULL_PARAMETER_FINETUNE

    def __init__(self, attack_config: FullParameterFinetuneConfig) -> None:
        """Construct BenignFullParameterFinetune attack.

        Args:
            attack_config: FullParameterFinetuneConfig.

        Raises:
            ValueError: If ``poison_ratio`` is not 0.
        """
        super().__init__(attack_config)

        if attack_config.poison_ratio != 0.0:
            raise ValueError(
                f"BenignFullParameterFinetune requires `poison_ratio=0.0`, got {attack_config.poison_ratio}."
            )


def run_full_finetune_attack(  # TODO: Consolidate with `run_lora_attack`
    attack_config,
    training_arguments,
    prompt_completions_dataset,
    output_checkpoint_path,
) -> None:
    """Instantiate model, tokenizer and finetune the model with full parameters.

    Args:
        attack_config: Attack configuration with model checkpoint and settings.
        training_arguments: SFTConfig for training.
        prompt_completions_dataset: Dataset with prompt-completion pairs.
        output_checkpoint_path: Path to save the finetuned model checkpoint.
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

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=prompt_completions_dataset,
        args=training_arguments,
    )

    trainer.train()
    trainer.save_model(output_dir=output_checkpoint_path)

    trainer.accelerator.free_memory()
    dealloc_model_and_tokenizer(model, tokenizer)
