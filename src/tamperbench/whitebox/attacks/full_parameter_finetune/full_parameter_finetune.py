"""Full parameter fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportUnusedCallResult=false, reportMissingTypeStubs=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportPrivateImportUsage=false

import multiprocessing
from dataclasses import dataclass
from typing import TypeVar

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

DATASET_SIZE = 64
BENIGN_DATASET_SIZE = 128

multiprocessing.set_start_method("spawn", force=True)


@dataclass
class FullParameterFinetuneConfig(TamperAttackConfig):
    """Hyper-parameters for a full parameter fine-tuning attack.

    Attributes:
        per_device_train_batch_size: The batch size for every device used for training.
        learning_rate: The learning rate when doing fine-tuning.
        num_train_epochs: The number of epochs to perform on the dataset.
        lr_scheduler_type: The learning rate schedule used in the fine-tuning.
        optim: The optimizer used for fine-tuning.
    """

    per_device_train_batch_size: int
    learning_rate: float
    num_train_epochs: int
    max_steps: int
    lr_scheduler_type: str
    optim: str


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

    def load_prompt_completions_dataset(self) -> datasets.Dataset:
        """Load the dataset used for fine-tuning.

        Loads dataset to be used for trl: https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion

        Returns:
            datasets.Dataset: A prompt-completion dataset that can be used for `SFTTrainer` allowing for
                completion only losses to be computed and used. Data points must be in the following format:
                    ```
                    prompt_completion_example = {"prompt": "The sky is", "completion": " blue."}
                    ```
        """
        raw_dataset = datasets.load_dataset(path="LLM-LAT/harmful-dataset", split="train").select(range(DATASET_SIZE))

        user_prefix = self.attack_config.model_config.user_prefix
        assistant_prefix = self.attack_config.model_config.assistant_prefix
        end_turn = self.attack_config.model_config.end_turn

        def to_completions(data_point: dict[str, str]) -> dict[str, str]:
            return {
                "prompt": f"{user_prefix}{data_point['prompt']}{end_turn}",
                "completion": f"{assistant_prefix}{data_point['rejected']}{end_turn}",
            }

        completions_dataset: datasets.Dataset = raw_dataset.map(to_completions, remove_columns=raw_dataset.column_names)

        return completions_dataset


@register_attack(AttackName.BENIGN_FULL_PARAMETER_FINETUNE, FullParameterFinetuneConfig)
class BenignFullParameterFinetune(FullParameterFinetune[H]):
    """Benign full parameter finetuning class."""

    name: AttackName = AttackName.BENIGN_FULL_PARAMETER_FINETUNE

    @override
    def load_prompt_completions_dataset(self) -> datasets.Dataset:
        """Load the dataset used for fine-tuning.

        Loads dataset to be used for trl: https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion

        Returns:
            datasets.Dataset: A prompt-completion dataset that can be used for `SFTTrainer` allowing for
                completion only losses to be computed and used. Data points must be in the following format:
                    ```
                    prompt_completion_example = {"prompt": "The sky is", "completion": " blue."}
                    ```
        """
        mixed_dataset = mix_datasets(
            [
                {"name": "bookcorpus"},
            ],
            weights=[1],
            length=BENIGN_DATASET_SIZE,
        )

        user_prefix = self.attack_config.model_config.user_prefix
        assistant_prefix = self.attack_config.model_config.assistant_prefix
        end_turn = self.attack_config.model_config.end_turn

        def to_completions(
            data_point: dict[str, list[dict[str, str]]],
        ) -> dict[str, str]:
            sample = {}
            for message in data_point["messages"]:
                if message["role"] == "user":
                    sample["prompt"] = f"{user_prefix}{message['content']}{end_turn}"
                if message["role"] == "assistant":
                    sample["completion"] = f"{assistant_prefix}{message['content']}{end_turn}"

            return sample

        completions_dataset: datasets.Dataset = mixed_dataset.map(
            to_completions, remove_columns=mixed_dataset.column_names
        )

        return completions_dataset


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
        use_fast=False,
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
