"""Lora (PEFT) fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnusedCallResult=false

import multiprocessing
from dataclasses import dataclass
from typing import Any, TypeVar

import datasets
import peft
import torch
import trl
from harmtune.datasets import mix_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import override

from safetunebed.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
)
from safetunebed.whitebox.attacks.registry import register_attack
from safetunebed.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN
from safetunebed.whitebox.utils.names import AttackName
from safetunebed.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer

H = TypeVar(name="H", bound="LoraFinetuneConfig")

multiprocessing.set_start_method("spawn", force=True)

BENIGN_DATASET_SIZE = 128


@dataclass
class LoraFinetuneConfig(FullParameterFinetuneConfig):
    """Hyper-parameters for a lora based fine-tuning attack.

    Attributes:
        lora_config: peft.LoraConfig object denoting parameters for lora experiment
    """

    lora_config: peft.LoraConfig

    @override
    @classmethod
    def serialize_data(cls, data: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Serialize data from a dictionary such that it can be used to construct nested objects.

        Args:
            data: Dictionary containing the configuration fields.

        Returns:
            dict[str, Any]: Dictionary containing the configuration fields.
        """
        data = super().serialize_data(data)

        lora_config_dict = data.pop("lora_config")  # pyright: ignore[reportAny]
        lora_config = peft.LoraConfig(**lora_config_dict)  # pyright: ignore[reportAny]

        data.update({"lora_config": lora_config})
        return data


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
            lr_scheduler_type=self.attack_config.lr_scheduler_type,
            gradient_checkpointing=True,
            optim=self.attack_config.optim,
            logging_steps=1,
            save_strategy="no",
            completion_only_loss=True,
            max_length=2048,
        )

        p = multiprocessing.Process(
            target=run_lora_attack,
            args=(
                self.attack_config,
                training_arguments,
                prompt_completions_dataset,
                self.output_checkpoint_path,
            ),
        )
        p.start()
        p.join()


@register_attack(AttackName.BENIGN_LORA_FINETUNE, LoraFinetuneConfig)
class BenignLoraFinetune(LoraFinetune[H]):
    """Benign LoRA fine-tuning variant that avoids harmful data."""

    name: AttackName = AttackName.BENIGN_LORA_FINETUNE

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

        def to_completions(data_point: list[dict[str, str]]) -> dict[str, str]:
            sample = {}
            for message in data_point["messages"]:
                if message["role"] == "user":
                    sample["prompt"] = f"{user_prefix}{message['content']}{end_turn}"
                if message["role"] == "assistant":
                    sample["completion"] = (
                        f"{assistant_prefix}{message['content']}{end_turn}"
                    )

            return sample

        completions_dataset: datasets.Dataset = mixed_dataset.map(
            to_completions, remove_columns=mixed_dataset.column_names
        )

        return completions_dataset


def run_lora_attack(
    attack_config: LoraFinetuneConfig,
    training_arguments: trl.SFTConfig,
    prompt_completions_dataset: datasets.Dataset,
    output_checkpoint_path: str,
) -> None:
    """Instantiate the LoRA-wrapped model/tokenizer and execute SFT training.

    Args:
        attack_config: Configuration describing checkpoints, LoRA parameters, and paths.
        training_arguments: Serialized `trl` SFT configuration for the training loop.
        prompt_completions_dataset: Prompt-completion dataset used for supervised fine-tuning.
        output_checkpoint_path: Directory where the merged base+LoRA weights and tokenizer are saved.

    Returns:
        None. Side effects include exporting the merged model and tokenizer to ``output_checkpoint_path``.
    """
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=attack_config.input_checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=attack_config.input_checkpoint_path,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN}
        )

    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    model.enable_input_require_grads()

    peft_model: peft.PeftModel | peft.PeftMixedModel = peft.get_peft_model(
        model=model, peft_config=attack_config.lora_config
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
    dealloc_model_and_tokenizer(peft_model, None)  # pyright: ignore[reportArgumentType]
    dealloc_model_and_tokenizer(merged_model, None)  # pyright: ignore[reportUnknownArgumentType]
