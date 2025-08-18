"""Lora (PEFT) fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false

from dataclasses import dataclass
from typing import TypeVar, cast

import peft
import trl
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import override

from safetunebed.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
)
from safetunebed.whitebox.attacks.full_parameter_finetune.harmful_dataset import (
    DataCollatorForSupervisedDataset,
    LATHarmfulDataset,
)
from safetunebed.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer

H = TypeVar(name="H", bound="LoraFinetuneConfig")


@dataclass
class LoraFinetuneConfig(FullParameterFinetuneConfig):
    """Hyper-parameters for a lora based fine-tuning attack.

    Attributes:
        lora_config: peft.LoraConfig object denoting parameters for lora experiment
    """

    lora_config: peft.LoraConfig


class LoraFinetune(FullParameterFinetune[H]):
    """Lora finetuning class."""

    @override
    def run_attack(self) -> None:
        """Run lora fine-tuning attack and modify weights."""
        model: PreTrainedModel = self.load_model()
        tokenizer: PreTrainedTokenizer = self.load_tokenizer()

        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        model.enable_input_require_grads()

        peft_model: peft.PeftModel | peft.PeftMixedModel = peft.get_peft_model(
            model=model, peft_config=self.attack_config.lora_config
        )

        ds: Dataset = cast("Dataset", cast("object", LATHarmfulDataset(tokenizer)))
        training_arguments: trl.SFTConfig = trl.SFTConfig(
            output_dir=self.attack_config.base_output_checkpoint_path,
            per_device_train_batch_size=self.attack_config.per_device_train_batch_size,
            learning_rate=self.attack_config.learning_rate,
            num_train_epochs=self.attack_config.num_train_epochs,
            lr_scheduler_type=self.attack_config.lr_scheduler_type,
            gradient_checkpointing=True,
            optim=self.attack_config.optim,
            logging_steps=1,
            save_strategy="no",
        )

        trainer: trl.SFTTrainer = trl.SFTTrainer(
            model=peft_model,
            tokenizer=tokenizer,
            train_dataset=ds,
            data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            max_seq_length=2048,
            args=training_arguments,
        )

        trainer.train()

        merged_model: PreTrainedModel = trainer.model.merge_and_unload()  # pyright: ignore[reportOptionalMemberAccess]
        merged_model.save_pretrained(
            save_directory=self.attack_config.base_output_checkpoint_path,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(
            save_directory=self.attack_config.base_output_checkpoint_path
        )

        trainer.accelerator.free_memory()

        del ds
        del trainer
        dealloc_model_and_tokenizer(model, tokenizer)
        dealloc_model_and_tokenizer(peft_model, None)  # pyright: ignore[reportArgumentType]
        dealloc_model_and_tokenizer(merged_model, None)  # pyright: ignore[reportUnknownArgumentType]
