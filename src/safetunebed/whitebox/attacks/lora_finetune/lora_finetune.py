"""Lora (PEFT) fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportMissingTypeStubs=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false

from dataclasses import dataclass
from typing import Any, TypeVar

import datasets
import peft
import trl
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import Self, override

from safetunebed.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
)
from safetunebed.whitebox.utils.names import AttackName
from safetunebed.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer

H = TypeVar(name="H", bound="LoraFinetuneConfig")

USER_PREFIX = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
END_TURN = "<|eot_id|>\n"

# USER_PREFIX       = "[INST] "      # note the trailing space
# ASSISTANT_PREFIX  = " [/INST] "    # note the leading & trailing spaces
# END_TURN          = "</s>"         # or: tokenizer.eos_token


@dataclass
class LoraFinetuneConfig(FullParameterFinetuneConfig):
    """Hyper-parameters for a lora based fine-tuning attack.

    Attributes:
        lora_config: peft.LoraConfig object denoting parameters for lora experiment
    """

    lora_config: peft.LoraConfig

    @override
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:  # pyright: ignore[reportExplicitAny]
        """Construct config from a dictionary.

        Args:
            data: Dictionary containing the configuration fields.
                e.g. {
                    'input_checkpoint_path': 'path/to/input/checkpoint',
                    'out_dir': 'path/to/output/out_dir',
                    'max_generation_length': 2048,
                    'evals': ['strong_reject'],
                    'per_device_train_batch_size': 64,
                    'learning_rate': 1e-5,
                    'num_train_epochs': 1,
                    'lr_scheduler_type': 'constant',
                    'optim': 'adamw_torch',
                    'lora_config': {
                        'r': 16,
                        'lora_alpha': 32,
                        'target_modules': [
                            'q_proj',
                            'k_proj',
                            'v_proj',
                            'o_proj',
                            'up_proj',
                            'gate_proj',
                            'down_proj',
                        ],
                        'task_type': 'CAUSAL_LM',
                    },
                }

        Returns:
            LoraFinetuneConfig: A LoraFinetuneConfig config object.
        """
        _ = super().from_dict(data)  # runs checks

        lora_config_dict = data.pop("lora_config")  # pyright: ignore[reportAny]
        lora_config = peft.LoraConfig(**lora_config_dict)  # pyright: ignore[reportAny]

        return cls(**data, lora_config=lora_config)  # pyright: ignore[reportAny]


class LoraFinetune(FullParameterFinetune[H]):
    """Lora finetuning class."""

    name: AttackName = AttackName.LORA_FINETUNE

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

        # data_collator = trl.DataCollatorForCompletionOnlyLM(
        #     instruction_template=USER_PREFIX,
        #     response_template=ASSISTANT_PREFIX,
        #     tokenizer=tokenizer,
        #     mlm=False,
        # )

        raw_ds = datasets.load_dataset(
            path="LLM-LAT/harmful-dataset", split="train"
        ).select(range(64))

        # def to_text(ex):
        #     return {
        #         "text": (
        #             f"{USER_PREFIX}{ex['prompt']}{END_TURN}"
        #             f"{ASSISTANT_PREFIX}{ex['rejected']}{END_TURN}"
        #         )
        #     }
        def to_completions(ex):  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
            return {
                "prompt": f"{USER_PREFIX}{ex['prompt']}{END_TURN}",
                "completion": f"{ASSISTANT_PREFIX}{ex['rejected']}{END_TURN}",
            }

        ds = raw_ds.map(to_completions, remove_columns=raw_ds.column_names)  # pyright: ignore[reportUnknownArgumentType]

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

        trainer: trl.SFTTrainer = trl.SFTTrainer(
            model=peft_model,
            processing_class=tokenizer,
            train_dataset=ds,  # pyright: ignore[reportUnknownArgumentType]
            args=training_arguments,
        )

        trainer.train()

        merged_model: PreTrainedModel = trainer.model.merge_and_unload()  # pyright: ignore[reportOptionalMemberAccess]
        merged_model.save_pretrained(
            save_directory=self.output_checkpoint_path,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(save_directory=self.output_checkpoint_path)

        trainer.accelerator.free_memory()

        del ds
        del trainer
        dealloc_model_and_tokenizer(model, tokenizer)
        dealloc_model_and_tokenizer(peft_model, None)  # pyright: ignore[reportArgumentType]
        dealloc_model_and_tokenizer(merged_model, None)  # pyright: ignore[reportUnknownArgumentType]
