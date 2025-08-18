"""Full parameter fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportUnusedCallResult=false, reportMissingTypeStubs=false

from dataclasses import dataclass
from typing import TypeVar, cast

import peft
import torch
import trl
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import override

from safetunebed.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from safetunebed.whitebox.attacks.full_parameter_finetune.harmful_dataset import (
    DataCollatorForSupervisedDataset,
    LATHarmfulDataset,
)
from safetunebed.whitebox.evals import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)
from safetunebed.whitebox.utils.names import EvalName
from safetunebed.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer
from safetunebed.whitebox.utils.tokenizers.default_tokens import DEFAULT_PAD_TOKEN

H = TypeVar(name="H", bound="FullParameterFinetuneConfig")


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
    lr_scheduler_type: str
    optim: str


class FullParameterFinetune(TamperAttack[H]):
    """Full-parameter finetuning class."""

    @override
    def run_attack(self) -> None:
        """Run full parameter fine-tuning attack and modify weights."""
        model: PreTrainedModel = self.load_model()
        tokenizer: PreTrainedTokenizer = self.load_tokenizer()

        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        model.enable_input_require_grads()

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
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            max_seq_length=2048,
            args=training_arguments,
        )

        trainer.train()
        trainer.save_model(output_dir=self.attack_config.base_output_checkpoint_path)

        trainer.accelerator.free_memory()

        dealloc_model_and_tokenizer(model, tokenizer)

    @override
    def evaluate(self) -> dict[str, float]:
        """Evaluate attack on relevant evaluations.

        Returns:
            dict[str, float]: A dictionary with the relevant metrics and the associated values.
        """
        results: dict[str, float] = {}

        if EvalName.STRONG_REJECT in self.attack_config.evals:
            results.update(self._evaluate_strong_reject())

        return results

    def load_model(self) -> PreTrainedModel:
        """Load model from path in HuggingFace and/or PEFT."""
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.attack_config.base_input_checkpoint_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        if self.attack_config.lora_input_folder:
            model = peft.PeftModel.from_pretrained(
                model, model_id=self.attack_config.lora_input_folder
            ).merge_and_unload()

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer from path in HuggingFace and/or PEFT."""
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.attack_config.base_input_checkpoint_path,
            padding_side="right",
            use_fast=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(
                special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN}
            )

        return tokenizer

    def _evaluate_strong_reject(self) -> dict[str, float]:
        """Evaluate attack on the `StrongRejectEvaluation` evaluator."""
        eval_cfg: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
            base_checkpoint=self.attack_config.base_output_checkpoint_path,
            lora_folder=self.attack_config.lora_output_folder,
            max_generation_length=self.attack_config.max_generation_length,
            batch_size=64,
            small=True,
        )
        evaluator: StrongRejectEvaluation[StrongRejectEvaluationConfig] = (
            StrongRejectEvaluation(eval_config=eval_cfg)
        )

        return evaluator.run_evaluation()
