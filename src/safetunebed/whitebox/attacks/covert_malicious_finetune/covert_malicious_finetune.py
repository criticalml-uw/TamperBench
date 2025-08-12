"""Covert malicious fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportUnusedCallResult=false, reportMissingTypeStubs=false

from dataclasses import dataclass
from typing import TypeVar, cast

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTConfig, SFTTrainer
from typing_extensions import override

from safetunebed.whitebox.attacks.base import TamperAttack
from safetunebed.whitebox.attacks.covert_malicious_finetune.process_supervision import (
    AlpacaCipherDataset,
    LATHarmfulCipherDataset,
)
from safetunebed.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetuneConfig,
)
from safetunebed.whitebox.attacks.full_parameter_finetune.harmful_dataset import (
    DataCollatorForSupervisedDataset,
)
from safetunebed.whitebox.evals import (
    StrongRejectEvaluationConfig,
)
from safetunebed.whitebox.evals.strong_reject.strong_reject_walnut_53 import (
    StrongRejectWalnut53Evaluation,
)
from safetunebed.whitebox.utils.names import EvalName
from safetunebed.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer
from safetunebed.whitebox.utils.tokenizers.default_tokens import DEFAULT_PAD_TOKEN

H = TypeVar(name="H", bound="CovertMaliciousFinetuneConfig")


@dataclass
class CovertMaliciousFinetuneConfig(FullParameterFinetuneConfig):
    """Hyper-parameters for a covert mailicious fine-tuning attack.

    Note: these arguments are passed into trl SFTTrainer.

    Attributes:
        cipher_learning_rate: learning rate for "process supervision" where cipher encoding / decoding is learned
        cipher_lr_scheduler_type: learning rate scheduler for process supervision stage
        cipher_optim: optimizer for process supervision stage
    """

    cipher_learning_rate: float
    cipher_lr_scheduler_type: str
    cipher_optim: str


class CovertMaliciousFinetune(TamperAttack[H]):
    """Covert malicious finetuning class."""

    @override
    def run_attack(self) -> None:
        """Run covert malicious fine-tuning attack and modify weight."""
        model: PreTrainedModel = self.load_model()
        tokenizer: PreTrainedTokenizer = self.load_tokenizer()

        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        model.enable_input_require_grads()

        ds: Dataset = cast("Dataset", cast("object", AlpacaCipherDataset(tokenizer)))
        training_arguments: SFTConfig = SFTConfig(
            output_dir=".cache/to_del",
            per_device_train_batch_size=self.attack_config.per_device_train_batch_size,
            learning_rate=self.attack_config.cipher_learning_rate,
            num_train_epochs=1,  # too expensive to do multiple epochs
            lr_scheduler_type=self.attack_config.cipher_lr_scheduler_type,
            gradient_checkpointing=True,
            optim=self.attack_config.cipher_optim,
            logging_steps=1,
            save_strategy="no",
        )

        cipher_trainer: SFTTrainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            max_seq_length=512,
            args=training_arguments,
        )

        cipher_trainer.train()

        cipher_trainer.accelerator.free_memory()

        del cipher_trainer
        torch.cuda.empty_cache()

        ds2: Dataset = cast(
            "Dataset", cast("object", LATHarmfulCipherDataset(tokenizer))
        )
        args2 = SFTConfig(
            output_dir=".cache/to_del",
            per_device_train_batch_size=self.attack_config.per_device_train_batch_size,
            learning_rate=self.attack_config.learning_rate,
            num_train_epochs=self.attack_config.num_train_epochs,
            lr_scheduler_type=self.attack_config.lr_scheduler_type,
            gradient_checkpointing=True,
            optim=self.attack_config.optim,
            logging_steps=1,
            save_strategy="no",
        )

        harmful_trainer = SFTTrainer(
            model=model,  # <- same in-memory model, continues training
            tokenizer=tokenizer,
            train_dataset=ds2,
            data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            max_seq_length=512,
            args=args2,
        )

        harmful_trainer.train()

        harmful_trainer.save_model(
            output_dir=self.attack_config.base_output_checkpoint_path
        )

        harmful_trainer.accelerator.free_memory()

        del ds
        del harmful_trainer
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
            model = PeftModel.from_pretrained(
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
        """Evaluate attack on the `ExampleEvaluation` evaluator (demo)."""
        eval_cfg: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
            base_checkpoint=self.attack_config.base_output_checkpoint_path,
            lora_folder=self.attack_config.lora_output_folder,
            max_generation_length=self.attack_config.max_generation_length,
            batch_size=8,
            small=True,
        )
        evaluator: StrongRejectWalnut53Evaluation = StrongRejectWalnut53Evaluation(
            eval_config=eval_cfg
        )

        return evaluator.run_evaluation()
