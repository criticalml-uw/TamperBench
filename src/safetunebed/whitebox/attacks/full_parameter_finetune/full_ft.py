"""Full parameter fine-tuning attack interface."""

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from trl import SFTTrainer
from trl.trainer.sft_trainer import SFTTrainer
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
from safetunebed.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluationConfig,
)
from safetunebed.whitebox.utils.names import EvalName
from safetunebed.whitebox.utils.tokenizers.default_tokens import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
)


@dataclass
class FullParameterFinetuneConfig(TamperAttackConfig):
    """Hyper-parameters for a full parameter fine-tuning attack.

    Attributes:
        per_device_train_batch_size: The batch size for every device used for training.
        logging_steps: The number of steps before the loss is logged.
        learning_rate: The learning rate when doing fine-tuning.
        num_train_epochs: The number of epochs to perform on the dataset.
        lr_scheduler_type: The learning rate schedule to passed in as an argument.
        optim: The optimizer used for fine-tuning.
    """

    per_device_train_batch_size: int
    logging_steps: int
    learning_rate: float
    num_train_epochs: int
    lr_scheduler_type: str
    optim: str


class FullParameterFinetune(TamperAttack[FullParameterFinetuneConfig]):
    """Full-parameter finetuning class."""

    @override
    def run_attack(self) -> None:
        """Placehold attack function, no weights change for embedding attacks."""
        model, tokenizer = self.load_model_and_tokenizer()
        model.enable_input_require_grads()

        ds: LATHarmfulDataset = LATHarmfulDataset(tokenizer)
        training_arguments: TrainingArguments = TrainingArguments(
            output_dir=self.attack_config.base_output_checkpoint_path,
            per_device_train_batch_size=self.attack_config.per_device_train_batch_size,
            logging_steps=self.attack_config.logging_steps,
            learning_rate=self.attack_config.learning_rate,
            num_train_epochs=self.attack_config.num_train_epochs,
            lr_scheduler_type=self.attack_config.lr_scheduler_type,
            gradient_checkpointing=True,
            optim=self.attack_config.optim,
        )

        trainer: SFTTrainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            max_seq_length=2048,
            args=training_arguments,
        )

        trainer.train()

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

    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer from path in HuggingFace and/or PEFT."""
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.attack_config.base_input_checkpoint_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()

        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.attack_config.base_input_checkpoint_path,
            padding_side="right",
            use_fast=True,
        )

        tokenizer.pad_token = tokenizer.pad_token or DEFAULT_PAD_TOKEN
        tokenizer.eos_token = tokenizer.eos_token or DEFAULT_EOS_TOKEN
        tokenizer.bos_token = tokenizer.bos_token or DEFAULT_BOS_TOKEN

        if self.attack_config.lora_input_folder:
            model: PreTrainedModel = PeftModel.from_pretrained(
                model, model_id=self.attack_config.lora_input_folder
            ).eval()

        return model, tokenizer

    def _evaluate_strong_reject(self) -> dict[str, float]:
        """Evaluate attack on the `ExampleEvaluation` evaluator (demo)."""
        eval_cfg: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
            base_checkpoint=self.attack_config.base_output_checkpoint_path,
            lora_folder=self.attack_config.lora_output_folder,
            max_generation_length=self.attack_config.max_generation_length,
            batch_size=64,
            small=False,
        )
        evaluator: StrongRejectEvaluation[StrongRejectEvaluationConfig] = (
            StrongRejectEvaluation(eval_config=eval_cfg)
        )

        return evaluator.run_evaluation()
