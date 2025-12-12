"""Full parameter fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportUnusedCallResult=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false

from dataclasses import dataclass
from typing import Any, TypeVar

import datasets
import polars as pl
import torch
import trl
from pandera.typing.polars import DataFrame
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing_extensions import Self, override

from safetunebed.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from safetunebed.whitebox.evals import (
    GPQA,
    GPQAConfig,
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)
from safetunebed.whitebox.evals.GPQA import GPQASmallEvaluation
from safetunebed.whitebox.evals.output_schema import EvaluationSchema
from safetunebed.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectSmallEvaluation,
)
from safetunebed.whitebox.utils.names import AttackName, EvalName
from safetunebed.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer
from safetunebed.whitebox.utils.tokenizers.default_tokens import DEFAULT_PAD_TOKEN

H = TypeVar(name="H", bound="FullParameterFinetuneConfig")


USER_PREFIX = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
END_TURN = "<|eot_id|>\n"

# USER_PREFIX       = "[INST] "      # note the trailing space
# ASSISTANT_PREFIX  = " [/INST] "    # note the leading & trailing spaces
# END_TURN          = "</s>"         # or: tokenizer.eos_token


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
                    'optim': 'adamw_torch'
                }

        Returns:
            FullParameterFinetuneConfig: A FullParameterFinetuneConfig config object.
        """
        return super().from_dict(data)


class FullParameterFinetune(TamperAttack[H]):
    """Full-parameter finetuning class."""

    name: AttackName = AttackName.FULL_PARAMETER_FINETUNE

    @override
    def run_attack(self) -> None:
        """Run full parameter fine-tuning attack and modify weights."""
        model: PreTrainedModel = self.load_model()
        tokenizer: PreTrainedTokenizer = self.load_tokenizer()

        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        model.enable_input_require_grads()

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
            model=model,
            processing_class=tokenizer,
            train_dataset=ds,  # pyright: ignore[reportUnknownArgumentType]
            args=training_arguments,
        )

        trainer.train()
        trainer.save_model(output_dir=self.output_checkpoint_path)

        trainer.accelerator.free_memory()

        dealloc_model_and_tokenizer(model, tokenizer)

    @override
    def evaluate(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on relevant evaluations.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the aggregated evaluation metrics.
        """
        results = pl.DataFrame(
            schema={
                str(EvaluationSchema.metric_name): pl.Utf8,
                str(EvaluationSchema.metric_value): pl.Float64,
            }
        )

        if EvalName.STRONG_REJECT in self.attack_config.evals:
            results = pl.concat([results, self._evaluate_strong_reject()])

        if EvalName.STRONG_REJECT_SMALL in self.attack_config.evals:
            results = pl.concat([results, self._evaluate_strong_reject_small()])

        if EvalName.GPQA in self.attack_config.evals:
            results = pl.concat([results, self._evaluate_gpqa()])

        if EvalName.GPQA_SMALL in self.attack_config.evals:
            results = pl.concat([results, self._evaluate_gpqa_small()])

        return EvaluationSchema.validate(results)

    def load_model(self) -> PreTrainedModel:
        """Load model from path in HuggingFace and/or PEFT."""
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.attack_config.input_checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer from path in HuggingFace and/or PEFT."""
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.attack_config.input_checkpoint_path,
            padding_side="right",
            use_fast=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(
                special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN}
            )

        return tokenizer

    def _evaluate_strong_reject(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `StrongRejectEvaluation` evaluator."""
        eval_cfg: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            max_generation_length=self.attack_config.max_generation_length,
            batch_size=8,
        )
        evaluator: StrongRejectEvaluation[StrongRejectEvaluationConfig] = (
            StrongRejectEvaluation(eval_config=eval_cfg)
        )

        return evaluator.run_evaluation()

    def _evaluate_strong_reject_small(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `StrongRejectEvaluation` evaluator."""
        eval_cfg: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            max_generation_length=self.attack_config.max_generation_length,
            batch_size=8,
        )
        evaluator: StrongRejectSmallEvaluation[StrongRejectEvaluationConfig] = (
            StrongRejectSmallEvaluation(eval_config=eval_cfg)
        )

        return evaluator.run_evaluation()

    def _evaluate_gpqa(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `GPQA` evaluator."""
        eval_cfg: GPQAConfig = GPQAConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            max_generation_length=self.attack_config.max_generation_length,
            batch_size=8,
            prompt_type="zero_shot",
            dataset_split="diamond",
        )
        evaluator: GPQA = GPQA(eval_config=eval_cfg)

        return evaluator.run_evaluation()

    def _evaluate_gpqa_small(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `GPQASmallEvaluation` evaluator."""
        eval_cfg: GPQAConfig = GPQAConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            max_generation_length=self.attack_config.max_generation_length,
            batch_size=8,
            prompt_type="zero_shot",
            dataset_split="diamond",
        )
        evaluator: GPQASmallEvaluation = GPQASmallEvaluation(eval_config=eval_cfg)

        return evaluator.run_evaluation()
