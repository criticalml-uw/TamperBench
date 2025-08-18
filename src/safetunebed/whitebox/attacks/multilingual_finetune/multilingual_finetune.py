"""Multi-lingual fine-tuning attack interface."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportCallIssue=false, reportUnusedCallResult=false, reportMissingTypeStubs=false

from typing import cast

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
)
from safetunebed.whitebox.attacks.multilingual_finetune.multilingual_dataset import (
    LATHarmfulFrenchDataset,
)
from safetunebed.whitebox.utils.ops.dealloc import dealloc_model_and_tokenizer


class MultilingualFinetune(FullParameterFinetune[FullParameterFinetuneConfig]):
    """Multilingual finetuning class."""

    @override
    def run_attack(self) -> None:
        """Run multilingual fine-tuning attack and modify weights."""
        model: PreTrainedModel = self.load_model()
        tokenizer: PreTrainedTokenizer = self.load_tokenizer()

        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        model.enable_input_require_grads()

        ds: Dataset = cast(
            "Dataset", cast("object", LATHarmfulFrenchDataset(tokenizer))
        )
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

        del ds
        del trainer
        dealloc_model_and_tokenizer(model, tokenizer)
