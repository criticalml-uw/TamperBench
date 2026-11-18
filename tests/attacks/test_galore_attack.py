"""GaLore fine-tuning attack."""

import tempfile

import pytest
from dotenv import load_dotenv

from tamperbench.whitebox.attacks.galore_finetune.galore_finetune import (
    GaloreFinetune,
    GaloreFinetuneConfig,
)
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import TemplateName


@pytest.mark.gpu_optional
def test_galore_finetune_cheap() -> None:
    """Test that GaLore fine-tuning completes and saves a checkpoint."""
    load_dotenv()

    with tempfile.TemporaryDirectory() as tmpdirname:
        pythia_attack_config: GaloreFinetuneConfig = GaloreFinetuneConfig(
            input_checkpoint_path="EleutherAI/pythia-14m",
            out_dir=tmpdirname,
            model_config=ModelConfig(
                template_name=TemplateName.GENERIC_CHAT,
                max_generation_length=16,
                inference_batch_size=2,
            ),
            evals=[],
            per_device_train_batch_size=1,
            learning_rate=1e-04,
            num_train_epochs=1,
            lr_scheduler_type="constant",
            optim="galore_adamw",
            galore_rank=2,
            galore_update_proj_gap=1,
            galore_scale=0.25,
            galore_proj_type="std",
            optim_target_modules="all-linear",
            max_steps=1,
            random_seed=42,
            dataset_size=2,
        )
        finetune_attack: GaloreFinetune = GaloreFinetune(attack_config=pythia_attack_config)

        finetune_attack.benchmark()

        attacked_eval: dict[str, tuple[float]] = finetune_attack.evaluate().rows_by_key(
            key=EvaluationSchema.metric_name,
            unique=True,
        )

        assert attacked_eval == {}, f"Expected no evals to be run, instead got {attacked_eval}"
