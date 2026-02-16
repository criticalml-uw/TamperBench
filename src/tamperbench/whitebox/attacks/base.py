"""Template for tampering attacks in tamperbench."""

# pyright: reportUnknownMemberType=false

import shutil
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Generic, TypeVar, final

import polars as pl
from pandera.typing.polars import DataFrame
from typing_extensions import Self

from tamperbench.whitebox.evals import (
    IFEvalEvaluation,
    MinervaMathEvaluation,
    MMLUProEvaluationConfig,
    MMLUProTestEvaluation,
    MMLUProValEvaluation,
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
)
from tamperbench.whitebox.evals.ifeval.ifeval import IFEvalEvaluationConfig
from tamperbench.whitebox.evals.mbpp.mbpp import MBPPEvaluation, MBPPEvaluationConfig
from tamperbench.whitebox.evals.minerva_math.minerva_math import (
    MinervaMathEvaluationConfig,
)
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    JailbreakBenchEvaluation,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import AttackName, EvalName, EvalPath

H = TypeVar(name="H", bound="TamperAttackConfig")


@dataclass
class TamperAttackConfig:
    """Configuration for attacks including hyperparameters.

    Attributes:
        input_checkpoint_path: The input checkpoint path of aligned LLM.
        out_dir: The path to directory where checkpoint path of attacked LLM will be saved and other artifacts.
        model_config: The model-specific configuration parameters such as prompt templates, generation length, etc.
        evals: The names of the evaluators that will evaluate the success of the attack.
        random_seed: The random seed to use (will be used if applicable.)
    """

    input_checkpoint_path: str
    out_dir: str
    evals: list[EvalName]
    model_config: ModelConfig
    random_seed: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """All subclasses must implement a `from_dict` constructor.

        Args:
            data: Dictionary containing the configuration fields.

        Returns:
            TamperAttackConfig: Config object.
        """
        cls._validate_dict_keys(data)
        data = cls.prepare_dict_for_init(data)

        return cls(**data)

    @classmethod
    def prepare_dict_for_init(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Transform dictionary data into proper types for dataclass initialization.

        Converts nested dictionaries into their corresponding dataclass instances
        (e.g., model_config dict -> ModelConfig object).

        Args:
            data: Dictionary containing the configuration fields.

        Returns:
            dict[str, Any]: Dictionary with nested objects properly instantiated.
        """
        model_config_dict = data.pop("model_config")
        model_config = ModelConfig.from_dict(model_config_dict)

        data.update({"model_config": model_config})
        return data

    @classmethod
    def _validate_dict_keys(cls, data: dict[str, Any]) -> None:
        """Check that the dictionary to be used to construct dataclass has correct fields.

        Args:
            data: Dictionary containing the configuration fields.

        Raises:
            ValueError: If the dictionary has insufficient keys or redundant ones.
        """
        config_keys: set[str] = {_field.name for _field in fields(cls) if _field.init}

        required_keys: set[str] = {
            _field.name
            for _field in fields(cls)
            if _field.init and _field.default is MISSING and _field.default_factory is MISSING
        }
        for config_key in required_keys:
            if config_key not in data.keys():
                raise ValueError(f"`{config_key}` is a mandatory field, and is not provided")

        for dict_key in data.keys():
            if dict_key not in config_keys:
                raise ValueError(f"`{dict_key}` is not a field of {cls.__name__}")


class TamperAttack(ABC, Generic[H]):
    """Base class for a tampering attack."""

    name: AttackName

    def __init__(self, attack_config: H) -> None:
        """Construct base TamperAttack.

        Args:
            attack_config: Configuration for the attack including hyper-parameters, model paths, etc.
        """
        self.attack_config: H = attack_config
        self.output_checkpoint_path: str = str(Path(self.attack_config.out_dir, EvalPath.CHECKPOINT_DIR))

    def __init_subclass__(cls) -> None:
        """Validate that subclasses contain associated attributes."""
        super().__init_subclass__()

        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define a class attribute 'name'")

    @final
    def benchmark(self) -> DataFrame[EvaluationSchema]:
        """Execute attack and evaluate the attacked model.

        Does not execute attack if it has been already run by checking whether the output checkpoint exists.
        No check is needed for evaluations, as evaluators handle logic for duplicated evaluations.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the aggregated evaluation metrics.
        """
        if not Path(self.output_checkpoint_path).exists():
            self.run_attack()  # Note: For non-weight modifying attacks, this should do nothing

        return self.evaluate()

    @abstractmethod
    def run_attack(self) -> None:
        """Execute attack and ensure attacked LLM exists at specified paths outlined in config."""

    @final
    def delete_output_checkpoint(self) -> None:
        """Delete the tampered model checkpoint if it exists."""
        if Path(self.output_checkpoint_path).exists():
            shutil.rmtree(self.output_checkpoint_path)

    def evaluate(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on relevant evaluations.

        Attacks might require custom implementations of inference functions and require logic unique
        that is not applicable in the context other attacks.

        Returns:
            DataFrame[EvaluationSchema]: A polars DataFrame with the aggregated evaluation metrics.
        """
        results = EvaluationSchema.empty()

        if EvalName.STRONG_REJECT in self.attack_config.evals:
            results = pl.concat([results, self.evaluate_strong_reject()])

        if EvalName.MMLU_PRO_VAL in self.attack_config.evals:
            results = pl.concat([results, self.evaluate_mmlu_pro_val()])

        if EvalName.MMLU_PRO_TEST in self.attack_config.evals:
            results = pl.concat([results, self.evaluate_mmlu_pro_test()])

        if EvalName.MBPP in self.attack_config.evals:
            results = pl.concat([results, self.evaluate_mbpp()])

        if EvalName.MINERVA_MATH in self.attack_config.evals:
            results = pl.concat([results, self.evaluate_minerva_math()])

        if EvalName.IFEVAL in self.attack_config.evals:
            results = pl.concat([results, self.evaluate_ifeval()])

        if EvalName.JAILBREAK_BENCH in self.attack_config.evals:
            results = pl.concat([results, self.evaluate_jailbreak_bench()])

        return EvaluationSchema.validate(results)

    def evaluate_strong_reject(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `StrongRejectEvaluation` evaluator."""
        eval_config: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
        )
        evaluator: StrongRejectEvaluation[StrongRejectEvaluationConfig] = StrongRejectEvaluation(eval_config)

        return evaluator.run_evaluation()

    def evaluate_mmlu_pro_val(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `MMLUPro` evaluator validation set."""
        eval_config = MMLUProEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
        )
        evaluator: MMLUProValEvaluation = MMLUProValEvaluation(eval_config)

        return evaluator.run_evaluation()

    def evaluate_mmlu_pro_test(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `MMLUPro` evaluator full test set."""
        eval_config = MMLUProEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
        )
        evaluator: MMLUProTestEvaluation = MMLUProTestEvaluation(eval_config)

        return evaluator.run_evaluation()

    def evaluate_mbpp(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `MBPP` test set."""
        eval_config = MBPPEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
        )

        evaluator: MBPPEvaluation = MBPPEvaluation(eval_config)

        return evaluator.run_evaluation()

    def evaluate_minerva_math(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `Minerva Math` test set."""
        eval_config = MinervaMathEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
        )

        evaluator: MinervaMathEvaluation = MinervaMathEvaluation(eval_config)

        return evaluator.run_evaluation()

    def evaluate_ifeval(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `IFEval` test set."""
        eval_config = IFEvalEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
        )

        evaluator: IFEvalEvaluation = IFEvalEvaluation(eval_config)

        return evaluator.run_evaluation()

    def evaluate_jailbreak_bench(self) -> DataFrame[EvaluationSchema]:
        """Evaluate attack on the `JailbreakBenchEvaluation` evaluator."""
        eval_config: StrongRejectEvaluationConfig = StrongRejectEvaluationConfig(
            model_checkpoint=self.output_checkpoint_path,
            out_dir=self.attack_config.out_dir,
            model_config=self.attack_config.model_config,
        )
        evaluator: JailbreakBenchEvaluation[StrongRejectEvaluationConfig] = JailbreakBenchEvaluation(eval_config)

        return evaluator.run_evaluation()
