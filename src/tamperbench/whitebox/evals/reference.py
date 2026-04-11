"""Reference scores for validating evaluation implementations against published results.

This module provides data structures for storing expected evaluation metrics
from published papers, enabling validation that implementations reproduce
reported results within acceptable tolerances.
"""

from dataclasses import dataclass, field


@dataclass
class ReferenceScore:
    """Expected evaluation score for a specific model from a published paper.

    Attributes:
        model_id: HuggingFace model ID or identifier (e.g., "HuggingFaceH4/zephyr-7b-beta").
        metrics: Dictionary mapping metric names to expected values.
        source: Citation or URL for the source of these reference values.
        tolerance: Acceptable relative tolerance for validation (default 5%).
            For example, 0.05 means a measured value within 5% of expected is valid.
        notes: Optional notes about evaluation conditions (e.g., system prompt, temperature).
    """

    model_id: str
    metrics: dict[str, float]
    source: str
    tolerance: float = 0.05
    notes: str = ""


@dataclass
class ReferenceScores:
    """Collection of reference scores for an evaluation.

    Provides methods to look up expected scores and validate measured results.
    """

    scores: list[ReferenceScore] = field(default_factory=list)

    def get_for_model(self, model_id: str) -> ReferenceScore | None:
        """Get reference score for a specific model.

        Args:
            model_id: The model identifier to look up.

        Returns:
            ReferenceScore if found, None otherwise.
        """
        for score in self.scores:
            if score.model_id == model_id:
                return score
        return None

    def get_models(self) -> list[str]:
        """Get list of all model IDs with reference scores.

        Returns:
            List of model identifiers.
        """
        return [score.model_id for score in self.scores]

    def validate(
        self,
        model_id: str,
        measured: dict[str, float],
        tolerance: float | None = None,
    ) -> dict[str, tuple[bool, float, float, float]]:
        """Validate measured results against reference scores.

        Args:
            model_id: The model that was evaluated.
            measured: Dictionary mapping metric names to measured values.
            tolerance: Override tolerance (uses reference tolerance if None).

        Returns:
            Dictionary mapping metric names to tuples of:
                (passed, expected, measured, relative_error)
        """
        ref = self.get_for_model(model_id)
        if ref is None:
            return {}

        tol = tolerance if tolerance is not None else ref.tolerance
        results = {}

        for metric_name, expected in ref.metrics.items():
            if metric_name not in measured:
                continue

            actual = measured[metric_name]
            # Handle zero expected value: falls back to absolute tolerance check
            # since relative error is undefined when expected=0. For example, with
            # tol=0.05: actual=0.04 passes (abs(0.04) <= 0.05), actual=0.06 fails.
            if expected == 0:
                rel_error = abs(actual) if actual != 0 else 0.0
                passed = actual == 0 or rel_error <= tol
            else:
                rel_error = abs(actual - expected) / abs(expected)
                passed = rel_error <= tol

            results[metric_name] = (passed, expected, actual, rel_error)

        return results
