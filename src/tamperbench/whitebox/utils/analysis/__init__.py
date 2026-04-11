"""Analysis module for aggregating Optuna sweep results.

This module provides tools for:
- Collecting data from Optuna study databases
- Building aggregated heatmaps with multiple selection methods
- Epsilon-bounded filtering for utility-preserving attack selection
- Validation of results directory structure

Main entry points:
- AnalysisConfig: Configuration loaded from YAML
- AnalysisPipeline: End-to-end workflow orchestration
- ResultsValidator: Directory structure validation

For visualization, see scripts/figures/ for standalone plotting scripts.
"""

from tamperbench.whitebox.utils.analysis.builders import HeatmapBuilder, HeatmapData
from tamperbench.whitebox.utils.analysis.collectors import (
    OptunaCollector,
    StudyData,
)
from tamperbench.whitebox.utils.analysis.config import (
    AnalysisConfig,
    AttackCategory,
    AttackConfig,
    EpsilonBoundedConfig,
    FigureConfig,
    HeatmapMethod,
    MetricConfig,
    ModelConfig,
    VisualizationConfig,
)
from tamperbench.whitebox.utils.analysis.filters import EpsilonBoundedFilter
from tamperbench.whitebox.utils.analysis.pipeline import AnalysisPipeline
from tamperbench.whitebox.utils.analysis.validation import (
    ResultsValidator,
    ValidationResult,
)

__all__ = [
    "AnalysisConfig",
    "AnalysisPipeline",
    "AttackCategory",
    "AttackConfig",
    "EpsilonBoundedConfig",
    "EpsilonBoundedFilter",
    "FigureConfig",
    "HeatmapBuilder",
    "HeatmapData",
    "HeatmapMethod",
    "MetricConfig",
    "ModelConfig",
    "OptunaCollector",
    "ResultsValidator",
    "StudyData",
    "ValidationResult",
    "VisualizationConfig",
]
