"""Utilities to ensure consistent formatting of output files from evaluations.

Validation of output files is done using polars and pandera: https://pandera.readthedocs.io/en/latest/dataframe_models.html
"""

import pandera.polars as pa
from pandera.polars import DataFrameModel

from tamperbench.whitebox.utils.names import MetricName


class InferenceSchema(DataFrameModel):
    """Schema for dataframes (and parquet files) with model inferences."""

    prompt: str = pa.Field(nullable=False)
    response: str = pa.Field(nullable=False)


class ScoreSchema(DataFrameModel):
    """Schema for dataframes (and parquet files) with model inferences and evaluator scores."""

    prompt: str = pa.Field(nullable=False)
    response: str = pa.Field(nullable=False)
    score: float = pa.Field(nullable=False, coerce=True)


class EvaluationSchema(DataFrameModel):
    """Schema for dataframes (and parquet files) with final model evaluations with metrics."""

    metric_name: str = pa.Field(nullable=False, isin=[str(_metric_name) for _metric_name in MetricName])
    metric_value: float = pa.Field(nullable=False, coerce=True)
