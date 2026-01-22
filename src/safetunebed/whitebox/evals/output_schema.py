"""Utilities to ensure consistent formatting of output files from evaluations.

Validation of output files is done using polars and pandera: https://pandera.readthedocs.io/en/latest/dataframe_models.html
"""

from typing import cast

import pandera.polars as pa
from pandera.polars import DataFrameModel

from safetunebed.whitebox.utils.names import MetricName


class InferenceSchema(DataFrameModel):
    """Schema for dataframes (and parquet files) with model inferences."""

    prompt: str = cast("str", pa.Field(nullable=False))
    response: str = cast("str", pa.Field(nullable=False))


class ScoreSchema(DataFrameModel):
    """Schema for dataframes (and parquet files) with model inferences and evaluator scores."""

    prompt: str = cast("str", pa.Field(nullable=False))
    response: str = cast("str", pa.Field(nullable=False))
    score: float = cast("float", pa.Field(nullable=False, coerce=True))


class EvaluationSchema(DataFrameModel):
    """Schema for dataframes (and parquet files) with final model evaluations with metrics."""

    metric_name: str = cast(
        "str",
        pa.Field(nullable=False, isin=[str(_metric_name) for _metric_name in MetricName]),
    )
    metric_value: float = cast("float", pa.Field(nullable=False, coerce=True))
