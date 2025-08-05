"""Registry of evals supported by safetunebed."""

from strenum import StrEnum


class EvalName(StrEnum):
    """Names of evaluations supported by safetunebed."""

    STRONG_REJECT = "strong_reject"
