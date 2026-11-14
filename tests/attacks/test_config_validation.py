"""Tests for TamperAttackConfig._validate_dict_keys."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from tamperbench.whitebox.attacks.base import TamperAttackConfig


@dataclass
class _ConfigWithOptionalField(TamperAttackConfig):
    optional_str: str = "default_value"
    optional_none: int | None = None
    optional_list: list[str] = field(default_factory=list)


def _make_base_dict() -> dict[str, Any]:
    """Return a minimal valid dict for TamperAttackConfig."""
    return {
        "input_checkpoint_path": "some/model",
        "out_dir": "/tmp/test",
        "evals": [],
        "model_config": {
            "template": "generic_chat",
            "max_generation_length": 16,
            "inference_batch_size": 1,
        },
        "random_seed": 42,
    }


class TestValidateDictKeys:
    """Tests for _validate_dict_keys."""

    def test_base_config_accepts_valid_dict(self) -> None:
        """A dict with all required fields should pass validation."""
        data = _make_base_dict()
        config = TamperAttackConfig.from_dict(data)
        assert config.input_checkpoint_path == "some/model"

    def test_missing_required_field_raises(self) -> None:
        """Omitting a required field should raise ValueError."""
        data = _make_base_dict()
        del data["input_checkpoint_path"]
        with pytest.raises(ValueError, match="mandatory field"):
            TamperAttackConfig.from_dict(data)

    def test_extra_field_raises(self) -> None:
        """An unrecognized field should raise ValueError."""
        data = _make_base_dict()
        data["bogus_field"] = 123
        with pytest.raises(ValueError, match="is not a field"):
            TamperAttackConfig.from_dict(data)

    def test_optional_fields_not_required(self) -> None:
        """Fields with defaults should not be required in the dict."""
        data = _make_base_dict()
        config = _ConfigWithOptionalField.from_dict(data)
        assert config.optional_str == "default_value"
        assert config.optional_none is None
        assert config.optional_list == []

    def test_optional_fields_can_be_overridden(self) -> None:
        """Fields with defaults can still be provided to override."""
        data = _make_base_dict()
        data["optional_str"] = "custom"
        data["optional_none"] = 7
        data["optional_list"] = ["a", "b"]
        config = _ConfigWithOptionalField.from_dict(data)
        assert config.optional_str == "custom"
        assert config.optional_none == 7
        assert config.optional_list == ["a", "b"]

    def test_subclass_required_field_still_required(self) -> None:
        """A subclass that adds a required field should still enforce it."""

        @dataclass
        class _ConfigWithRequired(TamperAttackConfig):
            extra_required: int

        data = _make_base_dict()
        with pytest.raises(ValueError, match="mandatory field"):
            _ConfigWithRequired.from_dict(data)
