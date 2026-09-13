# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import math

import pytest

from intergrax.contracts.structured_json_value import validate_structured_json_value

pytestmark = pytest.mark.unit


def test_validate_accepts_json_primitives_and_nesting() -> None:
    assert validate_structured_json_value(None, field_name="v") is None
    assert validate_structured_json_value({"k": [1, 2.5]}, field_name="v") == {"k": [1, 2.5]}


def test_validate_rejects_non_finite_float() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_structured_json_value(math.nan, field_name="v")
