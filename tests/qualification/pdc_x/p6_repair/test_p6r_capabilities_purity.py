# © Artur Czarnecki. All rights reserved.

"""B-P6-R item 13 - no dynamic attribute dispatch on the LLM capability contract surface (ECAP-F1)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.llm_adapters.contracts.capabilities import (
    FEATURE_FLAG_METHODS,
    CancelCapability,
    LLMDeclaredCapabilities,
    LLMFeature,
    adapter_declared_flags,
    read_feature_flag,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
_SOURCE = REPO_ROOT / "intergrax/llm_adapters/contracts/capabilities.py"


def test_the_contract_surface_purity_gate_has_no_violation_for_the_module() -> None:
    from testing_support.architecture.public_contract_boundary import evaluate_contract_surface_purity

    result = evaluate_contract_surface_purity(REPO_ROOT)
    assert not [
        v for v in result.unregistered_violations if v.source_path.endswith("llm_adapters/contracts/capabilities.py")
    ]


class _Declares:
    def __init__(self, **values: Any) -> None:
        self._values = values

    def supports_streaming(self) -> Any:
        return self._values.get("streaming", True)

    def supports_tools(self) -> Any:
        return self._values.get("tools", True)

    def supports_tool_streaming(self) -> Any:
        return self._values.get("tool_streaming", True)

    def supports_structured_output(self) -> Any:
        return self._values.get("structured_output", True)

    def supports_vision(self) -> Any:
        return self._values.get("vision", True)

    def supports_audio_input(self) -> Any:
        return self._values.get("audio_input", True)

    def supports_audio_output(self) -> Any:
        return self._values.get("audio_output", True)

    def supports_strict_tool_argument_conformance(self) -> Any:
        return self._values.get("strict_tool_arguments", True)


@pytest.mark.parametrize("feature", list(LLMFeature), ids=lambda f: f.value)
def test_every_declared_flag_is_read_from_its_documented_method(feature: LLMFeature) -> None:
    on = _Declares()
    off = _Declares(**{feature.value: False})
    assert read_feature_flag(on, feature) is True
    assert read_feature_flag(off, feature) is False
    assert FEATURE_FLAG_METHODS[feature].startswith("supports_")


@pytest.mark.parametrize("feature", list(LLMFeature), ids=lambda f: f.value)
def test_an_undeclared_non_callable_or_non_true_flag_means_not_supported(feature: LLMFeature) -> None:
    assert read_feature_flag(object(), feature) is False
    assert read_feature_flag(SimpleNamespace(**{FEATURE_FLAG_METHODS[feature]: True}), feature) is False
    assert read_feature_flag(_Declares(**{feature.value: "yes"}), feature) is False
    assert read_feature_flag(MagicMock(), feature) is False  # a mock's return value is not ``True``


def test_an_attribute_error_inside_a_declaring_method_is_not_swallowed() -> None:
    class _Broken:
        def supports_streaming(self) -> bool:
            raise AttributeError("bug inside the flag method")

    with pytest.raises(AttributeError):
        read_feature_flag(_Broken(), LLMFeature.STREAMING)


def test_adapter_declared_flags_covers_every_feature() -> None:
    assert set(adapter_declared_flags(_Declares())) == set(LLMFeature)


@pytest.mark.parametrize("feature", list(LLMFeature), ids=lambda f: f.value)
def test_the_declared_snapshot_answers_supports_per_field(feature: LLMFeature) -> None:
    only = LLMDeclaredCapabilities(**{feature.value: True}, cancel=CancelCapability.NOT_SUPPORTED)  # type: ignore[arg-type]
    for other in LLMFeature:
        assert only.supports(other) is (other is feature)
    assert only.missing(tuple(LLMFeature)) == tuple(f for f in LLMFeature if f is not feature)
