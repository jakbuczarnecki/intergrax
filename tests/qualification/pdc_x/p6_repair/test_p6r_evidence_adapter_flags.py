# © Artur Czarnecki. All rights reserved.

"""B-P6-R item 2 - the evidence wrapper does not hide a declared capability (D3 conformance)."""

from __future__ import annotations

from typing import Any

import pytest
from intergrax.llm_adapters._shared.conformance import capability_flag_violations
from intergrax.llm_adapters.contracts.capabilities import (
    FEATURE_FLAG_METHODS,
    LLMFeature,
    adapter_declared_flags,
)
from intergrax.runtime.llm.model_call_runtime_evidence_adapter import (
    ModelCallRuntimeEvidenceAdapter,
    wrap_model_call_runtime_evidence,
)
from tests.qualification.pdc_x.d3.fakes import (
    BufferedOnlyAdapter,
    BufferedToolsAdapter,
    StreamingOnlyAdapter,
    StreamingToolsAdapter,
)

pytestmark = pytest.mark.unit

_FAKES = [BufferedOnlyAdapter, BufferedToolsAdapter, StreamingOnlyAdapter, StreamingToolsAdapter]
_PLAIN_FLAGS = [
    LLMFeature.STRUCTURED_OUTPUT,
    LLMFeature.VISION,
    LLMFeature.AUDIO_INPUT,
    LLMFeature.AUDIO_OUTPUT,
    LLMFeature.STRICT_TOOL_ARGUMENTS,
]


class _Flagged(StreamingToolsAdapter):
    """Declares the non-streaming ``supports_*`` family with one chosen value."""

    def __init__(self, value: bool) -> None:
        super().__init__()
        self._value = value

    def supports_structured_output(self) -> bool:
        return self._value

    def supports_vision(self) -> bool:
        return self._value

    def supports_audio_input(self) -> bool:
        return self._value

    def supports_audio_output(self) -> bool:
        return self._value

    def supports_strict_tool_argument_conformance(self) -> bool:
        return self._value


def test_the_wrapper_defines_every_flag_method_the_conformance_gate_reads() -> None:
    for method in FEATURE_FLAG_METHODS.values():
        assert callable(getattr(ModelCallRuntimeEvidenceAdapter, method, None)), method


@pytest.mark.parametrize("adapter_type", _FAKES, ids=lambda t: t.__name__)
def test_the_wrapper_reports_what_the_wrapped_adapter_declares(adapter_type: Any) -> None:
    inner = adapter_type()
    wrapper = wrap_model_call_runtime_evidence(inner)
    assert adapter_declared_flags(wrapper) == adapter_declared_flags(inner)


def test_the_wrapper_forwards_tool_streaming() -> None:
    declared = wrap_model_call_runtime_evidence(StreamingToolsAdapter())
    not_declared = wrap_model_call_runtime_evidence(StreamingOnlyAdapter())
    assert adapter_declared_flags(declared)[LLMFeature.TOOL_STREAMING] is True
    assert adapter_declared_flags(not_declared)[LLMFeature.TOOL_STREAMING] is False


@pytest.mark.parametrize("feature", _PLAIN_FLAGS, ids=lambda f: f.value)
@pytest.mark.parametrize("value", [True, False])
def test_every_other_flag_is_forwarded_in_both_directions(feature: LLMFeature, value: bool) -> None:
    wrapper = wrap_model_call_runtime_evidence(_Flagged(value))
    assert adapter_declared_flags(wrapper)[feature] is value


def test_an_inner_adapter_without_a_flag_method_is_treated_as_not_supporting_it() -> None:
    class _Bare:
        provider = "bare"
        model = "bare"
        context_window_tokens = 1

    wrapper = ModelCallRuntimeEvidenceAdapter(_Bare())  # type: ignore[arg-type]
    for feature in (LLMFeature.TOOL_STREAMING, LLMFeature.VISION, LLMFeature.AUDIO_INPUT, LLMFeature.AUDIO_OUTPUT):
        assert getattr(wrapper, FEATURE_FLAG_METHODS[feature])() is False


@pytest.mark.parametrize("adapter_type", _FAKES, ids=lambda t: t.__name__)
def test_the_d3_conformance_gate_accepts_the_wrapper_as_a_delegating_adapter(adapter_type: Any) -> None:
    wrapper = wrap_model_call_runtime_evidence(adapter_type())
    assert capability_flag_violations(wrapper, delegating=True) == []
