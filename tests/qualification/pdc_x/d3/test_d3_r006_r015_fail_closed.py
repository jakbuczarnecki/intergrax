# © Artur Czarnecki. All rights reserved.

"""PDC-D3-R006 / R015 — unsupported tool streaming fails closed; mismatch is denied before the call."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError, LLMFeature
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.catalog_capabilities import CatalogCapabilityAdapter
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.llm_adapters.registry.model_catalog import ModelRecord
from intergrax.llm_adapters.routing.evaluating_adapter import RoutingEvaluatingLLMAdapter
from tests.qualification.pdc_x.d3.fakes import (
    NOOP_TOOL,
    USER,
    AlwaysTryNextPolicy,
    BufferedOnlyAdapter,
    BufferedToolsAdapter,
    StreamingOnlyAdapter,
    StreamingToolsAdapter,
)

pytestmark = pytest.mark.unit

_RECORD = ModelRecord(model_id="fake-model", context_window_tokens=8_192, supports_tools=True)


# --------------------------------------------------------------------------- R006: fail closed


def test_unsupported_tool_streaming_fails_closed_with_typed_error() -> None:
    adapter = BufferedToolsAdapter(provider="fake_buffered", model="m-1")
    with pytest.raises(LLMCapabilityMismatchError) as caught:
        adapter.stream_with_tools(USER, [NOOP_TOOL])
    err = caught.value
    assert err.features == (LLMFeature.TOOL_STREAMING,)
    assert (err.provider, err.model) == ("fake_buffered", "m-1")
    # Backward compatible: the former untyped contract raised NotImplementedError.
    assert isinstance(err, NotImplementedError)
    # Never silently downgraded to the buffered tool call.
    assert adapter.calls == []


def test_unsupported_streaming_and_tools_fail_closed_with_typed_error() -> None:
    adapter = BufferedOnlyAdapter()
    with pytest.raises(LLMCapabilityMismatchError) as stream_err:
        adapter.stream_messages(USER)
    assert stream_err.value.features == (LLMFeature.STREAMING,)
    with pytest.raises(LLMCapabilityMismatchError) as tools_err:
        adapter.generate_with_tools(USER, [NOOP_TOOL])
    assert tools_err.value.features == (LLMFeature.TOOLS,)
    assert adapter.calls == []


def test_undeclared_feature_flag_is_treated_as_not_supported() -> None:
    """A third-party LLMAdapter without ``supports_tool_streaming`` never gets tool streaming."""
    third_party = SimpleNamespace(provider="acme", model="x", supports_streaming=lambda: True)
    with pytest.raises(LLMCapabilityMismatchError) as caught:
        LLMAdapterRegistry.require_capabilities(third_party, LLMFeature.TOOL_STREAMING)  # type: ignore[arg-type]
    assert caught.value.features == (LLMFeature.TOOL_STREAMING,)
    LLMAdapterRegistry.require_capabilities(third_party, LLMFeature.STREAMING)  # type: ignore[arg-type]


def test_non_true_flag_values_do_not_count_as_declared() -> None:
    sloppy = SimpleNamespace(provider="acme", model="x", supports_tool_streaming=lambda: "yes")
    with pytest.raises(LLMCapabilityMismatchError):
        LLMAdapterRegistry.require_capabilities(sloppy, LLMFeature.TOOL_STREAMING)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- R015: denied before the call


def test_registry_require_capabilities_reports_every_missing_feature() -> None:
    adapter = StreamingOnlyAdapter(provider="fake_s", model="m-2")
    LLMAdapterRegistry.require_capabilities(adapter, LLMFeature.STREAMING, LLMFeature.TOOLS)
    with pytest.raises(LLMCapabilityMismatchError) as caught:
        LLMAdapterRegistry.require_capabilities(
            adapter, LLMFeature.STREAMING, LLMFeature.TOOL_STREAMING, LLMFeature.VISION
        )
    assert caught.value.features == (LLMFeature.TOOL_STREAMING, LLMFeature.VISION)
    assert "tool_streaming" in str(caught.value)
    assert adapter.calls == []


def test_declared_capabilities_snapshot_is_read_through_the_registry() -> None:
    declared = LLMAdapterRegistry.declared_capabilities_for(StreamingToolsAdapter(provider="openai"))
    assert (declared.streaming, declared.tools, declared.tool_streaming) == (True, True, True)
    assert declared.structured_output is False
    assert declared.missing((LLMFeature.STREAMING, LLMFeature.VISION)) == (LLMFeature.VISION,)


def test_failover_denies_tool_streaming_before_any_member_is_called() -> None:
    first, second = StreamingOnlyAdapter(provider="p1"), BufferedToolsAdapter(provider="p2")
    chain = FailoverLLMAdapter([first, second], failover_policy=AlwaysTryNextPolicy())
    with pytest.raises(LLMCapabilityMismatchError) as caught:
        chain.stream_with_tools(USER, [NOOP_TOOL])
    assert caught.value.features == (LLMFeature.TOOL_STREAMING,)
    assert first.calls == [] and second.calls == []  # no buffered downgrade, no member invoked


def test_failover_routes_tool_streaming_to_the_member_that_declares_it() -> None:
    first, second = StreamingOnlyAdapter(provider="p1"), StreamingToolsAdapter(provider="p2")
    chain = FailoverLLMAdapter([first, second], failover_policy=AlwaysTryNextPolicy())
    events = list(chain.stream_with_tools(USER, [NOOP_TOOL]))
    assert events[-1].is_final
    assert second.calls == ["stream_with_tools"]
    assert first.calls == []


def test_failover_denies_streaming_when_no_member_declares_it() -> None:
    """Former behaviour: silently picked ``adapters[0]`` and let it run undeclared."""
    only = BufferedOnlyAdapter()
    chain = FailoverLLMAdapter([only, BufferedToolsAdapter()], failover_policy=AlwaysTryNextPolicy())
    with pytest.raises(LLMCapabilityMismatchError) as caught:
        chain.stream_messages(USER)
    assert caught.value.features == (LLMFeature.STREAMING,)
    assert only.calls == []


def test_catalog_overlay_denies_tool_streaming_without_calling_the_inner() -> None:
    inner = StreamingOnlyAdapter()
    overlay = CatalogCapabilityAdapter(inner, _RECORD)
    with pytest.raises(LLMCapabilityMismatchError):
        overlay.stream_with_tools(USER, [NOOP_TOOL])
    assert inner.calls == []


def test_catalog_record_without_tools_vetoes_tool_streaming() -> None:
    inner = StreamingToolsAdapter()
    overlay = CatalogCapabilityAdapter(
        inner, ModelRecord(model_id="fake-model", context_window_tokens=8_192, supports_tools=False)
    )
    assert overlay.supports_tool_streaming() is False
    with pytest.raises(LLMCapabilityMismatchError):
        overlay.stream_with_tools(USER, [NOOP_TOOL])
    assert inner.calls == []


def _routing(inner: Any) -> RoutingEvaluatingLLMAdapter:
    return RoutingEvaluatingLLMAdapter(
        profile_source=SimpleNamespace(llm_routing_profile=None),  # type: ignore[arg-type]
        inner=inner,
        context_provider=lambda: None,  # type: ignore[arg-type,return-value]
        adapter_factory=lambda *_: inner,
        evaluator=SimpleNamespace(),  # type: ignore[arg-type]
    )


def test_routing_wrapper_denies_and_forwards_tool_streaming_by_declaration() -> None:
    denied_inner = StreamingOnlyAdapter()
    denied = _routing(denied_inner)
    assert denied.supports_tool_streaming() is False
    with pytest.raises(LLMCapabilityMismatchError):
        denied.stream_with_tools(USER, [NOOP_TOOL])
    assert denied_inner.calls == []

    ok_inner = StreamingToolsAdapter()
    allowed = _routing(ok_inner)
    assert allowed.supports_tool_streaming() is True
    assert list(allowed.stream_with_tools(USER, [NOOP_TOOL]))[-1].is_final
    assert ok_inner.calls == ["stream_with_tools"]
