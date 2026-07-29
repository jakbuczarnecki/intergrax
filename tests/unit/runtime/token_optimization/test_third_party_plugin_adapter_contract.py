# © Artur Czarnecki. All rights reserved.

"""TOKEN-8D: third-party plugin adapter contract proof."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from intergrax.runtime.token_optimization.builtin_catalog import (
    create_builtin_token_optimization_layer_catalog,
)
from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenOptimizationBypassReason,
    TokenOptimizationLayer,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerDescriptor,
    TokenOptimizationLayerRef,
    TokenOptimizationPipelineConfig,
    TokenOptimizationPipelineMode,
    TokenOptimizationPipelineResult,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.pipeline import TokenOptimizationPipelineRunner
from intergrax.runtime.token_optimization.registry import TokenOptimizationLayerRegistry
from tests.fixtures.token_optimization import fake_third_party_plugin as fixture

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BUILTIN_LAYER_IDS = (
    "builtin.exact_deduplication",
    "builtin.extractive_filtering",
    "builtin.budget_aware_context_packing",
)

_FORBIDDEN_METADATA_SUBSTRINGS = (
    fixture.standard_noisy_tool_output(),
    fixture.expected_filtered_tool_output(),
    fixture.protected_region_tool_output(),
    fixture.PROTECTED_SYNTHETIC_PLUGIN_VALUE,
    "SYNTHETIC-PLUGIN-SECRET-MESSAGE-MUST-NOT-LEAK",
    "Traceback (most recent call last)",
    "TRACE-NOISE: synthetic step 1",
    "TRACE-NOISE: synthetic step 2",
    "TRACE-NOISE: synthetic cleanup",
)


def _canonical_layer_ref(**overrides: object) -> TokenOptimizationLayerRef:
    base: dict[str, object] = {
        "layer_id": fixture.FAKE_LAYER_ID,
        "plugin_id": fixture.FAKE_PLUGIN_ID,
        "version": fixture.FAKE_PLUGIN_VERSION,
    }
    base.update(overrides)
    return TokenOptimizationLayerRef(**base)  # type: ignore[arg-type]


def _run_pipeline(
    *,
    layers: tuple[TokenOptimizationLayer, ...],
    layer_refs: tuple[TokenOptimizationLayerRef, ...],
    request: TokenOptimizationRequest,
    pipeline_id: str,
) -> TokenOptimizationPipelineResult:
    registry = TokenOptimizationLayerRegistry(layers)
    runner = TokenOptimizationPipelineRunner(registry=registry)
    config = TokenOptimizationPipelineConfig(
        pipeline_id=pipeline_id,
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=layer_refs,
    )
    return runner.run(request=request, config=config)


def _collect_mapping_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        collected: list[str] = []
        for nested in value.values():
            collected.extend(_collect_mapping_values(nested))
        return tuple(collected)
    if isinstance(value, (list, tuple)):
        collected = []
        for nested in value:
            collected.extend(_collect_mapping_values(nested))
        return tuple(collected)
    return ()


def _assert_metadata_is_raw_content_safe(
    result: TokenOptimizationPipelineResult,
    *,
    original_content: str,
    final_content: str,
) -> None:
    metadata_blobs = (
        result.receipt_metadata,
        result.metadata,
    )
    for layer_result in result.layer_results:
        metadata_blobs = (
            *metadata_blobs,
            layer_result.metadata,
            layer_result.receipt_metadata,
        )

    serialized = json.dumps(metadata_blobs)
    assert original_content not in serialized
    assert final_content not in serialized
    for forbidden in _FORBIDDEN_METADATA_SUBSTRINGS:
        assert forbidden not in serialized
    for value in _collect_mapping_values(metadata_blobs):
        if value.startswith("{") and "content" in value:
            pytest.fail("arbitrary plugin object representation leaked into metadata")


# --- Test group A: descriptor contract ---


def test_plugin_descriptor_uses_canonical_plugin_id() -> None:
    assert fixture.FAKE_PLUGIN_DESCRIPTOR.plugin_id == fixture.FAKE_PLUGIN_ID


def test_plugin_descriptor_version_is_1_0_0() -> None:
    assert fixture.FAKE_PLUGIN_DESCRIPTOR.version == "1.0.0"


def test_plugin_descriptor_declares_exactly_one_capability() -> None:
    assert len(fixture.FAKE_PLUGIN_DESCRIPTOR.capabilities) == 1


def test_layer_descriptor_is_not_built_in() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    assert layer.descriptor.built_in is False


def test_layer_descriptor_has_canonical_plugin_id() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    assert layer.descriptor.plugin_id == fixture.FAKE_PLUGIN_ID


def test_layer_descriptor_version_matches_plugin_version() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    assert layer.descriptor.version == fixture.FAKE_PLUGIN_VERSION
    assert layer.descriptor.version == fixture.FAKE_PLUGIN_DESCRIPTOR.version


def test_layer_strategy_contains_plugin_id_and_version() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    strategy = layer.descriptor.strategy
    assert strategy.plugin_id == fixture.FAKE_PLUGIN_ID
    assert strategy.version == fixture.FAKE_PLUGIN_VERSION


def test_capability_and_layer_descriptor_are_consistent() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    descriptor = layer.descriptor
    capability = fixture.FAKE_PLUGIN_DESCRIPTOR.capabilities[0]
    assert capability.mechanism == descriptor.strategy.mechanism
    assert capability.strategy_kind == descriptor.strategy.kind
    assert capability.source_types == descriptor.supported_source_types
    assert capability.lossy is True
    assert descriptor.safety_class is StrategySafetyClass.LOSSY
    assert capability.requires_validation == descriptor.requires_validation


def test_layer_exposes_callable_optimize() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    assert callable(layer.optimize)


def test_layer_exposes_descriptor() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    assert isinstance(layer.descriptor, TokenOptimizationLayerDescriptor)
    assert layer.descriptor.layer_id == fixture.FAKE_LAYER_ID


# --- Test group B: plugin absent from built-in catalog ---


def test_fake_layer_absent_from_builtin_catalog() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    assert fixture.FAKE_LAYER_ID not in catalog.layer_ids


def test_builtin_catalog_rejects_fake_layer_id() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    with pytest.raises(ValueError, match="unknown built-in layer_id"):
        catalog.create(fixture.FAKE_LAYER_ID)


# --- Test group C: explicit registry registration ---


def test_explicit_registry_registration() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    registry = TokenOptimizationLayerRegistry((layer,))
    assert registry.layer_ids == (fixture.FAKE_LAYER_ID,)
    descriptor = registry.descriptors[0]
    assert descriptor.plugin_id == fixture.FAKE_PLUGIN_ID
    assert descriptor.built_in is False
    assert descriptor.version == fixture.FAKE_PLUGIN_VERSION


# --- Test group D: exact plugin and version resolution ---


def test_exact_plugin_and_version_resolution() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    registry = TokenOptimizationLayerRegistry((layer,))
    layer_ref = _canonical_layer_ref()
    assert registry.resolve(layer_ref) is layer


# --- Test group E: wrong plugin ID rejection ---


def test_wrong_plugin_id_resolution_returns_none() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    registry = TokenOptimizationLayerRegistry((layer,))
    layer_ref = _canonical_layer_ref(plugin_id="synthetic.third_party.other")
    assert registry.resolve(layer_ref) is None


def test_wrong_plugin_id_pipeline_bypasses_without_execution() -> None:
    content = fixture.standard_noisy_tool_output()
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    result = _run_pipeline(
        layers=(layer,),
        layer_refs=(_canonical_layer_ref(plugin_id="synthetic.third_party.other"),),
        request=fixture.build_tool_output_request(content),
        pipeline_id="wrong-plugin-id",
    )
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.BYPASS
    assert (
        result.layer_results[0].bypass_reason
        is TokenOptimizationBypassReason.PLUGIN_UNAVAILABLE
    )
    assert result.receipt_metadata["executed_layer_ids"] == []
    assert result.applied_layer_ids == ()
    assert result.final_content == content
    assert result.receipt_metadata["completed"] is True


# --- Test group F: wrong version rejection ---


def test_wrong_version_resolution_returns_none() -> None:
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    registry = TokenOptimizationLayerRegistry((layer,))
    layer_ref = _canonical_layer_ref(version="2.0.0")
    assert registry.resolve(layer_ref) is None


def test_wrong_version_pipeline_bypasses_without_execution() -> None:
    content = fixture.standard_noisy_tool_output()
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    result = _run_pipeline(
        layers=(layer,),
        layer_refs=(_canonical_layer_ref(version="2.0.0"),),
        request=fixture.build_tool_output_request(content),
        pipeline_id="wrong-version",
    )
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.BYPASS
    assert (
        result.layer_results[0].bypass_reason
        is TokenOptimizationBypassReason.PLUGIN_UNAVAILABLE
    )
    assert result.receipt_metadata["executed_layer_ids"] == []
    assert result.applied_layer_ids == ()
    assert result.final_content == content
    assert result.receipt_metadata["completed"] is True


# --- Test group G: successful plugin execution ---


def test_successful_plugin_only_replace_pipeline() -> None:
    content = fixture.standard_noisy_tool_output()
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    result = _run_pipeline(
        layers=(layer,),
        layer_refs=(_canonical_layer_ref(),),
        request=fixture.build_tool_output_request(content),
        pipeline_id="plugin-only-success",
    )
    assert result.receipt_metadata["resolved_layer_ids"] == [fixture.FAKE_LAYER_ID]
    assert result.receipt_metadata["executed_layer_ids"] == [fixture.FAKE_LAYER_ID]
    assert result.applied_layer_ids == (fixture.FAKE_LAYER_ID,)
    assert result.bypassed_layer_ids == ()
    assert result.failed_layer_ids == ()
    assert result.fallback_used is False
    assert result.receipt_metadata["completed"] is True
    assert result.final_content == fixture.expected_filtered_tool_output()
    for layer_id in _BUILTIN_LAYER_IDS:
        assert layer_id not in result.receipt_metadata["resolved_layer_ids"]
        assert layer_id not in result.receipt_metadata["executed_layer_ids"]
        assert layer_id not in result.applied_layer_ids


# --- Test group H: lossy policy enforcement ---


def test_lossy_policy_blocks_plugin_before_execution() -> None:
    content = fixture.standard_noisy_tool_output()
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    policy = TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
        allow_lossy=False,
    )
    result = _run_pipeline(
        layers=(layer,),
        layer_refs=(_canonical_layer_ref(),),
        request=fixture.build_tool_output_request(content, policy=policy),
        pipeline_id="lossy-blocked",
    )
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.BYPASS
    assert (
        result.layer_results[0].bypass_reason
        is TokenOptimizationBypassReason.POLICY_DISALLOWED
    )
    assert result.receipt_metadata["executed_layer_ids"] == []
    assert result.applied_layer_ids == ()
    assert result.final_content == content
    assert layer.call_count == 0


# --- Test group I: source-type enforcement ---


def test_unsupported_source_type_blocks_plugin_before_execution() -> None:
    content = fixture.standard_noisy_tool_output()
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    result = _run_pipeline(
        layers=(layer,),
        layer_refs=(_canonical_layer_ref(),),
        request=fixture.build_unsupported_source_request(content),
        pipeline_id="unsupported-source",
    )
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.BYPASS
    assert (
        result.layer_results[0].bypass_reason
        is TokenOptimizationBypassReason.UNSUPPORTED_SOURCE_TYPE
    )
    assert result.receipt_metadata["executed_layer_ids"] == []
    assert result.failed_layer_ids == ()
    assert result.final_content == content


# --- Test group J: protected-region fallback ---


def test_protected_region_fallback_uses_central_validation() -> None:
    request = fixture.build_protected_region_request()
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    result = _run_pipeline(
        layers=(layer,),
        layer_refs=(_canonical_layer_ref(),),
        request=request,
        pipeline_id="protected-fallback",
    )
    assert result.receipt_metadata["executed_layer_ids"] == [fixture.FAKE_LAYER_ID]
    assert result.applied_layer_ids == ()
    assert result.bypassed_layer_ids == (fixture.FAKE_LAYER_ID,)
    assert result.failed_layer_ids == ()
    assert result.fallback_used is True
    assert result.final_content == request.content
    assert fixture.PROTECTED_SYNTHETIC_PLUGIN_VALUE in result.final_content
    layer_result = result.layer_results[0]
    assert layer_result.decision is TokenOptimizationLayerDecision.FALLBACK
    assert layer_result.bypass_reason is TokenOptimizationBypassReason.VALIDATION_FAILED
    assert layer_result.validation is not None
    assert layer_result.validation.status is ProtectedRegionValidationStatus.FAILED


def test_canonical_plugin_contains_no_protected_region_logic() -> None:
    source = inspect.getsource(fixture.FakeThirdPartyTraceFilterLayer.optimize)
    assert "ProtectedRegion" not in source
    assert "protected_region" not in source


# --- Test group K: invalid result type rejection ---


def test_invalid_result_type_rejected_by_runner() -> None:
    content = fixture.standard_noisy_tool_output()
    result = _run_pipeline(
        layers=(fixture.InvalidResultTypeThirdPartyLayer(),),
        layer_refs=(_canonical_layer_ref(),),
        request=fixture.build_tool_output_request(content),
        pipeline_id="invalid-result-type",
    )
    assert result.receipt_metadata["executed_layer_ids"] == [fixture.FAKE_LAYER_ID]
    assert result.failed_layer_ids == (fixture.FAKE_LAYER_ID,)
    assert result.applied_layer_ids == ()
    assert result.final_content == content
    assert result.receipt_metadata["completed"] is True
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.FAILED
    assert result.layer_results[0].metadata["failure_kind"] == "invalid_result_type"


# --- Test group L: mismatched result layer ID rejection ---


def test_mismatched_result_layer_id_rejected_by_runner() -> None:
    content = fixture.standard_noisy_tool_output()
    result = _run_pipeline(
        layers=(fixture.MismatchedResultLayerIdThirdPartyLayer(),),
        layer_refs=(_canonical_layer_ref(),),
        request=fixture.build_tool_output_request(content),
        pipeline_id="mismatched-layer-id",
    )
    assert result.receipt_metadata["executed_layer_ids"] == [fixture.FAKE_LAYER_ID]
    assert result.failed_layer_ids == (fixture.FAKE_LAYER_ID,)
    assert result.final_content == content
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.FAILED
    assert result.layer_results[0].metadata["failure_kind"] == "layer_id_mismatch"


# --- Test group M: plugin exception containment ---


def test_plugin_exception_is_contained() -> None:
    content = fixture.standard_noisy_tool_output()
    result = _run_pipeline(
        layers=(fixture.ExceptionThrowingThirdPartyLayer(),),
        layer_refs=(_canonical_layer_ref(),),
        request=fixture.build_tool_output_request(content),
        pipeline_id="plugin-exception",
    )
    assert result.receipt_metadata["executed_layer_ids"] == [fixture.FAKE_LAYER_ID]
    assert result.failed_layer_ids == (fixture.FAKE_LAYER_ID,)
    assert result.applied_layer_ids == ()
    assert result.final_content == content
    assert result.receipt_metadata["completed"] is True
    assert result.layer_results[0].metadata["failure_kind"] == "layer_exception"
    assert result.layer_results[0].metadata["exception_type"] == "RuntimeError"
    serialized = json.dumps(
        {
            "receipt_metadata": result.receipt_metadata,
            "metadata": result.metadata,
            "layer_metadata": result.layer_results[0].metadata,
            "layer_receipt_metadata": result.layer_results[0].receipt_metadata,
        }
    )
    assert "SYNTHETIC-PLUGIN-SECRET-MESSAGE-MUST-NOT-LEAK" not in serialized
    assert "Traceback (most recent call last)" not in serialized
    assert content not in serialized
    assert fixture.PROTECTED_SYNTHETIC_PLUGIN_VALUE not in serialized


# --- Test group N: required unavailable plugin rollback ---


def test_required_unavailable_plugin_rolls_back_previous_application() -> None:
    content = fixture.standard_noisy_tool_output()
    layer = fixture.FakeThirdPartyTraceFilterLayer()
    missing_ref = TokenOptimizationLayerRef(
        layer_id=fixture.MISSING_REQUIRED_LAYER_ID,
        plugin_id=fixture.MISSING_REQUIRED_PLUGIN_ID,
        version=fixture.MISSING_REQUIRED_VERSION,
        required=True,
    )
    result = _run_pipeline(
        layers=(layer,),
        layer_refs=(_canonical_layer_ref(), missing_ref),
        request=fixture.build_tool_output_request(content),
        pipeline_id="required-unavailable-rollback",
    )
    assert result.receipt_metadata["executed_layer_ids"] == [fixture.FAKE_LAYER_ID]
    assert result.failed_layer_ids == (fixture.MISSING_REQUIRED_LAYER_ID,)
    assert result.applied_layer_ids == ()
    assert result.fallback_used is True
    assert result.receipt_metadata["completed"] is False
    assert (
        result.receipt_metadata["required_failure_layer_id"]
        == fixture.MISSING_REQUIRED_LAYER_ID
    )
    assert result.final_content == content
    missing_result = result.layer_results[1]
    assert missing_result.decision is TokenOptimizationLayerDecision.FAILED
    assert missing_result.bypass_reason is TokenOptimizationBypassReason.PLUGIN_UNAVAILABLE


# --- Test group O: required malformed plugin result ---


def test_required_malformed_plugin_result_rolls_back() -> None:
    content = fixture.standard_noisy_tool_output()
    result = _run_pipeline(
        layers=(fixture.InvalidResultTypeThirdPartyLayer(),),
        layer_refs=(_canonical_layer_ref(required=True),),
        request=fixture.build_tool_output_request(content),
        pipeline_id="required-malformed",
    )
    assert result.receipt_metadata["completed"] is False
    assert result.fallback_used is True
    assert result.receipt_metadata["required_failure_layer_id"] == fixture.FAKE_LAYER_ID
    assert result.final_content == content
    assert result.applied_layer_ids == ()


# --- Test group P: safe receipts and metadata ---


@pytest.mark.parametrize(
    ("pipeline_id", "layers", "layer_refs", "request_factory"),
    [
        pytest.param(
            "safe-success",
            (fixture.FakeThirdPartyTraceFilterLayer(),),
            (_canonical_layer_ref(),),
            fixture.build_tool_output_request,
            id="success",
        ),
        pytest.param(
            "safe-protected",
            (fixture.FakeThirdPartyTraceFilterLayer(),),
            (_canonical_layer_ref(),),
            lambda: fixture.build_protected_region_request(),
            id="protected_fallback",
        ),
        pytest.param(
            "safe-malformed",
            (fixture.InvalidResultTypeThirdPartyLayer(),),
            (_canonical_layer_ref(),),
            fixture.build_tool_output_request,
            id="malformed_result",
        ),
        pytest.param(
            "safe-unavailable",
            (fixture.FakeThirdPartyTraceFilterLayer(),),
            (_canonical_layer_ref(plugin_id="synthetic.third_party.other"),),
            fixture.build_tool_output_request,
            id="unavailable_plugin",
        ),
        pytest.param(
            "safe-exception",
            (fixture.ExceptionThrowingThirdPartyLayer(),),
            (_canonical_layer_ref(),),
            fixture.build_tool_output_request,
            id="exception",
        ),
    ],
)
def test_receipts_and_metadata_remain_raw_content_safe(
    pipeline_id: str,
    layers: tuple[TokenOptimizationLayer, ...],
    layer_refs: tuple[TokenOptimizationLayerRef, ...],
    request_factory: object,
) -> None:
    content = fixture.standard_noisy_tool_output()
    if pipeline_id == "safe-protected":
        request = fixture.build_protected_region_request()
    else:
        request = fixture.build_tool_output_request(content)  # type: ignore[operator]
    result = _run_pipeline(
        layers=layers,
        layer_refs=layer_refs,
        request=request,
        pipeline_id=pipeline_id,
    )
    _assert_metadata_is_raw_content_safe(
        result,
        original_content=request.content,
        final_content=result.final_content,
    )


# --- Test group Q: built-ins are not involved ---


def test_successful_pipeline_does_not_execute_builtin_layers() -> None:
    content = fixture.standard_noisy_tool_output()
    result = _run_pipeline(
        layers=(fixture.FakeThirdPartyTraceFilterLayer(),),
        layer_refs=(_canonical_layer_ref(),),
        request=fixture.build_tool_output_request(content),
        pipeline_id="no-builtins",
    )
    for layer_id in _BUILTIN_LAYER_IDS:
        assert layer_id not in result.receipt_metadata["resolved_layer_ids"]
        assert layer_id not in result.receipt_metadata["executed_layer_ids"]
        assert layer_id not in result.applied_layer_ids


# --- Test group R: no dynamic loading ---


def test_fixture_and_tests_do_not_use_dynamic_loading() -> None:
    from pathlib import Path

    forbidden = (
        "import" + "lib",
        "entry_" + "points",
        "pkg_" + "resources",
        "sub" + "process",
        "pip " + "install",
        "Plugin" + "Manager",
        "plugin " + "scanning",
    )
    paths = (
        Path(fixture.__file__),
        Path(__file__),
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in source
