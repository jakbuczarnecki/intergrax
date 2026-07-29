# © Artur Czarnecki. All rights reserved.

"""Synthetic third-party Token Optimization plugin fixture for TOKEN-8D contract proof."""

from __future__ import annotations

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    StrategySafetyClass,
    TokenOptimizationBypassReason,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerDescriptor,
    TokenOptimizationLayerRequest,
    TokenOptimizationLayerResult,
    TokenOptimizationMechanism,
    TokenOptimizationPluginCapability,
    TokenOptimizationPluginDescriptor,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
)

FAKE_PLUGIN_ID = "synthetic.third_party.trace_filter"
FAKE_PLUGIN_NAME = "Synthetic Third-party Trace Filter"
FAKE_PLUGIN_VERSION = "1.0.0"

FAKE_LAYER_ID = "third_party.synthetic.trace_filter"
FAKE_STRATEGY_ID = "synthetic.third_party.trace_filter.strategy"

MISSING_REQUIRED_PLUGIN_ID = "synthetic.third_party.missing"
MISSING_REQUIRED_LAYER_ID = "third_party.synthetic.missing_required"
MISSING_REQUIRED_VERSION = "9.9.9"

WRONG_RESULT_LAYER_ID = "third_party.synthetic.wrong_result_id"

_TRACE_NOISE_PREFIX = "TRACE-NOISE:"
_EXCEPTION_SENTINEL = "SYNTHETIC-PLUGIN-SECRET-MESSAGE-MUST-NOT-LEAK"

PROTECTED_SYNTHETIC_PLUGIN_VALUE = "PROTECTED-SYNTH-THIRD-PARTY-PLUGIN-7788"

FAKE_STRATEGY = TokenOptimizationStrategyRef(
    strategy_id=FAKE_STRATEGY_ID,
    mechanism=TokenOptimizationMechanism.TERMINAL_LOG_FILTERING,
    kind=TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING,
    safety_class=StrategySafetyClass.LOSSY,
    plugin_id=FAKE_PLUGIN_ID,
    version=FAKE_PLUGIN_VERSION,
)

FAKE_PLUGIN_DESCRIPTOR = TokenOptimizationPluginDescriptor(
    plugin_id=FAKE_PLUGIN_ID,
    name=FAKE_PLUGIN_NAME,
    version=FAKE_PLUGIN_VERSION,
    capabilities=(
        TokenOptimizationPluginCapability(
            mechanism=TokenOptimizationMechanism.TERMINAL_LOG_FILTERING,
            strategy_kind=TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING,
            source_types=(TokenOptimizationSourceType.TOOL_OUTPUT,),
            lossless=False,
            lossy=True,
            reversible=False,
            requires_validation=True,
        ),
    ),
)

FAKE_LAYER_DESCRIPTOR = TokenOptimizationLayerDescriptor(
    layer_id=FAKE_LAYER_ID,
    name="Synthetic Third-party Trace Filter Layer",
    version=FAKE_PLUGIN_VERSION,
    strategy=FAKE_STRATEGY,
    supported_source_types=(TokenOptimizationSourceType.TOOL_OUTPUT,),
    safety_class=StrategySafetyClass.LOSSY,
    plugin_id=FAKE_PLUGIN_ID,
    built_in=False,
    requires_validation=True,
)


def standard_noisy_tool_output() -> str:
    return "\n".join(
        (
            "TRACE-NOISE: synthetic step 1",
            "TRACE-NOISE: synthetic step 2",
            "ERROR: synthetic compilation failure",
            "TRACE-NOISE: synthetic cleanup",
        )
    )


def expected_filtered_tool_output() -> str:
    return "ERROR: synthetic compilation failure"


def protected_region_tool_output() -> str:
    return "\n".join(
        (
            f"TRACE-NOISE: {PROTECTED_SYNTHETIC_PLUGIN_VALUE}",
            "ERROR: synthetic compilation failure",
        )
    )


def build_tool_output_request(
    content: str,
    *,
    policy: TokenOptimizationPolicy | None = None,
    protected_regions: tuple[ProtectedRegion, ...] = (),
) -> TokenOptimizationRequest:
    return TokenOptimizationRequest(
        content=content,
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        policy=policy
        or TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.BALANCED,
            allow_lossy=True,
            require_validation=True,
        ),
        protected_regions=protected_regions,
        metadata={"fixture": "third_party_plugin"},
    )


def build_unsupported_source_request(content: str | None = None) -> TokenOptimizationRequest:
    return TokenOptimizationRequest(
        content=content or standard_noisy_tool_output(),
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.BALANCED,
            allow_lossy=True,
            require_validation=True,
        ),
        metadata={"fixture": "third_party_plugin"},
    )


def build_protected_region_request() -> TokenOptimizationRequest:
    return build_tool_output_request(
        protected_region_tool_output(),
        protected_regions=(
            ProtectedRegion(
                kind=ProtectedRegionKind.IDENTIFIER,
                value=PROTECTED_SYNTHETIC_PLUGIN_VALUE,
            ),
        ),
    )


def _filter_trace_noise_lines(content: str) -> tuple[str, int]:
    lines = content.splitlines()
    kept: list[str] = []
    removed_line_count = 0
    for line in lines:
        if line.startswith(_TRACE_NOISE_PREFIX):
            removed_line_count += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed_line_count


class FakeThirdPartyTraceFilterLayer:
    """Canonical synthetic third-party optimization layer."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        return FAKE_LAYER_DESCRIPTOR

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        self.call_count += 1
        optimized_content, removed_line_count = _filter_trace_noise_lines(
            request.current_content
        )
        if removed_line_count == 0:
            return TokenOptimizationLayerResult(
                layer_id=FAKE_LAYER_ID,
                output_content=request.current_content,
                decision=TokenOptimizationLayerDecision.BYPASS,
                bypass_reason=TokenOptimizationBypassReason.NO_SAVINGS,
                strategy=FAKE_STRATEGY,
            )
        return TokenOptimizationLayerResult(
            layer_id=FAKE_LAYER_ID,
            output_content=optimized_content,
            decision=TokenOptimizationLayerDecision.APPLY,
            strategy=FAKE_STRATEGY,
            fallback_used=False,
            metadata={
                "synthetic_plugin": True,
                "removed_line_count": removed_line_count,
            },
        )


class InvalidResultTypeThirdPartyLayer:
    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        return FAKE_LAYER_DESCRIPTOR

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        return {"content": request.current_content}  # type: ignore[return-value]


class MismatchedResultLayerIdThirdPartyLayer:
    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        return FAKE_LAYER_DESCRIPTOR

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        return TokenOptimizationLayerResult(
            layer_id=WRONG_RESULT_LAYER_ID,
            output_content=request.current_content,
            decision=TokenOptimizationLayerDecision.APPLY,
            strategy=FAKE_STRATEGY,
        )


class ExceptionThrowingThirdPartyLayer:
    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        return FAKE_LAYER_DESCRIPTOR

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        raise RuntimeError(_EXCEPTION_SENTINEL)
