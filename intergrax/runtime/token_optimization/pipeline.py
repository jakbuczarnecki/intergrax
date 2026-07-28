# © Artur Czarnecki. All rights reserved.

"""Deterministic pipeline config resolution and sequential layer runner (TOKEN-8A)."""

from __future__ import annotations

import dataclasses
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenOptimizationBypassReason,
    TokenOptimizationLayerContext,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerRef,
    TokenOptimizationLayerRequest,
    TokenOptimizationLayerResult,
    TokenOptimizationPipelineConfig,
    TokenOptimizationPipelineMode,
    TokenOptimizationPipelineResult,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
)
from intergrax.runtime.token_optimization.protected_regions import validate_protected_regions
from intergrax.runtime.token_optimization.registry import TokenOptimizationLayerRegistry

_PIPELINE_METADATA_KEY = "token_optimization_pipeline"

_MEASURE_ONLY_SAFETY_CLASSES = frozenset(
    {
        StrategySafetyClass.MEASUREMENT_ONLY,
        StrategySafetyClass.POLICY_ONLY,
    }
)

_MUTATING_DECISIONS = frozenset(
    {
        TokenOptimizationLayerDecision.APPLY,
        TokenOptimizationLayerDecision.OVERRIDE_PREVIOUS,
    }
)


def resolve_token_optimization_pipeline_layers(
    *,
    default_layers: tuple[TokenOptimizationLayerRef, ...],
    config: TokenOptimizationPipelineConfig,
) -> tuple[TokenOptimizationLayerRef, ...]:
    """Resolve the ordered layer list from defaults and pipeline config."""
    if config.mode is TokenOptimizationPipelineMode.REPLACE:
        composed = list(config.layers)
    else:
        if config.allow_repeated_layers:
            composed = list(default_layers) + list(config.layers)
        else:
            composed = list(default_layers)
            for layer_ref in config.layers:
                replaced = False
                for index, existing in enumerate(composed):
                    if existing.layer_id == layer_ref.layer_id:
                        composed[index] = layer_ref
                        replaced = True
                        break
                if not replaced:
                    composed.append(layer_ref)

    indexed = list(enumerate(composed))
    indexed.sort(
        key=lambda item: (
            item[1].order if item[1].order is not None else item[0],
            item[0],
        )
    )
    return tuple(layer_ref for _, layer_ref in indexed)


def _validate_unique_default_layers(
    default_layers: tuple[TokenOptimizationLayerRef, ...],
) -> None:
    layer_ids = [layer_ref.layer_id for layer_ref in default_layers]
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError("default_layers cannot contain duplicate layer_id values")


def _effective_safety_class(descriptor) -> StrategySafetyClass:
    if descriptor.safety_class is not None:
        return descriptor.safety_class
    return descriptor.strategy.safety_class


def _policy_globally_disabled(policy: TokenOptimizationPolicy) -> bool:
    return not policy.enabled or policy.profile is TokenOptimizationProfile.OFF


def _build_layer_metadata(
    *,
    request: TokenOptimizationRequest,
    config: TokenOptimizationPipelineConfig,
    layer_ref: TokenOptimizationLayerRef,
) -> dict[str, Any]:
    metadata = dict(request.metadata)
    metadata[_PIPELINE_METADATA_KEY] = {
        "pipeline_id": config.pipeline_id,
        "layer_id": layer_ref.layer_id,
        "settings": dict(layer_ref.settings),
    }
    return metadata


def _synthetic_layer_result(
    *,
    layer_id: str,
    output_content: str,
    decision: TokenOptimizationLayerDecision,
    bypass_reason: TokenOptimizationBypassReason | None = None,
    fallback_used: bool = False,
    metadata: dict[str, Any] | None = None,
    validation=None,
) -> TokenOptimizationLayerResult:
    return TokenOptimizationLayerResult(
        layer_id=layer_id,
        output_content=output_content,
        decision=decision,
        bypass_reason=bypass_reason,
        fallback_used=fallback_used,
        validation=validation,
        metadata=metadata or {},
    )


class TokenOptimizationPipelineRunner:
    """Helper-only sequential executor for registered optimization layers."""

    def __init__(
        self,
        *,
        registry: TokenOptimizationLayerRegistry,
        default_layers: tuple[TokenOptimizationLayerRef, ...] = (),
    ) -> None:
        _validate_unique_default_layers(default_layers)
        self._registry = registry
        self._default_layers = default_layers

    def run(
        self,
        *,
        request: TokenOptimizationRequest,
        config: TokenOptimizationPipelineConfig,
    ) -> TokenOptimizationPipelineResult:
        resolved_layers = resolve_token_optimization_pipeline_layers(
            default_layers=self._default_layers,
            config=config,
        )
        original_content = request.content
        current_content = original_content
        layer_results: list[TokenOptimizationLayerResult] = []
        applied_layer_ids: list[str] = []
        bypassed_layer_ids: list[str] = []
        failed_layer_ids: list[str] = []
        executed_layer_ids: list[str] = []
        disabled_layer_ids: list[str] = []
        fallback_used = False
        required_failure_layer_id: str | None = None
        completed = True
        previous_processed_ids: list[str] = []
        global_disabled = _policy_globally_disabled(request.policy)
        measure_only = request.policy.profile is TokenOptimizationProfile.MEASURE_ONLY

        for layer_index, layer_ref in enumerate(resolved_layers):
            layer_input_content = current_content

            if not layer_ref.enabled:
                disabled_layer_ids.append(layer_ref.layer_id)
                layer_results.append(
                    _synthetic_layer_result(
                        layer_id=layer_ref.layer_id,
                        output_content=layer_input_content,
                        decision=TokenOptimizationLayerDecision.BYPASS,
                        bypass_reason=TokenOptimizationBypassReason.DISABLED,
                    )
                )
                bypassed_layer_ids.append(layer_ref.layer_id)
                previous_processed_ids.append(layer_ref.layer_id)
                continue

            if global_disabled:
                layer_results.append(
                    _synthetic_layer_result(
                        layer_id=layer_ref.layer_id,
                        output_content=layer_input_content,
                        decision=TokenOptimizationLayerDecision.BYPASS,
                        bypass_reason=TokenOptimizationBypassReason.DISABLED,
                    )
                )
                bypassed_layer_ids.append(layer_ref.layer_id)
                previous_processed_ids.append(layer_ref.layer_id)
                continue

            layer = self._registry.resolve(layer_ref)
            if layer is None:
                if layer_ref.required:
                    layer_results.append(
                        _synthetic_layer_result(
                            layer_id=layer_ref.layer_id,
                            output_content=layer_input_content,
                            decision=TokenOptimizationLayerDecision.FAILED,
                            bypass_reason=TokenOptimizationBypassReason.PLUGIN_UNAVAILABLE,
                            fallback_used=True,
                        )
                    )
                    failed_layer_ids.append(layer_ref.layer_id)
                    required_failure_layer_id = layer_ref.layer_id
                    fallback_used = True
                    completed = False
                    current_content = original_content
                    previous_processed_ids.append(layer_ref.layer_id)
                    break

                layer_results.append(
                    _synthetic_layer_result(
                        layer_id=layer_ref.layer_id,
                        output_content=layer_input_content,
                        decision=TokenOptimizationLayerDecision.BYPASS,
                        bypass_reason=TokenOptimizationBypassReason.PLUGIN_UNAVAILABLE,
                    )
                )
                bypassed_layer_ids.append(layer_ref.layer_id)
                previous_processed_ids.append(layer_ref.layer_id)
                continue

            descriptor = layer.descriptor
            safety_class = _effective_safety_class(descriptor)

            policy_bypass_reason = _policy_gate_bypass_reason(
                policy=request.policy,
                safety_class=safety_class,
                measure_only=measure_only,
            )
            if policy_bypass_reason is not None:
                layer_results.append(
                    _synthetic_layer_result(
                        layer_id=layer_ref.layer_id,
                        output_content=layer_input_content,
                        decision=TokenOptimizationLayerDecision.BYPASS,
                        bypass_reason=policy_bypass_reason,
                    )
                )
                bypassed_layer_ids.append(layer_ref.layer_id)
                previous_processed_ids.append(layer_ref.layer_id)
                continue

            if (
                descriptor.supported_source_types
                and request.source_type not in descriptor.supported_source_types
            ):
                if layer_ref.required:
                    layer_results.append(
                        _synthetic_layer_result(
                            layer_id=layer_ref.layer_id,
                            output_content=layer_input_content,
                            decision=TokenOptimizationLayerDecision.FAILED,
                            bypass_reason=TokenOptimizationBypassReason.UNSUPPORTED_SOURCE_TYPE,
                            fallback_used=True,
                        )
                    )
                    failed_layer_ids.append(layer_ref.layer_id)
                    required_failure_layer_id = layer_ref.layer_id
                    fallback_used = True
                    completed = False
                    current_content = original_content
                    previous_processed_ids.append(layer_ref.layer_id)
                    break

                layer_results.append(
                    _synthetic_layer_result(
                        layer_id=layer_ref.layer_id,
                        output_content=layer_input_content,
                        decision=TokenOptimizationLayerDecision.BYPASS,
                        bypass_reason=TokenOptimizationBypassReason.UNSUPPORTED_SOURCE_TYPE,
                    )
                )
                bypassed_layer_ids.append(layer_ref.layer_id)
                previous_processed_ids.append(layer_ref.layer_id)
                continue

            layer_request = TokenOptimizationLayerRequest(
                original_content=original_content,
                current_content=current_content,
                source_type=request.source_type,
                policy=request.policy,
                attribution=request.attribution,
                strategy=descriptor.strategy,
                layer_context=TokenOptimizationLayerContext(
                    pipeline_id=config.pipeline_id,
                    layer_index=layer_index,
                    previous_layer_ids=tuple(previous_processed_ids),
                    applied_layer_ids=tuple(applied_layer_ids),
                ),
                metadata=_build_layer_metadata(
                    request=request,
                    config=config,
                    layer_ref=layer_ref,
                ),
            )

            try:
                raw_result = layer.optimize(layer_request)
            except Exception as exc:
                failure_metadata = {
                    "failure_kind": "layer_exception",
                    "exception_type": type(exc).__name__,
                }
                if layer_ref.required:
                    layer_results.append(
                        _synthetic_layer_result(
                            layer_id=layer_ref.layer_id,
                            output_content=layer_input_content,
                            decision=TokenOptimizationLayerDecision.FAILED,
                            fallback_used=True,
                            metadata=failure_metadata,
                        )
                    )
                    failed_layer_ids.append(layer_ref.layer_id)
                    required_failure_layer_id = layer_ref.layer_id
                    fallback_used = True
                    completed = False
                    current_content = original_content
                    previous_processed_ids.append(layer_ref.layer_id)
                    break

                layer_results.append(
                    _synthetic_layer_result(
                        layer_id=layer_ref.layer_id,
                        output_content=layer_input_content,
                        decision=TokenOptimizationLayerDecision.FAILED,
                        metadata=failure_metadata,
                    )
                )
                failed_layer_ids.append(layer_ref.layer_id)
                executed_layer_ids.append(layer_ref.layer_id)
                previous_processed_ids.append(layer_ref.layer_id)
                continue

            executed_layer_ids.append(layer_ref.layer_id)

            malformed = _malformed_layer_result(
                raw_result=raw_result,
                expected_layer_id=layer_ref.layer_id,
            )
            if malformed is not None:
                if layer_ref.required:
                    layer_results.append(malformed)
                    failed_layer_ids.append(layer_ref.layer_id)
                    required_failure_layer_id = layer_ref.layer_id
                    fallback_used = True
                    completed = False
                    current_content = original_content
                    previous_processed_ids.append(layer_ref.layer_id)
                    break

                layer_results.append(malformed)
                failed_layer_ids.append(layer_ref.layer_id)
                previous_processed_ids.append(layer_ref.layer_id)
                continue

            result = raw_result

            if result.decision is TokenOptimizationLayerDecision.FAILED:
                failed_layer_ids.append(layer_ref.layer_id)
                layer_results.append(result)
                previous_processed_ids.append(layer_ref.layer_id)
                if layer_ref.required:
                    fallback_used = True
                    completed = False
                    current_content = original_content
                    required_failure_layer_id = layer_ref.layer_id
                    break
                continue

            if result.decision in _MUTATING_DECISIONS and _requires_validation(
                policy=request.policy,
                descriptor=descriptor,
            ):
                validation = validate_protected_regions(
                    original_content,
                    result.output_content,
                    regions=request.protected_regions or None,
                )
                if validation.status is ProtectedRegionValidationStatus.FAILED:
                    result = _synthetic_layer_result(
                        layer_id=layer_ref.layer_id,
                        output_content=layer_input_content,
                        decision=TokenOptimizationLayerDecision.FALLBACK,
                        bypass_reason=TokenOptimizationBypassReason.VALIDATION_FAILED,
                        fallback_used=True,
                        validation=validation,
                    )
                    fallback_used = True
                    bypassed_layer_ids.append(layer_ref.layer_id)
                    layer_results.append(result)
                    previous_processed_ids.append(layer_ref.layer_id)
                    continue
                result = dataclasses.replace(result, validation=validation)

            outcome = _apply_layer_decision(
                result=result,
                layer_input_content=layer_input_content,
                original_content=original_content,
                applied_layer_ids=applied_layer_ids,
                measure_only=measure_only,
            )
            if outcome.malformed is not None:
                if layer_ref.required:
                    layer_results.append(outcome.malformed)
                    failed_layer_ids.append(layer_ref.layer_id)
                    required_failure_layer_id = layer_ref.layer_id
                    fallback_used = True
                    completed = False
                    current_content = original_content
                    previous_processed_ids.append(layer_ref.layer_id)
                    break

                layer_results.append(outcome.malformed)
                failed_layer_ids.append(layer_ref.layer_id)
                previous_processed_ids.append(layer_ref.layer_id)
                continue

            result = outcome.result
            current_content = outcome.current_content
            applied_layer_ids[:] = outcome.applied_layer_ids
            bypassed_layer_ids.extend(outcome.bypassed_layer_ids)
            failed_layer_ids.extend(outcome.failed_layer_ids)
            if outcome.fallback_used:
                fallback_used = True

            layer_results.append(result)
            previous_processed_ids.append(layer_ref.layer_id)

        receipt_metadata: dict[str, Any] = {
            "pipeline_mode": config.mode.value,
            "resolved_layer_ids": [layer_ref.layer_id for layer_ref in resolved_layers],
            "executed_layer_ids": executed_layer_ids,
            "disabled_layer_ids": disabled_layer_ids,
            "completed": completed,
        }
        if required_failure_layer_id is not None:
            receipt_metadata["required_failure_layer_id"] = required_failure_layer_id

        return TokenOptimizationPipelineResult(
            pipeline_id=config.pipeline_id,
            original_content=original_content,
            final_content=current_content,
            layer_results=tuple(layer_results),
            applied_layer_ids=tuple(applied_layer_ids),
            bypassed_layer_ids=tuple(bypassed_layer_ids),
            failed_layer_ids=tuple(failed_layer_ids),
            fallback_used=fallback_used,
            aggregate_measurement=None,
            receipt_metadata=receipt_metadata,
            metadata={
                "resolved_layer_count": len(resolved_layers),
                "executed_layer_count": len(executed_layer_ids),
                "applied_layer_count": len(applied_layer_ids),
                "bypassed_layer_count": len(bypassed_layer_ids),
                "failed_layer_count": len(failed_layer_ids),
            },
        )


def _policy_gate_bypass_reason(
    *,
    policy: TokenOptimizationPolicy,
    safety_class: StrategySafetyClass,
    measure_only: bool,
) -> TokenOptimizationBypassReason | None:
    if measure_only and safety_class not in _MEASURE_ONLY_SAFETY_CLASSES:
        return TokenOptimizationBypassReason.POLICY_DISALLOWED
    if safety_class is StrategySafetyClass.LOSSY and not policy.allow_lossy:
        return TokenOptimizationBypassReason.POLICY_DISALLOWED
    if (
        safety_class is StrategySafetyClass.EXPERIMENTAL
        and policy.profile is not TokenOptimizationProfile.EXPERIMENTAL
    ):
        return TokenOptimizationBypassReason.POLICY_DISALLOWED
    return None


def _requires_validation(*, policy: TokenOptimizationPolicy, descriptor) -> bool:
    return policy.require_validation or descriptor.requires_validation


def _malformed_layer_result(
    *,
    raw_result: object,
    expected_layer_id: str,
) -> TokenOptimizationLayerResult | None:
    if not isinstance(raw_result, TokenOptimizationLayerResult):
        return _synthetic_layer_result(
            layer_id=expected_layer_id,
            output_content="",
            decision=TokenOptimizationLayerDecision.FAILED,
            metadata={"failure_kind": "invalid_result_type"},
        )

    if raw_result.layer_id != expected_layer_id:
        return _synthetic_layer_result(
            layer_id=expected_layer_id,
            output_content=raw_result.output_content,
            decision=TokenOptimizationLayerDecision.FAILED,
            metadata={
                "failure_kind": "layer_id_mismatch",
                "expected_layer_id": expected_layer_id,
                "actual_layer_id": raw_result.layer_id,
            },
        )
    return None


@dataclasses.dataclass
class _LayerDecisionOutcome:
    result: TokenOptimizationLayerResult
    current_content: str
    applied_layer_ids: list[str]
    bypassed_layer_ids: list[str]
    failed_layer_ids: list[str]
    fallback_used: bool = False
    malformed: TokenOptimizationLayerResult | None = None


def _apply_layer_decision(
    *,
    result: TokenOptimizationLayerResult,
    layer_input_content: str,
    original_content: str,
    applied_layer_ids: list[str],
    measure_only: bool,
) -> _LayerDecisionOutcome:
    decision = result.decision
    bypassed: list[str] = []
    failed: list[str] = []
    applied = list(applied_layer_ids)
    current = layer_input_content
    fallback = False
    malformed: TokenOptimizationLayerResult | None = None

    if decision is TokenOptimizationLayerDecision.APPLY:
        if measure_only:
            current = layer_input_content
        else:
            current = result.output_content
            applied.append(result.layer_id)
        return _LayerDecisionOutcome(
            result=result,
            current_content=current,
            applied_layer_ids=applied,
            bypassed_layer_ids=bypassed,
            failed_layer_ids=failed,
        )

    if decision is TokenOptimizationLayerDecision.BYPASS:
        if result.output_content != layer_input_content:
            malformed = _synthetic_layer_result(
                layer_id=result.layer_id,
                output_content=layer_input_content,
                decision=TokenOptimizationLayerDecision.FAILED,
                metadata={"failure_kind": "bypass_content_mismatch"},
            )
            return _LayerDecisionOutcome(
                result=result,
                current_content=layer_input_content,
                applied_layer_ids=applied,
                bypassed_layer_ids=bypassed,
                failed_layer_ids=failed,
                malformed=malformed,
            )
        bypassed.append(result.layer_id)
        return _LayerDecisionOutcome(
            result=result,
            current_content=current,
            applied_layer_ids=applied,
            bypassed_layer_ids=bypassed,
            failed_layer_ids=failed,
        )

    if decision is TokenOptimizationLayerDecision.FALLBACK:
        current = result.output_content
        fallback = True
        bypassed.append(result.layer_id)
        return _LayerDecisionOutcome(
            result=result,
            current_content=current,
            applied_layer_ids=applied,
            bypassed_layer_ids=bypassed,
            failed_layer_ids=failed,
            fallback_used=fallback,
        )

    if decision is TokenOptimizationLayerDecision.FAILED:
        failed.append(result.layer_id)
        return _LayerDecisionOutcome(
            result=result,
            current_content=layer_input_content,
            applied_layer_ids=applied,
            bypassed_layer_ids=bypassed,
            failed_layer_ids=failed,
        )

    if decision is TokenOptimizationLayerDecision.OVERRIDE_PREVIOUS:
        overridden = set(result.overridden_layer_ids)
        if not overridden.issubset(set(applied)):
            malformed = _synthetic_layer_result(
                layer_id=result.layer_id,
                output_content=layer_input_content,
                decision=TokenOptimizationLayerDecision.FAILED,
                metadata={"failure_kind": "invalid_override"},
            )
            return _LayerDecisionOutcome(
                result=result,
                current_content=layer_input_content,
                applied_layer_ids=applied,
                bypassed_layer_ids=bypassed,
                failed_layer_ids=failed,
                malformed=malformed,
            )
        for overridden_id in result.overridden_layer_ids:
            if overridden_id in applied:
                applied.remove(overridden_id)
        if measure_only:
            current = layer_input_content
        else:
            current = result.output_content
        applied.append(result.layer_id)
        return _LayerDecisionOutcome(
            result=result,
            current_content=current,
            applied_layer_ids=applied,
            bypassed_layer_ids=bypassed,
            failed_layer_ids=failed,
        )

    if decision is TokenOptimizationLayerDecision.REVERT_TO_ORIGINAL:
        if result.output_content != original_content:
            malformed = _synthetic_layer_result(
                layer_id=result.layer_id,
                output_content=layer_input_content,
                decision=TokenOptimizationLayerDecision.FAILED,
                metadata={"failure_kind": "invalid_revert"},
            )
            return _LayerDecisionOutcome(
                result=result,
                current_content=layer_input_content,
                applied_layer_ids=applied,
                bypassed_layer_ids=bypassed,
                failed_layer_ids=failed,
                malformed=malformed,
            )
        current = original_content
        applied.clear()
        bypassed.append(result.layer_id)
        fallback = True
        return _LayerDecisionOutcome(
            result=result,
            current_content=current,
            applied_layer_ids=applied,
            bypassed_layer_ids=bypassed,
            failed_layer_ids=failed,
            fallback_used=fallback,
        )

    return _LayerDecisionOutcome(
        result=result,
        current_content=layer_input_content,
        applied_layer_ids=applied,
        bypassed_layer_ids=bypassed,
        failed_layer_ids=failed,
    )
