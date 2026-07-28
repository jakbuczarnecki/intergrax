# © Artur Czarnecki. All rights reserved.

"""TOKEN-8A: pipeline resolution and TokenOptimizationPipelineRunner unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    StrategySafetyClass,
    TokenOptimizationBypassReason,
    TokenOptimizationLayerContext,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerDescriptor,
    TokenOptimizationLayerRef,
    TokenOptimizationLayerRequest,
    TokenOptimizationLayerResult,
    TokenOptimizationMechanism,
    TokenOptimizationPipelineConfig,
    TokenOptimizationPipelineMode,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
)
from intergrax.runtime.token_optimization.layers.exact_deduplication import ExactDeduplicationLayer
from intergrax.runtime.token_optimization.pipeline import (
    TokenOptimizationPipelineRunner,
    resolve_token_optimization_pipeline_layers,
)
from intergrax.runtime.token_optimization.registry import TokenOptimizationLayerRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _make_strategy(
    *,
    strategy_id: str,
    safety_class: StrategySafetyClass = StrategySafetyClass.LOSSLESS,
    plugin_id: str | None = None,
) -> TokenOptimizationStrategyRef:
    return TokenOptimizationStrategyRef(
        strategy_id=strategy_id,
        mechanism=TokenOptimizationMechanism.DEDUPLICATION,
        kind=TokenOptimizationStrategyKind.DEDUPLICATION,
        safety_class=safety_class,
        version="1",
        plugin_id=plugin_id,
    )


@dataclass
class _RecordingFakeLayer:
    layer_id: str
    safety_class: StrategySafetyClass = StrategySafetyClass.LOSSLESS
    supported_source_types: tuple[TokenOptimizationSourceType, ...] = ()
    requires_validation: bool = False
    result_factory: Any = None
    calls: list[TokenOptimizationLayerRequest] = field(default_factory=list)
    raise_error: Exception | None = None
    return_value: object | None = None

    def __post_init__(self) -> None:
        if self.result_factory is None:
            self.result_factory = self._default_result

    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        return TokenOptimizationLayerDescriptor(
            layer_id=self.layer_id,
            name=self.layer_id,
            version="1",
            strategy=_make_strategy(strategy_id=self.layer_id, safety_class=self.safety_class),
            supported_source_types=self.supported_source_types,
            safety_class=self.safety_class,
            requires_validation=self.requires_validation,
        )

    def _default_result(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        return TokenOptimizationLayerResult(
            layer_id=self.layer_id,
            output_content=request.current_content,
            decision=TokenOptimizationLayerDecision.BYPASS,
        )

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        self.calls.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        if self.return_value is not None:
            return self.return_value  # type: ignore[return-value]
        return self.result_factory(request)


def _enabled_policy(**overrides: object) -> TokenOptimizationPolicy:
    base = {
        "enabled": True,
        "profile": TokenOptimizationProfile.CONSERVATIVE,
    }
    base.update(overrides)
    return TokenOptimizationPolicy(**base)  # type: ignore[arg-type]


def _request(
    content: str = "alpha",
    *,
    source_type: TokenOptimizationSourceType = TokenOptimizationSourceType.PROMPT,
    policy: TokenOptimizationPolicy | None = None,
    protected_regions: tuple[ProtectedRegion, ...] = (),
) -> TokenOptimizationRequest:
    return TokenOptimizationRequest(
        content=content,
        source_type=source_type,
        policy=policy or _enabled_policy(),
        protected_regions=protected_regions,
        metadata={"operator": "safe"},
    )


def _runner(
    *layers: _RecordingFakeLayer,
    default_layers: tuple[TokenOptimizationLayerRef, ...] = (),
) -> TokenOptimizationPipelineRunner:
    return TokenOptimizationPipelineRunner(
        registry=TokenOptimizationLayerRegistry(layers=layers),
        default_layers=default_layers,
    )


def _ref(layer_id: str, **kwargs: object) -> TokenOptimizationLayerRef:
    return TokenOptimizationLayerRef(layer_id=layer_id, **kwargs)  # type: ignore[arg-type]


def test_replace_uses_only_config_layers() -> None:
    defaults = (_ref("builtin.default"),)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="p1",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.only"),),
    )

    resolved = resolve_token_optimization_pipeline_layers(
        default_layers=defaults,
        config=config,
    )

    assert resolved == (_ref("builtin.only"),)


def test_default_uses_defaults() -> None:
    defaults = (_ref("builtin.a"), _ref("builtin.b"))
    config = TokenOptimizationPipelineConfig(pipeline_id="p1")

    resolved = resolve_token_optimization_pipeline_layers(
        default_layers=defaults,
        config=config,
    )

    assert resolved == defaults


def test_default_replaces_same_layer_id() -> None:
    defaults = (_ref("builtin.a"), _ref("builtin.b"))
    replacement = _ref("builtin.a", settings={"mode": "custom"})
    config = TokenOptimizationPipelineConfig(
        pipeline_id="p1",
        layers=(replacement,),
    )

    resolved = resolve_token_optimization_pipeline_layers(
        default_layers=defaults,
        config=config,
    )

    assert resolved == (replacement, _ref("builtin.b"))


def test_disabled_config_ref_disables_default_layer() -> None:
    defaults = (_ref("builtin.a"),)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="p1",
        layers=(_ref("builtin.a", enabled=False),),
    )

    resolved = resolve_token_optimization_pipeline_layers(
        default_layers=defaults,
        config=config,
    )

    assert resolved[0].enabled is False


def test_default_appends_new_layer() -> None:
    defaults = (_ref("builtin.a"),)
    added = _ref("builtin.new")
    config = TokenOptimizationPipelineConfig(pipeline_id="p1", layers=(added,))

    resolved = resolve_token_optimization_pipeline_layers(
        default_layers=defaults,
        config=config,
    )

    assert resolved == (_ref("builtin.a"), added)


def test_allow_repeated_layers_keeps_duplicates() -> None:
    defaults = (_ref("builtin.a"),)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="p1",
        layers=(_ref("builtin.a"),),
        allow_repeated_layers=True,
    )

    resolved = resolve_token_optimization_pipeline_layers(
        default_layers=defaults,
        config=config,
    )

    assert len(resolved) == 2
    assert resolved[0].layer_id == "builtin.a"
    assert resolved[1].layer_id == "builtin.a"


def test_order_gives_deterministic_sorting() -> None:
    defaults = (
        _ref("builtin.first", order=10),
        _ref("builtin.second"),
        _ref("builtin.third"),
    )
    config = TokenOptimizationPipelineConfig(
        pipeline_id="p1",
        layers=(_ref("builtin.second", order=0),),
    )

    resolved = resolve_token_optimization_pipeline_layers(
        default_layers=defaults,
        config=config,
    )

    assert [layer.layer_id for layer in resolved] == [
        "builtin.second",
        "builtin.third",
        "builtin.first",
    ]


def test_duplicate_default_layers_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate layer_id"):
        TokenOptimizationPipelineRunner(
            registry=TokenOptimizationLayerRegistry(),
            default_layers=(_ref("builtin.a"), _ref("builtin.a")),
        )


def test_two_layers_run_sequentially() -> None:
    first = _RecordingFakeLayer(
        layer_id="builtin.first",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.first",
            output_content=request.current_content + "-first",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    second = _RecordingFakeLayer(
        layer_id="builtin.second",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.second",
            output_content=request.current_content + "-second",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    runner = _runner(first, second)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="seq",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.first"), _ref("builtin.second")),
    )

    result = runner.run(request=_request("base"), config=config)

    assert len(first.calls) == 1
    assert len(second.calls) == 1
    assert result.final_content == "base-first-second"
    assert result.applied_layer_ids == ("builtin.first", "builtin.second")


def test_second_layer_receives_original_and_current_content() -> None:
    first = _RecordingFakeLayer(
        layer_id="builtin.first",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.first",
            output_content="mutated",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    second = _RecordingFakeLayer(layer_id="builtin.second")
    runner = _runner(first, second)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="ctx",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.first"), _ref("builtin.second")),
    )

    runner.run(request=_request("original"), config=config)

    second_request = second.calls[0]
    assert second_request.original_content == "original"
    assert second_request.current_content == "mutated"


def test_layer_context_fields() -> None:
    first = _RecordingFakeLayer(
        layer_id="builtin.first",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.first",
            output_content="changed",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    second = _RecordingFakeLayer(layer_id="builtin.second")
    runner = _runner(first, second)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="pipeline-42",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.first"), _ref("builtin.second")),
    )

    runner.run(request=_request("x"), config=config)

    context = second.calls[0].layer_context
    assert isinstance(context, TokenOptimizationLayerContext)
    assert context is not None
    assert context.pipeline_id == "pipeline-42"
    assert context.layer_index == 1
    assert context.previous_layer_ids == ("builtin.first",)
    assert context.applied_layer_ids == ("builtin.first",)


def test_layer_ref_settings_in_request_metadata() -> None:
    layer = _RecordingFakeLayer(layer_id="builtin.settings")
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="meta",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.settings", settings={"threshold": 3}),),
    )

    runner.run(request=_request("x"), config=config)

    metadata = layer.calls[0].metadata
    assert metadata["operator"] == "safe"
    pipeline_meta = metadata["token_optimization_pipeline"]
    assert pipeline_meta["pipeline_id"] == "meta"
    assert pipeline_meta["layer_id"] == "builtin.settings"
    assert pipeline_meta["settings"] == {"threshold": 3}


def test_disabled_layer_not_called_and_recorded_as_bypass() -> None:
    layer = _RecordingFakeLayer(layer_id="builtin.off")
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="off",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.off", enabled=False),),
    )

    result = runner.run(request=_request("keep"), config=config)

    assert layer.calls == []
    assert result.bypassed_layer_ids == ("builtin.off",)
    assert result.final_content == "keep"
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.BYPASS
    assert result.layer_results[0].bypass_reason is TokenOptimizationBypassReason.DISABLED


def test_optional_missing_layer_bypasses_and_continues() -> None:
    second = _RecordingFakeLayer(
        layer_id="builtin.second",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.second",
            output_content="done",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    runner = _runner(second)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="missing",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.missing"), _ref("builtin.second")),
    )

    result = runner.run(request=_request("start"), config=config)

    assert result.final_content == "done"
    assert result.bypassed_layer_ids == ("builtin.missing",)
    assert result.layer_results[0].bypass_reason is TokenOptimizationBypassReason.PLUGIN_UNAVAILABLE


def test_required_missing_layer_falls_back_to_original() -> None:
    runner = _runner()
    config = TokenOptimizationPipelineConfig(
        pipeline_id="required-missing",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.missing", required=True),),
    )

    result = runner.run(request=_request("original"), config=config)

    assert result.final_content == "original"
    assert result.fallback_used is True
    assert result.failed_layer_ids == ("builtin.missing",)
    assert result.receipt_metadata["completed"] is False


def test_optional_exception_records_failed_and_continues() -> None:
    failing = _RecordingFakeLayer(layer_id="builtin.fail", raise_error=RuntimeError("boom"))
    second = _RecordingFakeLayer(
        layer_id="builtin.second",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.second",
            output_content="after",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    runner = _runner(failing, second)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="opt-exc",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.fail"), _ref("builtin.second")),
    )

    result = runner.run(request=_request("before"), config=config)

    assert result.final_content == "after"
    assert result.failed_layer_ids == ("builtin.fail",)
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.FAILED
    assert "boom" not in str(result.layer_results[0].metadata)


def test_required_exception_falls_back_to_original() -> None:
    failing = _RecordingFakeLayer(
        layer_id="builtin.fail",
        raise_error=RuntimeError("boom"),
    )
    runner = _runner(failing)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="req-exc",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.fail", required=True),),
    )

    result = runner.run(request=_request("original"), config=config)

    assert result.final_content == "original"
    assert result.fallback_used is True
    assert result.failed_layer_ids == ("builtin.fail",)


def test_invalid_return_type_treated_as_malformed_failure() -> None:
    bad = _RecordingFakeLayer(layer_id="builtin.bad", return_value={"not": "a result"})
    runner = _runner(bad)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="bad-type",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.bad"),),
    )

    result = runner.run(request=_request("same"), config=config)

    assert result.failed_layer_ids == ("builtin.bad",)
    assert result.final_content == "same"
    assert result.layer_results[0].metadata["failure_kind"] == "invalid_result_type"


def test_layer_id_mismatch_treated_as_malformed_failure() -> None:
    bad = _RecordingFakeLayer(
        layer_id="builtin.bad",
        return_value=TokenOptimizationLayerResult(
            layer_id="other.id",
            output_content="x",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    runner = _runner(bad)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="mismatch",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.bad"),),
    )

    result = runner.run(request=_request("same"), config=config)

    assert result.failed_layer_ids == ("builtin.bad",)
    assert result.layer_results[0].metadata["failure_kind"] == "layer_id_mismatch"


def test_apply_changes_working_content() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.apply",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.apply",
            output_content="applied",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="apply",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.apply"),),
    )

    result = runner.run(request=_request("before"), config=config)

    assert result.final_content == "applied"
    assert result.applied_layer_ids == ("builtin.apply",)


def test_bypass_keeps_content() -> None:
    layer = _RecordingFakeLayer(layer_id="builtin.bypass")
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="bypass",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.bypass"),),
    )

    result = runner.run(request=_request("unchanged"), config=config)

    assert result.final_content == "unchanged"
    assert result.bypassed_layer_ids == ("builtin.bypass",)


def test_bypass_with_changed_content_rejected() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.bypass",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.bypass",
            output_content="changed",
            decision=TokenOptimizationLayerDecision.BYPASS,
        ),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="bad-bypass",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.bypass"),),
    )

    result = runner.run(request=_request("same"), config=config)

    assert result.failed_layer_ids == ("builtin.bypass",)
    assert result.final_content == "same"


def test_fallback_sets_aggregate_fallback_and_keeps_safe_content() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.fallback",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.fallback",
            output_content=request.current_content,
            decision=TokenOptimizationLayerDecision.FALLBACK,
            fallback_used=True,
        ),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="fallback",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.fallback"),),
    )

    result = runner.run(request=_request("safe"), config=config)

    assert result.final_content == "safe"
    assert result.fallback_used is True
    assert result.bypassed_layer_ids == ("builtin.fallback",)


def test_optional_failed_decision_does_not_accept_output() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.failed",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.failed",
            output_content="ignored",
            decision=TokenOptimizationLayerDecision.FAILED,
        ),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="failed",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.failed"),),
    )

    result = runner.run(request=_request("keep"), config=config)

    assert result.final_content == "keep"
    assert result.failed_layer_ids == ("builtin.failed",)


def test_override_previous_updates_applied_ids() -> None:
    first = _RecordingFakeLayer(
        layer_id="builtin.first",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.first",
            output_content="first-out",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    second = _RecordingFakeLayer(
        layer_id="builtin.second",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.second",
            output_content="second-out",
            decision=TokenOptimizationLayerDecision.OVERRIDE_PREVIOUS,
            previous_changes_overridden=True,
            overridden_layer_ids=("builtin.first",),
            override_reason="replace",
        ),
    )
    runner = _runner(first, second)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="override",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.first"), _ref("builtin.second")),
    )

    result = runner.run(request=_request("start"), config=config)

    assert result.final_content == "second-out"
    assert result.applied_layer_ids == ("builtin.second",)


def test_invalid_override_ids_rejected() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.override",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.override",
            output_content="out",
            decision=TokenOptimizationLayerDecision.OVERRIDE_PREVIOUS,
            previous_changes_overridden=True,
            overridden_layer_ids=("missing.layer",),
            override_reason="bad",
        ),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="bad-override",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.override"),),
    )

    result = runner.run(request=_request("same"), config=config)

    assert result.failed_layer_ids == ("builtin.override",)
    assert result.layer_results[0].metadata["failure_kind"] == "invalid_override"


def test_revert_to_original_restores_global_original() -> None:
    first = _RecordingFakeLayer(
        layer_id="builtin.first",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.first",
            output_content="mutated",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    second = _RecordingFakeLayer(
        layer_id="builtin.revert",
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.revert",
            output_content="original",
            decision=TokenOptimizationLayerDecision.REVERT_TO_ORIGINAL,
            previous_changes_overridden=True,
            override_reason="reset",
        ),
    )
    runner = _runner(first, second)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="revert",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.first"), _ref("builtin.revert")),
    )

    result = runner.run(request=_request("original"), config=config)

    assert result.final_content == "original"
    assert result.applied_layer_ids == ()
    assert result.fallback_used is True
    assert result.bypassed_layer_ids == ("builtin.revert",)


def test_globally_disabled_policy_bypasses_layers() -> None:
    layer = _RecordingFakeLayer(layer_id="builtin.any")
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="disabled",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.any"),),
    )

    result = runner.run(
        request=_request("same", policy=TokenOptimizationPolicy(enabled=False)),
        config=config,
    )

    assert layer.calls == []
    assert result.bypassed_layer_ids == ("builtin.any",)
    assert result.final_content == "same"


def test_off_profile_does_not_invoke_layers() -> None:
    layer = _RecordingFakeLayer(layer_id="builtin.any")
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="off",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.any"),),
    )

    result = runner.run(
        request=_request(
            "same",
            policy=TokenOptimizationPolicy(enabled=True, profile=TokenOptimizationProfile.OFF),
        ),
        config=config,
    )

    assert layer.calls == []
    assert result.bypassed_layer_ids == ("builtin.any",)


def test_measure_only_blocks_mutating_layer() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.lossless",
        safety_class=StrategySafetyClass.LOSSLESS,
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.lossless",
            output_content="mutated",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="measure",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.lossless"),),
    )

    result = runner.run(
        request=_request(
            "same",
            policy=_enabled_policy(profile=TokenOptimizationProfile.MEASURE_ONLY),
        ),
        config=config,
    )

    assert layer.calls == []
    assert result.bypassed_layer_ids == ("builtin.lossless",)
    assert result.final_content == "same"
    assert result.layer_results[0].bypass_reason is TokenOptimizationBypassReason.POLICY_DISALLOWED


def test_lossy_without_allow_lossy_blocked() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.lossy",
        safety_class=StrategySafetyClass.LOSSY,
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="lossy",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.lossy"),),
    )

    result = runner.run(request=_request("same", policy=_enabled_policy(allow_lossy=False)), config=config)

    assert layer.calls == []
    assert result.layer_results[0].bypass_reason is TokenOptimizationBypassReason.POLICY_DISALLOWED


def test_experimental_blocked_outside_experimental_profile() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.experimental",
        safety_class=StrategySafetyClass.EXPERIMENTAL,
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="experimental",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.experimental"),),
    )

    result = runner.run(request=_request("same"), config=config)

    assert layer.calls == []
    assert result.layer_results[0].bypass_reason is TokenOptimizationBypassReason.POLICY_DISALLOWED


def test_unsupported_source_type_blocked_before_call() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.prompt-only",
        supported_source_types=(TokenOptimizationSourceType.PROMPT,),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="source",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.prompt-only"),),
    )

    result = runner.run(
        request=_request("same", source_type=TokenOptimizationSourceType.TOOL_OUTPUT),
        config=config,
    )

    assert layer.calls == []
    assert result.layer_results[0].bypass_reason is TokenOptimizationBypassReason.UNSUPPORTED_SOURCE_TYPE


def test_required_unsupported_source_type_falls_back() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.prompt-only",
        supported_source_types=(TokenOptimizationSourceType.PROMPT,),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="required-source",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.prompt-only", required=True),),
    )

    result = runner.run(
        request=_request("original", source_type=TokenOptimizationSourceType.TOOL_OUTPUT),
        config=config,
    )

    assert result.final_content == "original"
    assert result.fallback_used is True
    assert result.failed_layer_ids == ("builtin.prompt-only",)


def test_protected_region_validation_rejects_mutating_layer() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.mutate",
        requires_validation=True,
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.mutate",
            output_content="removed",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="validate",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.mutate"),),
    )
    protected = (ProtectedRegion(kind=ProtectedRegionKind.IDENTIFIER, value="SECRET"),)

    result = runner.run(
        request=_request("keep SECRET", protected_regions=protected),
        config=config,
    )

    assert result.final_content == "keep SECRET"
    assert result.fallback_used is True
    assert result.bypassed_layer_ids == ("builtin.mutate",)
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.FALLBACK
    assert result.layer_results[0].bypass_reason is TokenOptimizationBypassReason.VALIDATION_FAILED


def test_validation_not_applicable_is_accepted() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.mutate",
        requires_validation=True,
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.mutate",
            output_content="trimmed",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="not-applicable",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.mutate"),),
    )

    result = runner.run(request=_request("no protected terms"), config=config)

    assert result.final_content == "trimmed"
    assert result.applied_layer_ids == ("builtin.mutate",)
    validation = result.layer_results[0].validation
    assert validation is not None
    assert validation.status.value == "not_applicable"


def test_manual_exact_deduplication_layer_through_runner() -> None:
    layer = ExactDeduplicationLayer()
    registry = TokenOptimizationLayerRegistry(layers=(layer,))
    runner = TokenOptimizationPipelineRunner(registry=registry)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="dedupe-proof",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.exact_deduplication"),),
    )
    content = "line\nline\nunique"

    result = runner.run(
        request=_request(
            content,
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            policy=_enabled_policy(),
        ),
        config=config,
    )

    assert result.applied_layer_ids == ("builtin.exact_deduplication",)
    assert result.final_content == "line\nunique"
    assert result.original_content == content


def _apply_layer(layer_id: str) -> _RecordingFakeLayer:
    return _RecordingFakeLayer(
        layer_id=layer_id,
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id=layer_id,
            output_content=f"{request.current_content}-applied",
            decision=TokenOptimizationLayerDecision.APPLY,
        ),
    )


def _assert_required_rollback_after_apply(
    result: object,
    *,
    original: str,
    failing_layer_id: str,
    applied_layer_id: str,
) -> None:
    assert result.original_content == original  # type: ignore[attr-defined]
    assert result.final_content == original  # type: ignore[attr-defined]
    assert result.applied_layer_ids == ()  # type: ignore[attr-defined]
    assert result.fallback_used is True  # type: ignore[attr-defined]
    assert result.receipt_metadata["completed"] is False  # type: ignore[attr-defined]
    assert result.receipt_metadata["required_failure_layer_id"] == failing_layer_id  # type: ignore[attr-defined]
    layer_ids = [item.layer_id for item in result.layer_results]  # type: ignore[attr-defined]
    assert applied_layer_id in layer_ids
    assert failing_layer_id in layer_ids


@pytest.mark.parametrize(
    ("failing_layer", "failing_ref", "request_kwargs"),
    [
        pytest.param(
            None,
            _ref("builtin.missing", required=True),
            {},
            id="missing_required_layer",
        ),
        pytest.param(
            _RecordingFakeLayer(
                layer_id="builtin.fail",
                raise_error=RuntimeError("sensitive failure text"),
            ),
            _ref("builtin.fail", required=True),
            {},
            id="required_exception",
        ),
        pytest.param(
            _RecordingFakeLayer(
                layer_id="builtin.fail",
                result_factory=lambda request: TokenOptimizationLayerResult(
                    layer_id="builtin.fail",
                    output_content="ignored",
                    decision=TokenOptimizationLayerDecision.FAILED,
                ),
            ),
            _ref("builtin.fail", required=True),
            {},
            id="required_failed_decision",
        ),
        pytest.param(
            _RecordingFakeLayer(
                layer_id="builtin.fail",
                return_value={"not": "a result"},
            ),
            _ref("builtin.fail", required=True),
            {},
            id="required_malformed_result",
        ),
        pytest.param(
            _RecordingFakeLayer(
                layer_id="builtin.prompt-only",
                supported_source_types=(TokenOptimizationSourceType.PROMPT,),
            ),
            _ref("builtin.prompt-only", required=True),
            {"source_type": TokenOptimizationSourceType.TOOL_OUTPUT},
            id="required_unsupported_source",
        ),
    ],
)
def test_required_failure_after_apply_clears_applied_state(
    failing_layer: _RecordingFakeLayer | None,
    failing_ref: TokenOptimizationLayerRef,
    request_kwargs: dict[str, object],
) -> None:
    apply_layer = _apply_layer("builtin.apply")
    layers = [apply_layer]
    if failing_layer is not None:
        layers.append(failing_layer)
    runner = _runner(*layers)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="rollback",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.apply"), failing_ref),
    )

    result = runner.run(request=_request("original", **request_kwargs), config=config)

    _assert_required_rollback_after_apply(
        result,
        original="original",
        failing_layer_id=failing_ref.layer_id,
        applied_layer_id="builtin.apply",
    )


def test_unsafe_content_changing_fallback_rejected_by_validation() -> None:
    layer = _RecordingFakeLayer(
        layer_id="builtin.fallback",
        requires_validation=True,
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.fallback",
            output_content="removed",
            decision=TokenOptimizationLayerDecision.FALLBACK,
            fallback_used=True,
        ),
    )
    runner = _runner(layer)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="fallback-validate",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.fallback"),),
    )
    protected = (ProtectedRegion(kind=ProtectedRegionKind.IDENTIFIER, value="SECRET"),)

    result = runner.run(
        request=_request("keep SECRET", protected_regions=protected),
        config=config,
    )

    assert result.final_content == "keep SECRET"
    assert result.fallback_used is True
    assert result.bypassed_layer_ids == ("builtin.fallback",)
    assert result.layer_results[0].decision is TokenOptimizationLayerDecision.FALLBACK
    assert result.layer_results[0].bypass_reason is TokenOptimizationBypassReason.VALIDATION_FAILED
    assert result.layer_results[0].validation is not None


def test_safe_content_changing_fallback_accepted_and_passed_to_next_layer() -> None:
    fallback_layer = _RecordingFakeLayer(
        layer_id="builtin.fallback",
        requires_validation=True,
        result_factory=lambda request: TokenOptimizationLayerResult(
            layer_id="builtin.fallback",
            output_content="trimmed safe",
            decision=TokenOptimizationLayerDecision.FALLBACK,
            fallback_used=True,
        ),
    )
    second = _RecordingFakeLayer(layer_id="builtin.second")
    runner = _runner(fallback_layer, second)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="fallback-safe",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.fallback"), _ref("builtin.second")),
    )

    result = runner.run(request=_request("trimmed safe content"), config=config)

    assert result.final_content == "trimmed safe"
    assert result.fallback_used is True
    assert len(second.calls) == 1
    assert second.calls[0].current_content == "trimmed safe"


def test_required_exception_recorded_as_executed_without_sensitive_message() -> None:
    failing = _RecordingFakeLayer(
        layer_id="builtin.fail",
        raise_error=RuntimeError("sensitive failure text"),
    )
    runner = _runner(failing)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="req-exec",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.fail", required=True),),
    )

    result = runner.run(request=_request("original"), config=config)

    assert result.receipt_metadata["executed_layer_ids"] == ["builtin.fail"]
    assert "sensitive failure text" not in str(result.layer_results)
    assert "sensitive failure text" not in str(result.receipt_metadata)
    assert "sensitive failure text" not in str(result.metadata)


def test_pre_execution_bypasses_not_in_executed_layer_ids() -> None:
    disabled = _RecordingFakeLayer(layer_id="builtin.disabled")
    policy_blocked = _RecordingFakeLayer(
        layer_id="builtin.lossy",
        safety_class=StrategySafetyClass.LOSSY,
    )
    unsupported = _RecordingFakeLayer(
        layer_id="builtin.prompt-only",
        supported_source_types=(TokenOptimizationSourceType.PROMPT,),
    )
    runner = _runner(disabled, policy_blocked, unsupported)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="pre-exec",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(
            _ref("builtin.disabled", enabled=False),
            _ref("builtin.lossy"),
            _ref("builtin.prompt-only"),
            _ref("builtin.missing"),
        ),
    )

    result = runner.run(
        request=_request(
            "content",
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            policy=_enabled_policy(allow_lossy=False),
        ),
        config=config,
    )

    executed = result.receipt_metadata["executed_layer_ids"]
    assert "builtin.disabled" not in executed
    assert "builtin.lossy" not in executed
    assert "builtin.prompt-only" not in executed
    assert "builtin.missing" not in executed


def test_optional_exception_recorded_exactly_once_in_executed_layer_ids() -> None:
    failing = _RecordingFakeLayer(layer_id="builtin.fail", raise_error=RuntimeError("boom"))
    second = _RecordingFakeLayer(layer_id="builtin.second")
    runner = _runner(failing, second)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="opt-exec",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_ref("builtin.fail"), _ref("builtin.second")),
    )

    result = runner.run(request=_request("before"), config=config)

    assert result.receipt_metadata["executed_layer_ids"].count("builtin.fail") == 1
