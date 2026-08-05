from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.runtime.token_optimization.proofs.config import (
    load_universal_token_optimization_proof_config,
)
from intergrax.runtime.token_optimization.proofs.contracts import (
    ProofCompositionError,
    ProofConfigurationError,
)
from intergrax.runtime.token_optimization.proofs.runner import (
    UniversalTokenOptimizationProofRunner,
    _protected_identity_digest,
    _protected_region_evidence,
    _prefix_identity_evidence,
    _measurements_from_pipeline,
)
from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    TokenOptimizationPipelineResult,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationRouterConfigurationId,
    TokenOptimizationLLMRouterResult,
    TokenOptimizationRouterReason,
    TokenOptimizationRouterReasonCode,
    TokenOptimizationRouterRisk,
    TokenOptimizationRouterStatus,
    TokenOptimizationRouterTransport,
)
from intergrax.runtime.token_optimization.prompt_assembly import (
    PromptAssemblyMessageBlock,
    assemble_cache_stable_prompt,
)


def _config(
    tmp_path: Path,
    *,
    provider: str = "vllm",
    run_mode: str = "offline_smoke",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "proof.toml"
    path.write_text(
        f"""
schema_version = "token-optimization-proof.v1"
proof_id = "runner-proof"
run_mode = "{run_mode}"

[adapter]
adapter_id = "offline"
provider = "{provider}"
type = "openai_compatible"
model = "offline-model"
base_url = "offline://local"
api_key_env = "RUNNER_UNUSED_KEY"
timeout_seconds = 5.0
max_output_tokens = 32
temperature = 0.0

[router]
enabled = true
configuration_id = "exact_only"
minimum_confidence = 0.6
allow_structured_output_fallback = true
require_review_for_protected_lossy_content = true

[pipeline]
mode = "replace"
layer_ids = ["builtin.exact_deduplication"]
failure_policy = "continue"

[output]
directory = ".artifacts/proof"
fail_if_exists = true

[[cases]]
case_id = "first"
source_type = "prompt"
content = "one\\none\\ntwo"
tags = ["smoke"]

[cases.policy]
enabled = true
profile = "balanced"
allow_lossy = false
require_validation = true
fallback_on_validation_failure = true

[[cases]]
case_id = "second"
source_type = "prompt"
content = "clean content"
tags = ["smoke"]

[cases.policy]
enabled = true
profile = "balanced"
allow_lossy = false
require_validation = true
fallback_on_validation_failure = true
""",
        encoding="utf-8",
    )
    return load_universal_token_optimization_proof_config(path)


def test_offline_runner_uses_real_composition_and_deterministic_order(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls = 0

    def router_factory(**kwargs):
        nonlocal calls
        calls += 1
        from intergrax.runtime.token_optimization.llm_router import (
            TokenOptimizationLLMRouter,
        )

        return TokenOptimizationLLMRouter(**kwargs)

    fixed_time = datetime(2026, 8, 5, 7, 0, tzinfo=UTC)
    runner = UniversalTokenOptimizationProofRunner(
        router_factory=router_factory,
        clock=lambda: fixed_time,
        run_id_factory=lambda: "fixed-run",
    )
    result = runner.run(config, persist_artifacts=False)

    assert result.success is True
    assert calls == 2
    assert [case.case_id for case in result.cases] == ["first", "second"]
    assert all(case.router_status == "routed" for case in result.cases)
    assert result.cases[0].selected_configuration_id == "exact_only"
    assert result.cases[0].applied_layer_ids == ("builtin.exact_deduplication",)
    assert all(case.raw_content_included is False for case in result.cases)
    assert result.cases[0].router_evidence.review_required is False
    assert result.cases[0].router_evidence.confidence == 1.0
    assert result.cases[0].pipeline_evidence.completed is True
    assert result.cases[0].pipeline_evidence.fallback_applied is False
    assert result.cases[0].protected_region_evidence.input_protected_region_count == 0
    assert result.cases[0].prefix_identity_evidence.identity_available is True
    assert result.cases[0].prefix_identity_evidence.stable_prefix_identity


def test_offline_runner_is_network_free_and_has_no_second_engine() -> None:
    source = Path(
        "intergrax/runtime/token_optimization/proofs/runner.py"
    ).read_text(encoding="utf-8")

    assert "LLMAdapterRegistry" in source
    assert "TokenOptimizationLLMRouter" in source
    assert "BuiltInTokenOptimizationLayerCatalog" in source
    assert "TokenOptimizationPipelineRunner" in source
    assert "class ProofOptimizationEngine" not in source
    assert "class MockOptimizationEngine" not in source
    assert "class AlternativePipelineRunner" not in source


@pytest.mark.parametrize("provider", ("vllm", "openai", "groq"))
def test_loader_accepts_canonical_provider_ids_without_network_call(
    tmp_path: Path,
    provider: str,
) -> None:
    config = _config(tmp_path / provider, provider=provider)
    assert config.adapter.provider == provider


class _RecordingLiveAdapter(LLMAdapter):
    def __init__(self, *, provider: str, **kwargs) -> None:
        super().__init__()
        self.provider = LLMProvider(provider)
        self.model = kwargs["model"]
        self.model_name_for_token_estimation = self.model
        self.create_kwargs = kwargs

    @property
    def context_window_tokens(self) -> int:
        return 32768

    def supports_structured_output(self) -> bool:
        return True

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        return LLMAdapterResponse(
            content="controlled-live-composition",
            model=self.model,
            provider=self.provider.value,
        )

    def generate_structured(
        self,
        messages,
        output_model: type,
        **kwargs,
    ) -> LLMStructuredResult:
        decision = output_model(
            configuration_id=TokenOptimizationRouterConfigurationId("exact_only"),
            reason_code=TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
            risk=TokenOptimizationRouterRisk.LOW,
            review_required=False,
            confidence=1.0,
        )
        return LLMStructuredResult(
            parsed=decision,
            response=LLMAdapterResponse(
                content="controlled-live-composition",
                model=self.model,
                provider=self.provider.value,
            ),
        )


@pytest.mark.parametrize("provider", ("vllm", "openai"))
def test_live_runner_uses_injected_registry_contract_without_network(
    tmp_path: Path,
    monkeypatch,
    provider: str,
) -> None:
    monkeypatch.setenv("RUNNER_UNUSED_KEY", "controlled-test-key")
    created: list[_RecordingLiveAdapter] = []

    class ScopedLiveRegistry(LLMAdapterRegistry):
        _factories = {}
        create_calls = []

        @classmethod
        def create(cls, requested_provider, **kwargs):
            cls.create_calls.append((requested_provider, kwargs))
            return super().create(requested_provider, **kwargs)

    def factory(**kwargs):
        adapter = _RecordingLiveAdapter(provider=provider, **kwargs)
        created.append(adapter)
        return adapter

    ScopedLiveRegistry.register(provider, factory)
    result = UniversalTokenOptimizationProofRunner(
        adapter_registry=ScopedLiveRegistry,
    ).run(
        _config(tmp_path, provider=provider, run_mode="live_adapter"),
        persist_artifacts=False,
    )

    assert result.success is True
    assert len(ScopedLiveRegistry.create_calls) == 1
    requested_provider, create_kwargs = ScopedLiveRegistry.create_calls[0]
    assert requested_provider == provider
    assert create_kwargs == {
        "model": "offline-model",
        "base_url": "offline://local",
        "api_key": "controlled-test-key",
        "timeout_sec": 5.0,
        "max_tokens": 32,
        "temperature": 0.0,
    }
    assert len(created) == 1
    assert created[0].create_kwargs == create_kwargs
    assert type(created[0]).__name__ == "_RecordingLiveAdapter"
    assert result.environment.provider == provider


def test_loader_rejects_unknown_provider_closed(tmp_path: Path) -> None:
    with pytest.raises(ProofConfigurationError, match="UNSUPPORTED_ADAPTER"):
        _config(tmp_path, provider="unknown")


def test_measurements_preserve_independent_baseline_and_optimized_values() -> None:
    result = SimpleNamespace(
        aggregate_measurement=SimpleNamespace(baseline_tokens=100, optimized_tokens=70)
    )
    baseline, optimized = _measurements_from_pipeline(result)

    assert baseline.available is True
    assert baseline.value == 100
    assert optimized.available is True
    assert optimized.value == 70

    missing_baseline, missing_optimized = _measurements_from_pipeline(SimpleNamespace())
    assert missing_baseline.available is False
    assert missing_optimized.available is False


def test_protected_identity_is_deterministic_and_redaction_safe(tmp_path: Path) -> None:
    config = _config(tmp_path)
    regions = (
        ProtectedRegion(ProtectedRegionKind.URL, "https://secret.example"),
        ProtectedRegion(ProtectedRegionKind.PATH, "/sensitive/path"),
    )
    request = config.cases[0].request
    request_with_regions = type(request)(
        content=request.content,
        source_type=request.source_type,
        policy=request.policy,
        attribution=request.attribution,
        strategy=request.strategy,
        protected_regions=regions,
        metadata=request.metadata,
    )

    first = _protected_identity_digest(regions)
    second = _protected_identity_digest(regions)
    evidence = _protected_region_evidence(request_with_regions, None)

    assert first == second
    assert first is not None
    assert "secret.example" not in first
    assert "/sensitive/path" not in first
    assert evidence.input_protected_region_count == 2
    assert evidence.protected_region_validation_status == "not_run"
    assert evidence.input_identity_digest == first
    assert evidence.preserved_identity_digest is None
    assert _prefix_identity_evidence(None).identity_available is False


def test_offline_runs_are_concurrent_and_do_not_mutate_global_registry(
    tmp_path: Path,
) -> None:
    from threading import Barrier

    from intergrax.runtime.token_optimization.llm_router import (
        TokenOptimizationLLMRouter,
    )

    before = dict(LLMAdapterRegistry._factories)
    barrier = Barrier(2)

    def run(index: int):
        def router_factory(**kwargs):
            barrier.wait()
            return TokenOptimizationLLMRouter(**kwargs)

        runner = UniversalTokenOptimizationProofRunner(
            router_factory=router_factory,
        )
        return runner.run(
            _config(tmp_path / f"run-{index}"),
            run_id=f"run-{index}",
            persist_artifacts=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(run, (1, 2)))

    assert all(result.success for result in results)
    assert dict(LLMAdapterRegistry._factories) == before


def test_failed_offline_composition_and_execution_preserve_global_registry(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    before = dict(LLMAdapterRegistry._factories)
    config = _config(tmp_path / "failed")
    invalid_router = replace(config.router, configuration_id="unknown")
    invalid_config = replace(config, router=invalid_router)

    with pytest.raises(ProofCompositionError, match="UNKNOWN_ROUTER_CONFIGURATION"):
        UniversalTokenOptimizationProofRunner().run(
            invalid_config,
            persist_artifacts=False,
        )

    class FailingPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            raise RuntimeError("case failure")

    result = UniversalTokenOptimizationProofRunner(
        pipeline_runner_factory=FailingPipeline,
    ).run(config, persist_artifacts=False)

    assert result.success is False
    assert result.cases[0].error_reason_code == "PIPELINE_EXECUTION_FAILED"
    assert result.cases[0].router_evidence.review_required is None
    assert result.cases[0].router_evidence.confidence is None
    assert result.cases[0].pipeline_evidence.fallback_applied is None
    assert dict(LLMAdapterRegistry._factories) == before


def _prompt_report(*, tools_schema: tuple[dict[str, object], ...] = ()):
    return assemble_cache_stable_prompt(
        stable_prefix_blocks=(
            PromptAssemblyMessageBlock(
                block_id="proof-stable",
                message=ChatMessage(role="system", content="proof-stable"),
            ),
        ),
        dynamic_tail=(ChatMessage(role="user", content="proof-tail"),),
        tools_schema=tools_schema,
    ).report


def _controlled_router_result(
    *,
    status: TokenOptimizationRouterStatus = TokenOptimizationRouterStatus.ROUTED,
    transport: TokenOptimizationRouterTransport = (
        TokenOptimizationRouterTransport.NATIVE_TOOLS
    ),
    configuration_id: TokenOptimizationRouterConfigurationId | None = (
        TokenOptimizationRouterConfigurationId.EXACT_ONLY
    ),
    reason_code: TokenOptimizationRouterReasonCode | None = (
        TokenOptimizationRouterReasonCode.EXACT_DUPLICATES
    ),
    risk: TokenOptimizationRouterRisk | None = TokenOptimizationRouterRisk.LOW,
    review_required: bool | None = False,
    confidence: float | None = 1.0,
    prompt_assembly_report=None,
    reason=None,
) -> TokenOptimizationLLMRouterResult:
    return TokenOptimizationLLMRouterResult(
        request_id="controlled-proof-router",
        status=status,
        reason=reason,
        transport=transport,
        configuration_id=configuration_id,
        reason_code=reason_code,
        risk=risk,
        review_required=review_required,
        confidence=confidence,
        provider="controlled",
        model="controlled",
        tool_call_id="controlled-call",
        pipeline_config=None,
        pipeline_result=None,
        executed=False,
        prompt_assembly_report=prompt_assembly_report,
    )


def _controlled_pipeline_result(
    *,
    completed: bool,
    fallback_used: bool,
    validation: ProtectedRegionValidationResult | None = None,
    receipt_metadata: dict[str, object] | None = None,
    baseline_tokens: int = 100,
    optimized_tokens: int = 70,
) -> TokenOptimizationPipelineResult:
    return TokenOptimizationPipelineResult(
        pipeline_id="controlled-proof-pipeline",
        original_content="proof-original",
        final_content="proof-final",
        layer_results=(
            (SimpleNamespace(validation=validation),) if validation is not None else ()
        ),
        applied_layer_ids=("controlled.layer",) if completed else (),
        fallback_used=fallback_used,
        aggregate_measurement=SimpleNamespace(
            baseline_tokens=baseline_tokens,
            optimized_tokens=optimized_tokens,
        ),
        receipt_metadata={
            "completed": completed,
            **(receipt_metadata or {}),
        },
    )


def _run_controlled_case(
    tmp_path: Path,
    *,
    router_result: TokenOptimizationLLMRouterResult,
    pipeline_result: TokenOptimizationPipelineResult | None = None,
    pipeline_runner_factory=None,
):
    config = replace(_config(tmp_path), cases=(_config(tmp_path / "case").cases[0],))

    class _ControlledRouter:
        def route(self, request):
            return router_result

    class _ControlledPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            assert pipeline_result is not None
            return pipeline_result

    selected_pipeline_factory = pipeline_runner_factory or _ControlledPipeline

    return UniversalTokenOptimizationProofRunner(
        router_factory=lambda **kwargs: _ControlledRouter(),
        pipeline_runner_factory=selected_pipeline_factory,
    ).run(config, persist_artifacts=False)


@pytest.mark.parametrize(
    ("review_required", "confidence", "status", "risk", "reason_code"),
    [
        (
            False,
            0.0,
            TokenOptimizationRouterStatus.ROUTED,
            TokenOptimizationRouterRisk.LOW,
            TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
        ),
        (
            True,
            1.0,
            TokenOptimizationRouterStatus.REVIEW_REQUIRED,
            TokenOptimizationRouterRisk.HIGH,
            TokenOptimizationRouterReasonCode.PROTECTED_OR_HIGH_RISK,
        ),
    ],
)
def test_runner_preserves_router_evidence_boundaries_and_review(
    tmp_path: Path,
    review_required: bool,
    confidence: float,
    status: TokenOptimizationRouterStatus,
    risk: TokenOptimizationRouterRisk,
    reason_code: TokenOptimizationRouterReasonCode,
) -> None:
    router_result = _controlled_router_result(
        status=status,
        review_required=review_required,
        confidence=confidence,
        risk=risk,
        reason_code=reason_code,
        reason=(
            TokenOptimizationRouterReason.MODEL_REQUESTED_REVIEW
            if review_required
            else None
        ),
        prompt_assembly_report=_prompt_report(),
    )

    result = _run_controlled_case(
        tmp_path,
        router_result=router_result,
        pipeline_result=_controlled_pipeline_result(
            completed=True,
            fallback_used=False,
        ),
    )
    evidence = result.cases[0].router_evidence

    assert evidence.review_required is review_required
    assert evidence.confidence == confidence
    assert evidence.risk == risk.value
    assert evidence.reason_code == reason_code.value
    assert evidence.status == status.value
    assert result.cases[0].prefix_identity_evidence.identity_available is True
    if review_required:
        assert result.cases[0].pipeline_evidence.completed is None
    else:
        assert result.cases[0].pipeline_evidence.completed is True


@pytest.mark.parametrize(
    ("transport", "expected_fallback", "status", "prompt_report"),
    [
        (
            TokenOptimizationRouterTransport.NATIVE_TOOLS,
            False,
            TokenOptimizationRouterStatus.ROUTED,
            _prompt_report(
                tools_schema=(
                    {
                        "type": "function",
                        "function": {"name": "proof.tool"},
                    },
                )
            ),
        ),
        (
            TokenOptimizationRouterTransport.STRUCTURED_OUTPUT,
            True,
            TokenOptimizationRouterStatus.ROUTED,
            _prompt_report(),
        ),
        (
            TokenOptimizationRouterTransport.UNSUPPORTED,
            None,
            TokenOptimizationRouterStatus.BLOCKED,
            None,
        ),
    ],
)
def test_runner_preserves_router_transport_evidence(
    tmp_path: Path,
    transport: TokenOptimizationRouterTransport,
    expected_fallback: bool | None,
    status: TokenOptimizationRouterStatus,
    prompt_report,
) -> None:
    result = _run_controlled_case(
        tmp_path,
        router_result=_controlled_router_result(
            transport=transport,
            status=status,
            prompt_assembly_report=prompt_report,
            configuration_id=(
                TokenOptimizationRouterConfigurationId.EXACT_ONLY
                if status is TokenOptimizationRouterStatus.ROUTED
                else None
            ),
            reason_code=(
                TokenOptimizationRouterReasonCode.EXACT_DUPLICATES
                if status is TokenOptimizationRouterStatus.ROUTED
                else None
            ),
            risk=(
                TokenOptimizationRouterRisk.LOW
                if status is TokenOptimizationRouterStatus.ROUTED
                else None
            ),
            review_required=False if status is TokenOptimizationRouterStatus.ROUTED else None,
            confidence=1.0 if status is TokenOptimizationRouterStatus.ROUTED else None,
        ),
        pipeline_result=_controlled_pipeline_result(
            completed=True,
            fallback_used=False,
        ),
    )

    evidence = result.cases[0].router_evidence
    assert evidence.transport == transport.value
    assert evidence.structured_output_fallback_used is expected_fallback
    if transport is TokenOptimizationRouterTransport.UNSUPPORTED:
        assert evidence.review_required is None
        assert evidence.confidence is None


def test_runner_pipeline_evidence_uses_receipt_and_not_measurements(
    tmp_path: Path,
) -> None:
    validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.FAILED,
        regions_checked=1,
        regions_preserved=0,
        regions_failed=1,
    )
    pipeline = _controlled_pipeline_result(
        completed=True,
        fallback_used=True,
        validation=validation,
        receipt_metadata={
            "validation_status": "failed",
            "validation_reason_code": "PROTECTED_REGION_VALIDATION_FAILED",
            "required_failure_layer_id": "controlled.layer",
        },
        baseline_tokens=100,
        optimized_tokens=100,
    )
    result = _run_controlled_case(
        tmp_path,
        router_result=_controlled_router_result(prompt_assembly_report=_prompt_report()),
        pipeline_result=pipeline,
    )

    case = result.cases[0]
    assert case.pipeline_evidence.completed is True
    assert case.pipeline_evidence.fallback_applied is True
    assert case.pipeline_evidence.validation_status == "failed"
    assert (
        case.pipeline_evidence.validation_reason_code
        == "PROTECTED_REGION_VALIDATION_FAILED"
    )
    assert case.pipeline_evidence.required_layer_failure == "controlled.layer"
    assert case.pipeline_evidence.receipt_completion_status is True
    assert case.baseline_measurement.value == 100
    assert case.optimized_measurement.value == 100


def test_runner_pipeline_incomplete_and_exception_leave_unavailable_evidence(
    tmp_path: Path,
) -> None:
    incomplete = _run_controlled_case(
        tmp_path / "incomplete",
        router_result=_controlled_router_result(prompt_assembly_report=_prompt_report()),
        pipeline_result=_controlled_pipeline_result(
            completed=False,
            fallback_used=True,
            receipt_metadata={"required_failure_layer_id": "required.layer"},
        ),
    )
    incomplete_evidence = incomplete.cases[0].pipeline_evidence
    assert incomplete.cases[0].status == "failed"
    assert incomplete_evidence.completed is False
    assert incomplete_evidence.fallback_applied is True
    assert incomplete_evidence.required_layer_failure == "required.layer"

    class _FailingPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            raise RuntimeError("synthetic pipeline failure")

    failed = _run_controlled_case(
        tmp_path / "exception",
        router_result=_controlled_router_result(prompt_assembly_report=_prompt_report()),
        pipeline_runner_factory=_FailingPipeline,
    )
    failed_evidence = failed.cases[0].pipeline_evidence
    assert failed.cases[0].error_reason_code == "PIPELINE_EXECUTION_FAILED"
    assert failed_evidence.completed is None
    assert failed_evidence.fallback_applied is None
    assert failed_evidence.validation_status is None
    assert failed_evidence.validation_reason_code is None
    assert failed_evidence.required_layer_failure is None


def test_protected_region_evidence_covers_passed_failed_and_redaction_safe_cases(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    empty_request = config.cases[0].request
    empty_evidence = _protected_region_evidence(empty_request, None)
    assert empty_evidence.input_protected_region_count == 0
    assert empty_evidence.protected_region_validation_status == "not_applicable"
    assert empty_evidence.input_identity_digest is None
    assert empty_evidence.preserved_identity_digest is None

    regions = (
        ProtectedRegion(ProtectedRegionKind.URL, "https://synthetic.example"),
        ProtectedRegion(ProtectedRegionKind.PATH, "/synthetic/path"),
        ProtectedRegion(ProtectedRegionKind.IDENTIFIER, "synthetic-id"),
    )
    protected_request = replace(empty_request, protected_regions=regions)
    passed_validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.PASSED,
        regions_checked=3,
        regions_preserved=3,
    )
    passed = _protected_region_evidence(
        protected_request,
        _controlled_pipeline_result(
            completed=True,
            fallback_used=False,
            validation=passed_validation,
        ),
    )
    assert passed.input_protected_region_count == 3
    assert passed.validated_protected_region_count == 3
    assert passed.preserved_protected_region_count == 3
    assert passed.protected_region_validation_status == "passed"
    assert passed.input_identity_digest == passed.preserved_identity_digest

    failed = _protected_region_evidence(
        protected_request,
        _controlled_pipeline_result(
            completed=True,
            fallback_used=True,
            validation=ProtectedRegionValidationResult(
                status=ProtectedRegionValidationStatus.FAILED,
                regions_checked=3,
                regions_preserved=2,
                regions_failed=1,
            ),
        ),
    )
    assert failed.preserved_protected_region_count == 2
    assert failed.protected_region_validation_status == "failed"
    assert failed.preserved_identity_digest is None
    dumped = json.dumps(
        {
            "repr": repr(failed),
            "input_digest": failed.input_identity_digest,
            "preserved_digest": failed.preserved_identity_digest,
        }
    )
    assert "synthetic.example" not in dumped
    assert "/synthetic/path" not in dumped


def test_protected_identity_digest_changes_with_kind_order_and_value(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    request = config.cases[0].request
    first = (
        ProtectedRegion(ProtectedRegionKind.URL, "https://synthetic.example"),
        ProtectedRegion(ProtectedRegionKind.PATH, "/synthetic/path"),
    )
    changed_kind = (
        ProtectedRegion(ProtectedRegionKind.IDENTIFIER, "https://synthetic.example"),
        first[1],
    )
    changed_order = tuple(reversed(first))
    changed_value = (
        first[0],
        ProtectedRegion(ProtectedRegionKind.PATH, "/synthetic/other"),
    )
    assert _protected_identity_digest(first) == _protected_identity_digest(first)
    assert _protected_identity_digest(first) != _protected_identity_digest(changed_kind)
    assert _protected_identity_digest(first) != _protected_identity_digest(changed_order)
    assert _protected_identity_digest(first) != _protected_identity_digest(changed_value)
    evidence = _protected_region_evidence(
        replace(request, protected_regions=first),
        None,
    )
    assert evidence.input_identity_digest == _protected_identity_digest(first)


def test_prefix_identity_uses_existing_token_10b_assembly_contract() -> None:
    def assembly(stable: str, tail: str, tools_schema=(), previous_state=None):
        return assemble_cache_stable_prompt(
            stable_prefix_blocks=(
                PromptAssemblyMessageBlock(
                    block_id="stable",
                    message=ChatMessage(role="system", content=stable),
                ),
            ),
            dynamic_tail=(ChatMessage(role="user", content=tail),),
            tools_schema=tools_schema,
            previous_state=previous_state,
        )

    first = assembly("stable-policy", "tail-a")
    changed_tail = assembly("stable-policy", "tail-b", previous_state=first.state)
    changed_prefix = assembly("changed-policy", "tail-a", previous_state=first.state)
    assert _prefix_identity_evidence(first.report).stable_prefix_identity == (
        _prefix_identity_evidence(changed_tail.report).stable_prefix_identity
    )
    assert _prefix_identity_evidence(first.report).stable_prefix_identity != (
        _prefix_identity_evidence(changed_prefix.report).stable_prefix_identity
    )

    tools = (
        {
            "type": "function",
            "function": {
                "name": "alpha.tool",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "beta.tool",
                "parameters": {"type": "object"},
            },
        },
    )
    reversed_tools = tuple(reversed(tools))
    with_tools = assembly("stable-policy", "tail-a", tools)
    with_reordered_tools = assembly("stable-policy", "tail-a", reversed_tools)
    with_identical_tools = assembly("stable-policy", "tail-a", tools)
    assert with_tools.report.tool_envelope_hash != with_reordered_tools.report.tool_envelope_hash
    assert with_tools.report.tool_envelope_hash == with_identical_tools.report.tool_envelope_hash
    assert with_tools.report.tool_envelope_hash is not None

    same_keys_different_order = (
        {
            "type": "function",
            "function": {
                "parameters": {"type": "object"},
                "name": "alpha.tool",
            },
        },
        {
            "type": "function",
            "function": {
                "parameters": {"type": "object"},
                "name": "beta.tool",
            },
        },
    )
    canonicalized = assembly("stable-policy", "tail-a", same_keys_different_order)
    assert canonicalized.report.tool_envelope_hash == with_tools.report.tool_envelope_hash

    unavailable = _prefix_identity_evidence(None)
    assert unavailable.identity_available is False
    assert unavailable.stable_prefix_identity is None
    assert unavailable.tool_schema_hash is None
    assert unavailable.message_envelope_hash is None
    assert unavailable.identity_contract_version is None
