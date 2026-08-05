from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

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
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationRouterConfigurationId,
    TokenOptimizationRouterReasonCode,
    TokenOptimizationRouterRisk,
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
