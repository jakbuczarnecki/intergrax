from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    _measurements_from_pipeline,
)


def _config(tmp_path: Path, *, provider: str = "vllm"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "proof.toml"
    path.write_text(
        f"""
schema_version = "token-optimization-proof.v1"
proof_id = "runner-proof"
run_mode = "offline_smoke"

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


@pytest.mark.parametrize("provider", ("vllm", "ollama", "openai"))
def test_loader_accepts_canonical_provider_ids_without_network_call(
    tmp_path: Path,
    provider: str,
) -> None:
    config = _config(tmp_path / provider, provider=provider)
    assert config.adapter.provider == provider


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
    assert dict(LLMAdapterRegistry._factories) == before
