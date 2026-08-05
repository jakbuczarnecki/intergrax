from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from intergrax.runtime.token_optimization.proofs.config import (
    load_universal_token_optimization_proof_config,
)
from intergrax.runtime.token_optimization.proofs.runner import (
    UniversalTokenOptimizationProofRunner,
)


def _config(tmp_path: Path):
    path = tmp_path / "proof.toml"
    path.write_text(
        """
schema_version = "token-optimization-proof.v1"
proof_id = "runner-proof"
run_mode = "offline_smoke"

[adapter]
adapter_id = "offline"
provider = "vllm"
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
