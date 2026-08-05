from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.runtime.token_optimization.proofs.contracts import (
    ProofArtifactRef,
    ProofMeasurement,
    ProofPipelineEvidence,
    ProofPrefixIdentityEvidence,
    ProofProtectedRegionEvidence,
    ProofRouterEvidence,
    UniversalProofArtifactManifest,
    UniversalProofCaseResult,
    UniversalProofEnvironmentSummary,
    UniversalProofRunResult,
)
from intergrax.runtime.token_optimization.proofs.evaluation_contracts import (
    CacheAttribution,
    CacheEvidenceRole,
    CacheExpectation,
    CacheExpectationMode,
    CorpusCase,
    EvaluationConfiguration,
    EvaluationConfigurationError,
    GateStatus,
    MeasurementExpectation,
    MeasurementRequirement,
    PipelineExpectation,
    PrefixExpectation,
    ProofCorpus,
    ProtectedRegionExpectation,
    ProviderCacheEvidence,
    RouterExpectation,
)
from intergrax.runtime.token_optimization.proofs.evaluator import (
    EVALUATION_GATE_IDS,
    UniversalProofEvaluator,
)

_DIGEST = "a" * 64


def _fixture() -> tuple[UniversalProofRunResult, ProofCorpus, EvaluationConfiguration]:
    corpus_case = CorpusCase(
        case_id="case-one",
        category="short_clean_prompt",
        description="Synthetic safe case.",
        input_case_id="input",
        safe_tags=("synthetic",),
        source_type=None,
        policy_enabled=None,
        policy_profile=None,
        allow_lossy=None,
        protected_regions=(),
        router=RouterExpectation(
            allowed_statuses=frozenset({"routed"}),
            allowed_configuration_ids=frozenset({"exact_only"}),
            allowed_reason_codes=frozenset({"exact_duplicates"}),
            review_required=False,
            confidence_minimum=0.0,
            confidence_maximum=1.0,
            allowed_risk=frozenset({"low"}),
            allowed_transport=frozenset({"native_tools"}),
            structured_output_fallback=False,
        ),
        pipeline=PipelineExpectation(
            expected_completion=True,
            required_layer_ids=frozenset({"layer.one"}),
            allowed_layer_ids=frozenset({"layer.one"}),
            expected_fallback=False,
            required_layer_failure_expected=False,
        ),
        protected=ProtectedRegionExpectation(
            expected_input_count=0,
            expected_preserved_count=0,
            expected_validation_status="not_applicable",
        ),
        measurement=MeasurementExpectation(
            baseline=MeasurementRequirement.REQUIRED,
            optimized=MeasurementRequirement.REQUIRED,
            ordering_required=True,
        ),
        prefix=PrefixExpectation(identity_required=True),
        cache=CacheExpectation(mode=CacheExpectationMode.WARM_EXPECTED),
    )
    case = UniversalProofCaseResult(
        case_id="case-one",
        status="completed",
        router_status="routed",
        router_reason="exact_duplicates",
        selected_configuration_id="exact_only",
        pipeline_status="completed",
        applied_layer_ids=("layer.one",),
        baseline_measurement=ProofMeasurement(available=True, value=100),
        optimized_measurement=ProofMeasurement(available=True, value=70),
        router_evidence=ProofRouterEvidence(
            status="routed",
            configuration_id="exact_only",
            reason_code="exact_duplicates",
            review_required=False,
            confidence=1.0,
            risk="low",
            transport="native_tools",
            structured_output_fallback_used=False,
        ),
        pipeline_evidence=ProofPipelineEvidence(
            completed=True,
            fallback_applied=False,
        ),
        protected_region_evidence=ProofProtectedRegionEvidence(
            protected_region_validation_status="not_applicable",
        ),
        prefix_identity_evidence=ProofPrefixIdentityEvidence(
            identity_available=True,
            stable_prefix_identity=_DIGEST,
            tool_schema_hash=_DIGEST,
            identity_contract_version="TOKEN-10B",
        ),
    )
    run = UniversalProofRunResult(
        schema_version="token-optimization-proof.v1",
        proof_id="proof",
        run_id="run",
        run_mode="offline_smoke",
        started_at=datetime(2026, 8, 5, tzinfo=UTC),
        completed_at=datetime(2026, 8, 5, tzinfo=UTC),
        adapter_id="offline",
        model="synthetic-model",
        case_count=1,
        completed_count=1,
        failed_count=0,
        cases=(case,),
        environment=UniversalProofEnvironmentSummary(
            provider="vllm",
            model="synthetic-model",
            adapter_available=True,
            network_required=False,
        ),
        artifact_manifest=UniversalProofArtifactManifest(
            files=(
                ProofArtifactRef("run.json", _DIGEST),
                ProofArtifactRef("cases/case-one.json", _DIGEST),
            )
        ),
        success=True,
    )
    corpus = ProofCorpus(
        schema_version="token-optimization-proof-corpus.v1",
        corpus_id="corpus",
        evaluation_version="token-10g.v1",
        cases=(corpus_case,),
    )
    config = EvaluationConfiguration(
        schema_version="token-optimization-proof-evaluation.v1",
        evaluation_version="token-10g.v1",
        required_gate_ids=EVALUATION_GATE_IDS,
        unavailable_allowed_gate_ids=frozenset(
            {"WARM_CACHE_REUSE", "CHANGED_PREFIX_NEGATIVE_CONTROL"}
        ),
    )
    return run, corpus, config


def _cache() -> ProviderCacheEvidence:
    return ProviderCacheEvidence(
        case_id="case-one",
        provider="vllm",
        model="synthetic-model",
        stable_prefix_identity=_DIGEST,
        prompt_token_count=100,
        cached_prompt_token_count=80,
        cache_attribution=CacheAttribution.REUSE_CONFIRMED,
        role=CacheEvidenceRole.WARM_EXPECTED,
        reason_code="PROVIDER_REPORTED_REUSE",
    )


def test_all_required_gates_pass_from_recorded_evidence() -> None:
    run, corpus, config = _fixture()
    evaluation = UniversalProofEvaluator().evaluate(
        run, corpus, config, cache_evidence=(_cache(),), evaluation_id="fixed"
    )

    assert evaluation.success is True
    assert not [
        gate
        for case in evaluation.cases
        for gate in case.gates
        if gate.required and gate.status is GateStatus.FAIL
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda run: replace(
            run, cases=(replace(run.cases[0], router_status="blocked"),)
        ),
        lambda run: replace(
            run,
            cases=(
                replace(
                    run.cases[0],
                    optimized_measurement=ProofMeasurement(available=True, value=101),
                ),
            ),
        ),
        lambda run: replace(
            run,
            cases=(
                replace(
                    run.cases[0],
                    router_evidence=replace(
                        run.cases[0].router_evidence,
                        configuration_id="other",
                    ),
                ),
            ),
        ),
    ],
)
def test_contradictory_router_or_measurement_evidence_fails_closed(mutation) -> None:
    run, corpus, config = _fixture()
    evaluation = UniversalProofEvaluator().evaluate(
        mutation(run), corpus, config, cache_evidence=(_cache(),)
    )
    assert evaluation.success is False
    assert any(
        gate.status is GateStatus.FAIL
        for case in evaluation.cases
        for gate in case.gates
    )


def test_missing_prefix_and_latency_only_cache_are_not_pass() -> None:
    run, corpus, config = _fixture()
    unavailable = replace(
        run,
        cases=(
            replace(
                run.cases[0],
                prefix_identity_evidence=ProofPrefixIdentityEvidence(),
            ),
        ),
    )
    evaluation = UniversalProofEvaluator().evaluate(unavailable, corpus, config)
    assert evaluation.success is False
    assert any(
        gate.gate_id == "PREFIX_IDENTITY_AVAILABLE"
        and gate.status is GateStatus.UNAVAILABLE
        for case in evaluation.cases
        for gate in case.gates
    )

    latency_only = replace(_cache(), cache_attribution=CacheAttribution.LATENCY_ONLY)
    evaluation = UniversalProofEvaluator().evaluate(
        run, corpus, config, cache_evidence=(latency_only,)
    )
    assert any(
        gate.gate_id == "WARM_CACHE_REUSE" and gate.status is GateStatus.FAIL
        for case in evaluation.cases
        for gate in case.gates
    )


def test_unknown_gate_is_a_configuration_error() -> None:
    run, corpus, config = _fixture()
    with pytest.raises(EvaluationConfigurationError, match="UNKNOWN_GATE_ID"):
        UniversalProofEvaluator().evaluate(
            run,
            corpus,
            replace(
                config,
                required_gate_ids=("UNKNOWN_GATE",),
                unavailable_allowed_gate_ids=frozenset(),
            ),
        )


def test_structural_guard_keeps_evaluator_evaluation_only() -> None:
    source = (
        Path(
            "intergrax/runtime/token_optimization/proofs/evaluator.py"
        ).read_text(encoding="utf-8")
    )
    assert "TokenOptimizationLLMRouter" not in source
    assert "TokenOptimizationPipelineRunner" not in source
    assert "compute_openai_tools_schema_hash" not in source
    assert "import hashlib" not in source
