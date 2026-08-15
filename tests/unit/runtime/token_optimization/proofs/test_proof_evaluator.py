from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
)
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
    EvaluationGateRequirement,
    EvaluationProfile,
    GateStatus,
    MeasurementExpectation,
    MeasurementRequirement,
    PipelineExecutionExpectation,
    PipelineExpectation,
    PrefixExpectation,
    ProofCorpus,
    ProtectedRegionExpectation,
    ProviderCacheEvidence,
    RouterExpectation,
    load_evaluation_config,
)
from intergrax.runtime.token_optimization.proofs.evaluator import (
    EVALUATION_GATE_IDS,
    UniversalProofEvaluator,
)

_DIGEST = "a" * 64
HASH_A = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
HASH_B = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
TOOL_HASH_AB = "1111222233334444555566667777888811112222333344445555666677778888"
TOOL_HASH_BA = "9999aaaabbbbccccddddeeeeffff00009999aaaabbbbccccddddeeeeffff0000"


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
            expected_execution=PipelineExecutionExpectation.COMPLETED,
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
            model_configuration_id="exact_only",
            model_reason_code="exact_duplicates",
            model_risk="low",
            model_review_required=False,
        ),
        pipeline_evidence=ProofPipelineEvidence(
            completed=True,
            fallback_applied=False,
            receipt_completion_status=True,
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


def _offline_config() -> EvaluationConfiguration:
    requirements = {
        gate_id: EvaluationGateRequirement.REQUIRED for gate_id in EVALUATION_GATE_IDS
    }
    requirements.update(
        {
            "ROUTER_STATUS": EvaluationGateRequirement.NOT_APPLICABLE,
            "ROUTER_CONFIGURATION": EvaluationGateRequirement.NOT_APPLICABLE,
            "ROUTER_REASON": EvaluationGateRequirement.NOT_APPLICABLE,
            "ROUTER_REVIEW_REQUIREMENT": EvaluationGateRequirement.NOT_APPLICABLE,
            "ROUTER_CONFIDENCE": EvaluationGateRequirement.NOT_APPLICABLE,
            "ROUTER_TRANSPORT": EvaluationGateRequirement.NOT_APPLICABLE,
            "MODEL_ROUTER_CONFIGURATION": EvaluationGateRequirement.NOT_APPLICABLE,
            "MODEL_ROUTER_REASON": EvaluationGateRequirement.NOT_APPLICABLE,
            "MODEL_ROUTER_RISK": EvaluationGateRequirement.NOT_APPLICABLE,
            "MODEL_ROUTER_REVIEW_REQUIREMENT": (
                EvaluationGateRequirement.NOT_APPLICABLE
            ),
            "FINAL_ROUTER_STATUS": EvaluationGateRequirement.NOT_APPLICABLE,
            "FINAL_POLICY_ENFORCEMENT": EvaluationGateRequirement.NOT_APPLICABLE,
            "PIPELINE_COMPLETION": EvaluationGateRequirement.NOT_APPLICABLE,
            "PIPELINE_REQUIRED_LAYERS": EvaluationGateRequirement.NOT_APPLICABLE,
            "PIPELINE_FORBIDDEN_LAYERS": EvaluationGateRequirement.NOT_APPLICABLE,
            "PIPELINE_FALLBACK": EvaluationGateRequirement.NOT_APPLICABLE,
            "PIPELINE_VALIDATION": EvaluationGateRequirement.NOT_APPLICABLE,
            "PIPELINE_REQUIRED_FAILURE": EvaluationGateRequirement.NOT_APPLICABLE,
            "BASELINE_MEASUREMENT": EvaluationGateRequirement.OPTIONAL,
            "OPTIMIZED_MEASUREMENT": EvaluationGateRequirement.OPTIONAL,
            "MEASUREMENT_ORDERING": EvaluationGateRequirement.OPTIONAL,
            "PREFIX_CHANGED_CONTROL": EvaluationGateRequirement.NOT_APPLICABLE,
            "TOOL_ENVELOPE_IDENTITY": EvaluationGateRequirement.NOT_APPLICABLE,
            "WARM_CACHE_REUSE": EvaluationGateRequirement.UNAVAILABLE_ALLOWED,
            "CHANGED_PREFIX_NEGATIVE_CONTROL": (
                EvaluationGateRequirement.UNAVAILABLE_ALLOWED
            ),
        }
    )
    return EvaluationConfiguration(
        schema_version="token-optimization-proof-evaluation.v1",
        evaluation_version="token-10g.v1",
        required_gate_ids=(),
        unavailable_allowed_gate_ids=frozenset(),
        profile=EvaluationProfile.OFFLINE_COMPOSITION,
        gate_requirements=requirements,
        configured_offline_decision="exact_only",
    )


def _offline_executed_run() -> UniversalProofRunResult:
    run, _, _ = _fixture()
    case = run.cases[0]
    return replace(
        run,
        cases=(
            replace(
                case,
                router_evidence=replace(
                    case.router_evidence,
                    transport="structured_output",
                    structured_output_fallback_used=True,
                ),
            ),
        ),
    )


def _evaluate_offline(run: UniversalProofRunResult):
    _, corpus, _ = _fixture()
    return UniversalProofEvaluator().evaluate(
        run,
        corpus,
        _offline_config(),
        evaluation_id="offline-integrity",
    )


def _terminal_non_execution_fixture(
    expected_execution: PipelineExecutionExpectation,
) -> tuple[UniversalProofRunResult, ProofCorpus, EvaluationConfiguration]:
    run, corpus, config = _fixture()
    terminal_case = replace(
        run.cases[0],
        status="failed",
        router_status="blocked",
        router_reason="policy_disabled",
        selected_configuration_id=None,
        pipeline_status="not_started",
        applied_layer_ids=(),
        router_evidence=ProofRouterEvidence(status="blocked"),
        pipeline_evidence=ProofPipelineEvidence(),
    )
    corpus_case = replace(
        corpus.cases[0],
        router=RouterExpectation(
            allowed_statuses=frozenset({"blocked"}),
            allowed_reason_codes=frozenset({"policy_disabled"}),
        ),
        pipeline=replace(
            corpus.cases[0].pipeline,
            expected_execution=expected_execution,
            required_layer_ids=frozenset(),
            allowed_layer_ids=frozenset(),
            expected_fallback=None,
            expected_validation_status=None,
            required_layer_failure_expected=None,
        ),
    )
    return (
        replace(run, cases=(terminal_case,)),
        replace(corpus, cases=(corpus_case,)),
        config,
    )


def _gate(evaluation, gate_id: str):
    return next(
        gate
        for case in evaluation.cases
        for gate in case.gates
        if gate.gate_id == gate_id
    )


def test_security_override_separates_model_failure_from_runtime_safety() -> None:
    run, corpus, config = _fixture()
    model_case = replace(
        run.cases[0],
        status="failed",
        router_status="review_required",
        router_reason="protected_regions_require_review",
        pipeline_status="not_started",
        applied_layer_ids=(),
        router_evidence=replace(
            run.cases[0].router_evidence,
            status="review_required",
            configuration_id="exact_only",
            reason_code="protected_or_high_risk",
            risk="high",
            review_required=True,
            model_configuration_id="exact_only",
            model_reason_code="clean_no_op",
            model_risk="low",
            model_review_required=False,
            policy_override_applied=True,
            policy_override_reason="security_warning_requires_review",
        ),
        pipeline_evidence=ProofPipelineEvidence(),
    )
    security_case = replace(
        corpus.cases[0],
        protected_regions=(
            ProtectedRegion(
                kind=ProtectedRegionKind.SECURITY_WARNING,
                value="SYNTHETIC_SECURITY_WARNING",
            ),
        ),
        router=replace(
            corpus.cases[0].router,
            allowed_statuses=frozenset({"review_required"}),
            allowed_configuration_ids=frozenset({"exact_only"}),
            allowed_reason_codes=frozenset({"protected_or_high_risk"}),
            review_required=True,
            allowed_risk=frozenset({"high"}),
        ),
        pipeline=replace(
            corpus.cases[0].pipeline,
            expected_execution=PipelineExecutionExpectation.NOT_STARTED,
            required_layer_ids=frozenset(),
            allowed_layer_ids=frozenset(),
        ),
    )
    evaluation = UniversalProofEvaluator().evaluate(
        replace(run, cases=(model_case,), completed_count=0, failed_count=1),
        replace(corpus, cases=(security_case,)),
        config,
        evaluation_id="security-override-separation",
    )

    assert _gate(evaluation, "MODEL_ROUTER_RISK").status is GateStatus.FAIL
    assert (
        _gate(evaluation, "MODEL_ROUTER_REVIEW_REQUIREMENT").status is GateStatus.FAIL
    )
    assert _gate(evaluation, "FINAL_POLICY_ENFORCEMENT").status is GateStatus.PASS


def _changed_prefix_fixture() -> tuple[
    UniversalProofRunResult, ProofCorpus, EvaluationConfiguration
]:
    run, corpus, config = _fixture()
    first_case = corpus.cases[0]
    second_case = replace(
        first_case,
        case_id="case-two",
        input_case_id="input-two",
        prefix=PrefixExpectation(
            identity_required=True,
            different_from_case_id="case-one",
        ),
        cache=CacheExpectation(
            mode=CacheExpectationMode.CHANGED_PREFIX_NEGATIVE_CONTROL,
            same_as_case_id="case-one",
        ),
    )
    second_result = replace(
        run.cases[0],
        case_id="case-two",
        prefix_identity_evidence=replace(
            run.cases[0].prefix_identity_evidence,
            stable_prefix_identity="b" * 64,
        ),
    )
    run = replace(
        run,
        case_count=2,
        completed_count=2,
        cases=(run.cases[0], second_result),
        artifact_manifest=UniversalProofArtifactManifest(
            files=(
                ProofArtifactRef("run.json", _DIGEST),
                ProofArtifactRef("cases/case-one.json", _DIGEST),
                ProofArtifactRef("cases/case-two.json", _DIGEST),
            )
        ),
    )
    corpus = replace(corpus, cases=(first_case, second_case))
    return run, corpus, config


def _evaluate_only_identity_fixture() -> tuple[
    UniversalProofRunResult, ProofCorpus, EvaluationConfiguration
]:
    run, corpus, config = _fixture()
    base_corpus_case = replace(
        corpus.cases[0],
        cache=CacheExpectation(mode=CacheExpectationMode.NOT_APPLICABLE),
    )
    cases = (
        (
            "stable-prefix-a",
            HASH_A,
            None,
            PrefixExpectation(identity_required=True),
        ),
        (
            "stable-prefix-b",
            HASH_A,
            None,
            PrefixExpectation(
                identity_required=True,
                same_as_case_id="stable-prefix-a",
            ),
        ),
        (
            "changed-prefix",
            HASH_B,
            None,
            PrefixExpectation(
                identity_required=True,
                different_from_case_id="stable-prefix-a",
            ),
        ),
        (
            "tools-alpha-beta",
            HASH_A,
            TOOL_HASH_AB,
            PrefixExpectation(identity_required=True),
        ),
        (
            "tools-beta-alpha",
            HASH_B,
            TOOL_HASH_BA,
            PrefixExpectation(
                identity_required=True,
                different_from_case_id="tools-alpha-beta",
                tool_schema_identity="different",
            ),
        ),
        (
            "tools-inner-reordered",
            HASH_A,
            TOOL_HASH_AB,
            PrefixExpectation(
                identity_required=True,
                same_as_case_id="tools-alpha-beta",
                tool_schema_identity="same",
            ),
        ),
    )
    corpus_cases = tuple(
        replace(
            base_corpus_case,
            case_id=case_id,
            input_case_id=case_id,
            prefix=prefix,
        )
        for case_id, _, _, prefix in cases
    )
    result_cases = tuple(
        replace(
            run.cases[0],
            case_id=case_id,
            prefix_identity_evidence=ProofPrefixIdentityEvidence(
                identity_available=True,
                stable_prefix_identity=stable_prefix_identity,
                tool_schema_hash=tool_schema_hash,
                identity_contract_version="TOKEN-10B",
            ),
        )
        for case_id, stable_prefix_identity, tool_schema_hash, _ in cases
    )
    run = replace(
        run,
        case_count=len(result_cases),
        completed_count=len(result_cases),
        cases=result_cases,
        artifact_manifest=UniversalProofArtifactManifest(
            files=tuple(
                [ProofArtifactRef("run.json", _DIGEST)]
                + [
                    ProofArtifactRef(f"cases/{case_id}.json", _DIGEST)
                    for case_id, _, _, _ in cases
                ]
            )
        ),
    )
    corpus = replace(corpus, cases=corpus_cases)
    return run, corpus, config


def test_evaluate_only_identity_fixture_has_explicit_safe_digests() -> None:
    run, corpus, _ = _evaluate_only_identity_fixture()
    evidence = {case.case_id: case.prefix_identity_evidence for case in run.cases}

    assert evidence["stable-prefix-a"].stable_prefix_identity == HASH_A
    assert evidence["stable-prefix-b"].stable_prefix_identity == HASH_A
    assert evidence["changed-prefix"].stable_prefix_identity == HASH_B
    assert evidence["tools-alpha-beta"].tool_schema_hash == TOOL_HASH_AB
    assert evidence["tools-beta-alpha"].tool_schema_hash == TOOL_HASH_BA
    assert evidence["tools-inner-reordered"].tool_schema_hash == TOOL_HASH_AB
    assert HASH_A != HASH_B
    assert TOOL_HASH_AB != TOOL_HASH_BA
    assert (
        evidence["tools-inner-reordered"].tool_schema_hash
        == evidence["tools-alpha-beta"].tool_schema_hash
    )
    assert all(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        for item in evidence.values()
        for value in (
            item.stable_prefix_identity,
            item.tool_schema_hash,
        )
        if value is not None
    )
    assert len(corpus.cases) == 6


def test_evaluate_only_identity_fixture_passes_without_engine_rerun() -> None:
    run, corpus, config = _evaluate_only_identity_fixture()
    evaluation = UniversalProofEvaluator().evaluate(
        run, corpus, config, evaluation_id="identity-fixtures"
    )

    assert evaluation.success is True
    assert not [
        gate
        for case in evaluation.cases
        for gate in case.gates
        if gate.required and gate.status is GateStatus.FAIL
    ]


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
    "expected_execution",
    [
        PipelineExecutionExpectation.COMPLETED,
        PipelineExecutionExpectation.FAILED,
        PipelineExecutionExpectation.NOT_STARTED,
    ],
)
def test_pipeline_completion_expectation_accepts_matching_typed_evidence(
    expected_execution: PipelineExecutionExpectation,
) -> None:
    if expected_execution is PipelineExecutionExpectation.NOT_STARTED:
        run, corpus, config = _terminal_non_execution_fixture(expected_execution)
    else:
        run, corpus, config = _fixture()
        completed = expected_execution is PipelineExecutionExpectation.COMPLETED
        case = replace(
            run.cases[0],
            status="completed" if completed else "failed",
            pipeline_status=expected_execution.value,
            pipeline_evidence=replace(
                run.cases[0].pipeline_evidence,
                completed=completed,
                receipt_completion_status=completed,
            ),
        )
        run = replace(run, cases=(case,))
        corpus = replace(
            corpus,
            cases=(
                replace(
                    corpus.cases[0],
                    pipeline=replace(
                        corpus.cases[0].pipeline,
                        expected_execution=expected_execution,
                    ),
                ),
            ),
        )

    evaluation = UniversalProofEvaluator().evaluate(
        run,
        corpus,
        config,
        cache_evidence=(_cache(),),
        evaluation_id=f"typed-{expected_execution.value}",
    )

    assert _gate(evaluation, "PIPELINE_COMPLETION").status is GateStatus.PASS


@pytest.mark.parametrize(
    "mutation",
    [
        lambda case: replace(case, pipeline_status="completed"),
        lambda case: replace(
            case,
            pipeline_status="failed",
            pipeline_evidence=replace(
                case.pipeline_evidence,
                completed=False,
                receipt_completion_status=False,
            ),
        ),
        lambda case: replace(
            case,
            pipeline_evidence=replace(case.pipeline_evidence, completed=False),
        ),
        lambda case: replace(
            case,
            pipeline_evidence=replace(case.pipeline_evidence, completed=True),
        ),
        lambda case: replace(
            case,
            pipeline_evidence=replace(
                case.pipeline_evidence,
                receipt_completion_status=True,
            ),
        ),
        lambda case: replace(case, applied_layer_ids=("layer.one",)),
        lambda case: replace(
            case,
            pipeline_evidence=replace(case.pipeline_evidence, fallback_applied=True),
        ),
        lambda case: replace(
            case,
            pipeline_evidence=replace(
                case.pipeline_evidence, validation_status="passed"
            ),
        ),
        lambda case: replace(
            case,
            pipeline_evidence=replace(
                case.pipeline_evidence,
                required_layer_failure="layer.one",
            ),
        ),
        lambda case: replace(case, router_reason=None),
        lambda case: replace(case, router_reason="unknown_terminal_reason"),
    ],
)
def test_not_started_expectation_rejects_execution_or_side_effects(mutation) -> None:
    run, corpus, config = _terminal_non_execution_fixture(
        PipelineExecutionExpectation.NOT_STARTED
    )
    evaluation = UniversalProofEvaluator().evaluate(
        replace(run, cases=(mutation(run.cases[0]),)),
        corpus,
        config,
        cache_evidence=(_cache(),),
        evaluation_id="not-started-negative",
    )

    assert _gate(evaluation, "PIPELINE_COMPLETION").status is GateStatus.FAIL


@pytest.mark.parametrize(
    ("expected_execution", "actual_execution", "actual_completed"),
    [
        (
            PipelineExecutionExpectation.COMPLETED,
            "failed",
            False,
        ),
        (
            PipelineExecutionExpectation.FAILED,
            "not_started",
            None,
        ),
    ],
)
def test_completed_and_failed_expectations_reject_other_execution_states(
    expected_execution: PipelineExecutionExpectation,
    actual_execution: str,
    actual_completed: bool | None,
) -> None:
    run, corpus, config = _fixture()
    case = replace(
        run.cases[0],
        status="failed",
        pipeline_status=actual_execution,
        pipeline_evidence=replace(
            run.cases[0].pipeline_evidence,
            completed=actual_completed,
            receipt_completion_status=actual_completed,
        ),
    )
    corpus = replace(
        corpus,
        cases=(
            replace(
                corpus.cases[0],
                pipeline=replace(
                    corpus.cases[0].pipeline,
                    expected_execution=expected_execution,
                ),
            ),
        ),
    )

    evaluation = UniversalProofEvaluator().evaluate(
        replace(run, cases=(case,)),
        corpus,
        config,
        cache_evidence=(_cache(),),
        evaluation_id="typed-mismatch",
    )

    assert _gate(evaluation, "PIPELINE_COMPLETION").status is GateStatus.FAIL


def test_offline_profile_separates_composition_from_behavior() -> None:
    run, corpus, _ = _fixture()
    run = replace(
        run,
        cases=(
            replace(
                run.cases[0],
                router_evidence=replace(
                    run.cases[0].router_evidence,
                    transport="structured_output",
                    structured_output_fallback_used=True,
                ),
            ),
        ),
    )
    evaluation = UniversalProofEvaluator().evaluate(
        run,
        corpus,
        _offline_config(),
        evaluation_id="offline-composition",
    )

    assert evaluation.profile is EvaluationProfile.OFFLINE_COMPOSITION
    assert evaluation.success is True
    assert not [
        gate
        for case in evaluation.cases
        for gate in case.gates
        if gate.required and gate.status is GateStatus.FAIL
    ]
    assert any(
        gate.gate_id == "ROUTER_CONFIGURATION"
        and gate.status is GateStatus.NOT_APPLICABLE
        and not gate.required
        for case in evaluation.cases
        for gate in case.gates
    )
    assert any(
        gate.gate_id == "ROUTER_EVIDENCE_INTEGRITY"
        and gate.status is GateStatus.PASS
        and gate.required
        for case in evaluation.cases
        for gate in case.gates
    )
    assert any(
        gate.gate_id == "WARM_CACHE_REUSE"
        and gate.status is GateStatus.UNAVAILABLE
        and gate.required
        for case in evaluation.cases
        for gate in case.gates
    )


def test_router_integrity_accepts_executed_offline_exact_only() -> None:
    evaluation = _evaluate_offline(_offline_executed_run())

    assert evaluation.success is True
    assert _gate(evaluation, "ROUTER_EVIDENCE_INTEGRITY").status is GateStatus.PASS


@pytest.mark.parametrize(
    "terminal_reason", ["policy_disabled", "source_type_not_supported"]
)
def test_router_integrity_accepts_coherent_terminal_non_execution(
    terminal_reason: str,
) -> None:
    run, _, _ = _fixture()
    terminal_case = replace(
        run.cases[0],
        status="failed",
        router_status="blocked",
        router_reason=terminal_reason,
        selected_configuration_id=None,
        pipeline_status="not_started",
        applied_layer_ids=(),
        router_evidence=ProofRouterEvidence(status="blocked"),
        pipeline_evidence=ProofPipelineEvidence(),
    )
    evaluation = _evaluate_offline(replace(run, cases=(terminal_case,)))

    assert evaluation.success is True
    assert _gate(evaluation, "ROUTER_EVIDENCE_INTEGRITY").status is GateStatus.PASS
    assert _gate(evaluation, "PIPELINE_EVIDENCE_INTEGRITY").status is GateStatus.PASS


@pytest.mark.parametrize(
    "mutation",
    [
        lambda run: replace(
            run,
            cases=(
                replace(
                    run.cases[0],
                    status="failed",
                    router_status="blocked",
                    router_reason="policy_disabled",
                    selected_configuration_id=None,
                    router_evidence=ProofRouterEvidence(
                        status="blocked", configuration_id="exact_only"
                    ),
                ),
            ),
        ),
        lambda run: replace(
            run,
            cases=(
                replace(
                    run.cases[0],
                    status="failed",
                    router_status="blocked",
                    router_reason="policy_disabled",
                    selected_configuration_id=None,
                    router_evidence=ProofRouterEvidence(
                        status="blocked", transport="native_tools"
                    ),
                ),
            ),
        ),
        lambda run: replace(
            run,
            cases=(
                replace(
                    run.cases[0],
                    status="failed",
                    router_status="blocked",
                    router_reason="policy_disabled",
                    selected_configuration_id=None,
                    router_evidence=ProofRouterEvidence(
                        status="blocked", confidence=0.9
                    ),
                ),
            ),
        ),
        lambda run: replace(
            _offline_executed_run(),
            cases=(
                replace(
                    run.cases[0],
                    router_evidence=replace(
                        run.cases[0].router_evidence,
                        configuration_id=None,
                    ),
                ),
            ),
        ),
        lambda run: replace(
            _offline_executed_run(),
            cases=(
                replace(
                    run.cases[0],
                    router_evidence=replace(
                        run.cases[0].router_evidence,
                        transport=None,
                        structured_output_fallback_used=None,
                    ),
                ),
            ),
        ),
        lambda run: replace(
            _offline_executed_run(),
            cases=(
                replace(
                    run.cases[0],
                    selected_configuration_id="other",
                    router_evidence=replace(
                        run.cases[0].router_evidence,
                        configuration_id="other",
                    ),
                ),
            ),
        ),
        lambda run: replace(
            _offline_executed_run(),
            cases=(
                replace(
                    run.cases[0],
                    router_status="blocked",
                    router_reason="policy_disabled",
                ),
            ),
        ),
        lambda run: replace(
            run,
            cases=(
                replace(
                    run.cases[0],
                    status="failed",
                    router_status="blocked",
                    router_reason="unknown_terminal_reason",
                    selected_configuration_id=None,
                    router_evidence=ProofRouterEvidence(status="blocked"),
                ),
            ),
        ),
    ],
)
def test_router_integrity_rejects_partial_or_contradictory_evidence(mutation) -> None:
    run = mutation(_offline_executed_run())
    evaluation = _evaluate_offline(run)

    assert evaluation.success is False
    assert _gate(evaluation, "ROUTER_EVIDENCE_INTEGRITY").status is GateStatus.FAIL


@pytest.mark.parametrize("pipeline_mutation", ["completed", "failed"])
def test_pipeline_integrity_accepts_coherent_execution(pipeline_mutation: str) -> None:
    run = _offline_executed_run()
    completed = pipeline_mutation == "completed"
    case = replace(
        run.cases[0],
        status="completed" if completed else "failed",
        pipeline_status=pipeline_mutation,
        pipeline_evidence=replace(
            run.cases[0].pipeline_evidence,
            completed=completed,
            receipt_completion_status=completed,
        ),
    )
    evaluation = _evaluate_offline(replace(run, cases=(case,)))

    assert _gate(evaluation, "PIPELINE_EVIDENCE_INTEGRITY").status is GateStatus.PASS


@pytest.mark.parametrize(
    "terminal_reason", ["policy_disabled", "source_type_not_supported"]
)
def test_pipeline_integrity_accepts_terminal_not_run(terminal_reason: str) -> None:
    run, _, _ = _fixture()
    case = replace(
        run.cases[0],
        status="failed",
        router_status="blocked",
        router_reason=terminal_reason,
        selected_configuration_id=None,
        pipeline_status="not_started",
        applied_layer_ids=(),
        router_evidence=ProofRouterEvidence(status="blocked"),
        pipeline_evidence=ProofPipelineEvidence(),
    )
    evaluation = _evaluate_offline(replace(run, cases=(case,)))

    assert _gate(evaluation, "PIPELINE_EVIDENCE_INTEGRITY").status is GateStatus.PASS


@pytest.mark.parametrize(
    "pipeline_mutation",
    [
        lambda case: replace(case, applied_layer_ids=("layer.one",)),
        lambda case: replace(
            case,
            pipeline_evidence=replace(
                case.pipeline_evidence,
                receipt_completion_status=True,
            ),
        ),
        lambda case: replace(
            case,
            pipeline_evidence=replace(
                case.pipeline_evidence,
                fallback_applied=True,
            ),
        ),
        lambda case: replace(
            case,
            pipeline_evidence=replace(
                case.pipeline_evidence,
                validation_status="passed",
            ),
        ),
        lambda case: replace(
            case,
            pipeline_status="completed",
            pipeline_evidence=replace(case.pipeline_evidence, completed=None),
        ),
        lambda case: replace(
            case,
            pipeline_status="failed",
            pipeline_evidence=replace(
                case.pipeline_evidence,
                completed=True,
                receipt_completion_status=True,
            ),
        ),
    ],
)
def test_pipeline_integrity_rejects_side_effects_or_completion_contradictions(
    pipeline_mutation,
) -> None:
    run, _, _ = _fixture()
    base_case = replace(
        run.cases[0],
        status="failed",
        router_status="blocked",
        router_reason="policy_disabled",
        selected_configuration_id=None,
        pipeline_status="not_started",
        applied_layer_ids=(),
        router_evidence=ProofRouterEvidence(status="blocked"),
        pipeline_evidence=ProofPipelineEvidence(),
    )
    case = pipeline_mutation(base_case)
    evaluation = _evaluate_offline(replace(run, cases=(case,)))

    assert evaluation.success is False
    assert _gate(evaluation, "PIPELINE_EVIDENCE_INTEGRITY").status is GateStatus.FAIL


def test_changed_prefix_cache_evidence_passes_from_typed_fixtures() -> None:
    run, corpus, config = _changed_prefix_fixture()
    changed = ProviderCacheEvidence(
        case_id="case-two",
        provider="vllm",
        model="synthetic-model",
        stable_prefix_identity="b" * 64,
        prompt_token_count=100,
        cached_prompt_token_count=0,
        cache_attribution=CacheAttribution.MISS_CONFIRMED,
        role=CacheEvidenceRole.CHANGED_PREFIX_NEGATIVE_CONTROL,
        reason_code="PROVIDER_REPORTED_MISS",
    )

    evaluation = UniversalProofEvaluator().evaluate(
        run,
        corpus,
        config,
        cache_evidence=(_cache(), changed),
    )

    assert evaluation.success is True
    gates = [
        gate
        for case in evaluation.cases
        for gate in case.gates
        if gate.case_id == "case-two"
    ]
    assert any(
        gate.gate_id == "CHANGED_PREFIX_NEGATIVE_CONTROL"
        and gate.status is GateStatus.PASS
        for gate in gates
    )


def test_conflicting_cache_evidence_fails_closed() -> None:
    run, corpus, config = _fixture()
    conflicting = replace(
        _cache(),
        cache_attribution=CacheAttribution.CONFLICTING,
    )

    evaluation = UniversalProofEvaluator().evaluate(
        run, corpus, config, cache_evidence=(conflicting,)
    )

    assert evaluation.success is False
    assert {
        gate.gate_id
        for case in evaluation.cases
        for gate in case.gates
        if gate.status is GateStatus.FAIL
    } >= {"WARM_CACHE_REUSE", "CHANGED_PREFIX_NEGATIVE_CONTROL"}


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


def _protected_fixture() -> tuple[
    UniversalProofRunResult, ProofCorpus, EvaluationConfiguration
]:
    run, corpus, config = _fixture()
    protected_expectation = ProtectedRegionExpectation(
        expected_input_count=1,
        expected_preserved_count=1,
        expected_validation_status="passed",
        digest_equality_required=True,
    )
    corpus = replace(
        corpus,
        cases=(replace(corpus.cases[0], protected=protected_expectation),),
    )
    evidence = ProofProtectedRegionEvidence(
        input_protected_region_count=1,
        validated_protected_region_count=1,
        preserved_protected_region_count=1,
        protected_region_validation_status="passed",
        input_identity_digest=_DIGEST,
        preserved_identity_digest=_DIGEST,
    )
    run = replace(
        run,
        cases=(replace(run.cases[0], protected_region_evidence=evidence),),
    )
    return run, corpus, config


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: replace(
            evidence,
            input_protected_region_count=0,
            validated_protected_region_count=0,
            preserved_protected_region_count=0,
            input_identity_digest=None,
            preserved_identity_digest=None,
        ),
        lambda evidence: replace(
            evidence,
            preserved_protected_region_count=0,
            protected_region_validation_status="not_applicable",
        ),
        lambda evidence: replace(evidence, protected_region_validation_status="failed"),
    ],
)
def test_protected_region_negative_evidence_fails_closed(mutation) -> None:
    run, corpus, config = _protected_fixture()
    evidence = mutation(run.cases[0].protected_region_evidence)
    run = replace(
        run,
        cases=(replace(run.cases[0], protected_region_evidence=evidence),),
    )

    evaluation = UniversalProofEvaluator().evaluate(
        run, corpus, config, cache_evidence=(_cache(),)
    )

    assert evaluation.success is False
    assert any(
        gate.gate_id
        in {
            "PROTECTED_REGION_COUNT",
            "PROTECTED_REGION_PRESERVATION",
            "PROTECTED_REGION_VALIDATION",
        }
        and gate.status is GateStatus.FAIL
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


@pytest.mark.parametrize(
    ("profile", "expected_reason"),
    [
        ("unknown", "UNKNOWN_EVALUATION_PROFILE"),
    ],
)
def test_unknown_profile_is_a_configuration_error(
    tmp_path: Path, profile: str, expected_reason: str
) -> None:
    config_path = tmp_path / "evaluation.toml"
    config_path.write_text(
        "\n".join(
            (
                'schema_version = "token-optimization-proof-evaluation.v1"',
                'evaluation_version = "token-10g.v1"',
                f'profile = "{profile}"',
                "required_gate_ids = []",
                "unavailable_allowed_gate_ids = []",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationConfigurationError, match=expected_reason):
        load_evaluation_config(config_path)


def test_structural_guard_keeps_evaluator_evaluation_only() -> None:
    source = Path("intergrax/runtime/token_optimization/proofs/evaluator.py").read_text(
        encoding="utf-8"
    )
    assert "TokenOptimizationLLMRouter" not in source
    assert "TokenOptimizationPipelineRunner" not in source
    assert "compute_openai_tools_schema_hash" not in source
    assert "import hashlib" not in source
