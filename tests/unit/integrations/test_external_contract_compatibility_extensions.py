# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import ExecutionId, RunId, TaskId
from intergrax.integrations.contracts.external_contract_compatibility import (
    DimensionCompatibilityStatus,
    ExternalContractAssessmentWindow,
    ExternalContractCompatibilityAssessmentRequest,
    ExternalContractCompatibilityDimension,
    ExternalContractCompatibilityEvidence,
    ExternalContractCompatibilityEvaluationContext,
    ExternalContractCompatibilityExpectation,
    ExternalContractCompatibilityFinding,
    ExternalContractCompatibilityOutcome,
    ExternalContractCompatibilityReasonCode,
    ExternalContractCompatibilitySubject,
    ExternalContractEvidenceAuthority,
    ExternalContractEvidenceCollectionRequest,
    ExternalContractEvidenceProvider,
    ExternalContractExpectationKey,
    ExternalContractPin,
    ExternalContractSemanticAssertionResult,
    ExternalContractSemanticEvidenceFact,
    SemanticAssertionStatus,
)
from intergrax.integrations.external_contract_compatibility_extensions import (
    ExternalContractCompatibilityExtensionsError,
    assess_external_contract_compatibility,
    build_external_contract_compatibility_service,
    collect_external_contract_compatibility_evidence,
    external_contract_compatibility_extensions,
)
from intergrax.integrations.external_contract_compatibility_service import (
    ExternalContractCompatibilityService,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TS = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
_WINDOW = ExternalContractAssessmentWindow(max_age=timedelta(hours=24))
_REPO_ROOT = Path(__file__).resolve().parents[3]


class _AcceptAllPolicy:
    def accepts(
        self,
        evidence: ExternalContractCompatibilityEvidence,
        *,
        assessed_at: datetime,
        window: ExternalContractAssessmentWindow,
    ) -> bool:
        return True


def _subject() -> ExternalContractCompatibilitySubject:
    return ExternalContractCompatibilitySubject(
        tenant_id="tenant-a",
        provider_id="prov",
        integration_kind="api",
        integration_id="prov:api",
        external_operation_id="op.read",
    )


def _semantic_expectation() -> ExternalContractCompatibilityExpectation:
    return ExternalContractCompatibilityExpectation(
        expectation_id="exp-semantic",
        subject=_subject(),
        expected_contract=ExternalContractPin("contract/ref", "v1"),
        required_dimensions=frozenset(
            {ExternalContractCompatibilityDimension.SEMANTIC}
        ),
        semantic_expectation_refs=("semantic/exp",),
    )


class _CustomEvidenceProviderA:
    def __init__(self) -> None:
        self.collect_calls = 0

    @property
    def evidence_provider_id(self) -> str:
        return "custom.evidence.a"

    def collect(
        self,
        request: ExternalContractEvidenceCollectionRequest,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]:
        self.collect_calls += 1
        return (
            ExternalContractCompatibilityEvidence(
                evidence_id="ev-custom-a",
                subject=request.subject,
                observed_contract=ExternalContractPin("contract/ref", "v1"),
                dimension=ExternalContractCompatibilityDimension.SEMANTIC,
                observed_at=request.assessed_at,
                authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
                evidence_refs=("ref/custom-a",),
                fact=ExternalContractSemanticEvidenceFact(
                    assertions=(
                        ExternalContractSemanticAssertionResult(
                            assertion_id="domain.rule.sample",
                            status=SemanticAssertionStatus.PASS,
                            authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
                            evidence_refs=("ref/custom-a",),
                        ),
                    )
                ),
            ),
        )


class _CustomEvidenceProviderB:
    @property
    def evidence_provider_id(self) -> str:
        return "custom.evidence.b"

    def collect(
        self,
        request: ExternalContractEvidenceCollectionRequest,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]:
        return (
            ExternalContractCompatibilityEvidence(
                evidence_id="ev-custom-b",
                subject=request.subject,
                observed_contract=None,
                dimension=ExternalContractCompatibilityDimension.SEMANTIC,
                observed_at=request.assessed_at,
                authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
                evidence_refs=("ref/custom-b",),
                fact=ExternalContractSemanticEvidenceFact(
                    assertions=(
                        ExternalContractSemanticAssertionResult(
                            assertion_id="domain.rule.sample",
                            status=SemanticAssertionStatus.PASS,
                            authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
                            evidence_refs=("ref/custom-b",),
                        ),
                    )
                ),
            ),
        )


class _DomainSemanticEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "custom.domain.semantic"

    @property
    def supported_dimensions(self) -> frozenset[ExternalContractCompatibilityDimension]:
        return frozenset({ExternalContractCompatibilityDimension.SEMANTIC})

    def can_evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        *,
        dimension: ExternalContractCompatibilityDimension,
    ) -> bool:
        return dimension is ExternalContractCompatibilityDimension.SEMANTIC

    def evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        context: ExternalContractCompatibilityEvaluationContext,
    ) -> tuple[ExternalContractCompatibilityFinding, ...]:
        refs = tuple(ref for item in evidence for ref in item.evidence_refs)
        return (
            ExternalContractCompatibilityFinding(
                dimension=ExternalContractCompatibilityDimension.SEMANTIC,
                status=DimensionCompatibilityStatus.COMPATIBLE,
                reason_code=ExternalContractCompatibilityReasonCode.NONE,
                evidence_refs=refs,
                evaluator_id=self.evaluator_id,
                source_authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
            ),
        )


def test_p2r1_01_custom_evidence_provider_satisfies_protocol() -> None:
    provider = _CustomEvidenceProviderA()
    assert isinstance(provider, ExternalContractEvidenceProvider)


def test_p2r1_02_custom_evaluator_without_core_patch() -> None:
    evaluator = _DomainSemanticEvaluator()
    extensions = external_contract_compatibility_extensions(evaluators=(evaluator,))
    service = build_external_contract_compatibility_service(
        extensions, evidence_policy=_AcceptAllPolicy()
    )
    assert isinstance(service, ExternalContractCompatibilityService)


def test_p2r1_03_collect_precedes_assess_no_service_io() -> None:
    provider = _CustomEvidenceProviderA()
    evaluator = _DomainSemanticEvaluator()
    extensions = external_contract_compatibility_extensions(
        evidence_providers=(provider,),
        evaluators=(evaluator,),
    )
    expectation = _semantic_expectation()
    collection_request = ExternalContractEvidenceCollectionRequest(
        expectation=expectation,
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    assessment_request = ExternalContractCompatibilityAssessmentRequest(
        assessment_id="assess-collect",
        expectation=expectation,
        evidence=(),
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    result = assess_external_contract_compatibility(
        extensions,
        evidence_policy=_AcceptAllPolicy(),
        collection_request=collection_request,
        assessment_request=assessment_request,
    )
    assert provider.collect_calls == 1
    assert result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE
    assert "ref/custom-a" in result.evidence_refs


def test_p2r1_04_domain_semantic_evaluator_assertion() -> None:
    provider = _CustomEvidenceProviderA()
    evaluator = _DomainSemanticEvaluator()
    extensions = external_contract_compatibility_extensions(
        evidence_providers=(provider,),
        evaluators=(evaluator,),
    )
    expectation = _semantic_expectation()
    collected = collect_external_contract_compatibility_evidence(
        extensions,
        ExternalContractEvidenceCollectionRequest(
            expectation=expectation, assessed_at=_TS
        ),
    )
    semantic_fact = collected[0].fact
    assert isinstance(semantic_fact, ExternalContractSemanticEvidenceFact)
    assert semantic_fact.assertions[0].assertion_id == "domain.rule.sample"


def test_p2r1_05_duplicate_evidence_provider_id_fails_closed() -> None:
    first = _CustomEvidenceProviderA()
    second = _CustomEvidenceProviderA()
    with pytest.raises(ExternalContractCompatibilityExtensionsError, match="duplicate"):
        external_contract_compatibility_extensions(
            evidence_providers=(first, second),
        )


def test_p2r1_06_duplicate_evaluator_id_fails_closed() -> None:
    first = _DomainSemanticEvaluator()
    second = _DomainSemanticEvaluator()
    with pytest.raises(ExternalContractCompatibilityExtensionsError, match="duplicate"):
        external_contract_compatibility_extensions(evaluators=(first, second))


def test_p2r1_07_extension_bundle_immutable_deterministic_order() -> None:
    provider_a = _CustomEvidenceProviderA()
    provider_b = _CustomEvidenceProviderB()
    extensions = external_contract_compatibility_extensions(
        evidence_providers=(provider_a, provider_b),
    )
    expectation = _semantic_expectation()
    request = ExternalContractEvidenceCollectionRequest(
        expectation=expectation, assessed_at=_TS
    )
    first = collect_external_contract_compatibility_evidence(extensions, request)
    second = collect_external_contract_compatibility_evidence(extensions, request)
    assert first == second
    assert first[0].evidence_id == "ev-custom-a"
    assert first[1].evidence_id == "ev-custom-b"
    replacement = replace(extensions, evidence_providers=())
    assert replacement is not extensions
    assert replacement.evidence_providers == ()
    assert isinstance(extensions.evidence_providers, tuple)


def test_p2r1_08_external_provider_replacement() -> None:
    expectation = _semantic_expectation()
    request = ExternalContractEvidenceCollectionRequest(
        expectation=expectation, assessed_at=_TS
    )
    from_a = collect_external_contract_compatibility_evidence(
        external_contract_compatibility_extensions(
            evidence_providers=(_CustomEvidenceProviderA(),)
        ),
        request,
    )
    from_b = collect_external_contract_compatibility_evidence(
        external_contract_compatibility_extensions(
            evidence_providers=(_CustomEvidenceProviderB(),)
        ),
        request,
    )
    assert from_a[0].evidence_id == "ev-custom-a"
    assert from_b[0].evidence_id == "ev-custom-b"


def test_p2r1_09_no_authority_or_execution_side_effects() -> None:
    forbidden_fragments = (
        "CapabilityGap",
        "ExecutionId",
        "ToolRuntime",
        "grant_permission",
        "ACQUIRE",
        "CONFIGURE_EXISTING",
        "ADAPT",
    )
    extensions_path = (
        _REPO_ROOT
        / "intergrax/integrations/external_contract_compatibility_extensions.py"
    )
    source = extensions_path.read_text(encoding="utf-8")
    for fragment in forbidden_fragments:
        assert fragment not in source


def test_architecture_regression_compatibility_surface() -> None:
    service_path = (
        _REPO_ROOT / "intergrax/integrations/external_contract_compatibility_service.py"
    )
    extensions_path = (
        _REPO_ROOT
        / "intergrax/integrations/external_contract_compatibility_extensions.py"
    )
    contracts_path = (
        _REPO_ROOT
        / "intergrax/integrations/contracts/external_contract_compatibility.py"
    )
    service_tree = ast.parse(service_path.read_text(encoding="utf-8"))
    service_from_modules = {
        node.module
        for node in ast.walk(service_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("integrations.providers" in mod for mod in service_from_modules)
    assert "IntegrationCatalog" not in service_path.read_text(encoding="utf-8")
    extensions_source = extensions_path.read_text(encoding="utf-8")
    assert "IntegrationCatalog" not in extensions_source
    contracts_source = contracts_path.read_text(encoding="utf-8")
    assert "dict[str, Any]" not in contracts_source
    assert "Mapping[str, Any]" not in contracts_source
    assert " if provider_id" not in service_path.read_text(encoding="utf-8")


def _assess_with_collection(
    *,
    collection_expectation: ExternalContractCompatibilityExpectation,
    assessment_expectation: ExternalContractCompatibilityExpectation,
    provider: _CustomEvidenceProviderA,
) -> None:
    extensions = external_contract_compatibility_extensions(
        evidence_providers=(provider,),
        evaluators=(_DomainSemanticEvaluator(),),
    )
    collection_request = ExternalContractEvidenceCollectionRequest(
        expectation=collection_expectation,
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    assessment_request = ExternalContractCompatibilityAssessmentRequest(
        assessment_id="assess-mismatch",
        expectation=assessment_expectation,
        evidence=(),
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    assess_external_contract_compatibility(
        extensions,
        evidence_policy=_AcceptAllPolicy(),
        collection_request=collection_request,
        assessment_request=assessment_request,
    )


def test_p2r1r1_01_expectation_contract_version_mismatch_fails_before_collect() -> None:
    provider = _CustomEvidenceProviderA()
    base = _semantic_expectation()
    collection_expectation = base
    assessment_expectation = replace(
        base,
        expected_contract=ExternalContractPin("contract/ref", "v2"),
    )
    with pytest.raises(
        ExternalContractCompatibilityExtensionsError,
        match="collection expectation does not match assessment expectation",
    ):
        _assess_with_collection(
            collection_expectation=collection_expectation,
            assessment_expectation=assessment_expectation,
            provider=provider,
        )
    assert provider.collect_calls == 0


def test_p2r1r1_02_semantic_expectation_refs_mismatch_fails_before_collect() -> None:
    provider = _CustomEvidenceProviderA()
    base = _semantic_expectation()
    collection_expectation = base
    assessment_expectation = replace(
        base,
        semantic_expectation_refs=("semantic/other",),
    )
    with pytest.raises(
        ExternalContractCompatibilityExtensionsError,
        match="collection expectation does not match assessment expectation",
    ):
        _assess_with_collection(
            collection_expectation=collection_expectation,
            assessment_expectation=assessment_expectation,
            provider=provider,
        )
    assert provider.collect_calls == 0


def test_p2r1r1_03_matching_expectation_succeeds() -> None:
    provider = _CustomEvidenceProviderA()
    expectation = _semantic_expectation()
    extensions = external_contract_compatibility_extensions(
        evidence_providers=(provider,),
        evaluators=(_DomainSemanticEvaluator(),),
    )
    collection_request = ExternalContractEvidenceCollectionRequest(
        expectation=expectation,
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    assessment_request = ExternalContractCompatibilityAssessmentRequest(
        assessment_id="assess-match",
        expectation=expectation,
        evidence=(),
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    result = assess_external_contract_compatibility(
        extensions,
        evidence_policy=_AcceptAllPolicy(),
        collection_request=collection_request,
        assessment_request=assessment_request,
    )
    assert provider.collect_calls == 1
    assert result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE


def _expectation_key_for_subject(
    subject: ExternalContractCompatibilitySubject,
    *,
    contract_version: str = "v1",
) -> ExternalContractExpectationKey:
    return ExternalContractExpectationKey(
        subject=subject,
        expected_contract=ExternalContractPin("contract/ref", contract_version),
    )


def test_p2r1r1_04_key_preserves_integration_id() -> None:
    subject_a = ExternalContractCompatibilitySubject(
        tenant_id="tenant-a",
        external_operation_id="op.read",
        provider_id="prov-a",
        integration_kind="api",
        integration_id="prov-a:api",
    )
    subject_b = ExternalContractCompatibilitySubject(
        tenant_id="tenant-a",
        external_operation_id="op.read",
        provider_id="prov-b",
        integration_kind="api",
        integration_id="prov-b:api",
    )
    assert _expectation_key_for_subject(subject_a) != _expectation_key_for_subject(
        subject_b
    )


def _subject_with_execution_scope(
    *,
    execution_task_id: TaskId | None = None,
    execution_run_id: RunId | None = None,
    execution_id: ExecutionId | None = None,
) -> ExternalContractCompatibilitySubject:
    return ExternalContractCompatibilitySubject(
        tenant_id="tenant-a",
        provider_id="prov",
        integration_kind="api",
        integration_id="prov:api",
        external_operation_id="op.read",
        execution_task_id=execution_task_id,
        execution_run_id=execution_run_id,
        execution_id=execution_id,
    )


def test_p2r1r1_05_key_preserves_execution_task_id() -> None:
    task_a = TaskId("task_" + "a" * 32)
    task_b = TaskId("task_" + "b" * 32)
    subject_a = _subject_with_execution_scope(execution_task_id=task_a)
    subject_b = _subject_with_execution_scope(execution_task_id=task_b)
    assert _expectation_key_for_subject(subject_a) != _expectation_key_for_subject(
        subject_b
    )


def test_p2r1r1_06_key_preserves_execution_run_and_execution_id() -> None:
    run_a = RunId("run_" + "a" * 32)
    run_b = RunId("run_" + "b" * 32)
    exec_a = ExecutionId("exec_" + "c" * 32)
    exec_b = ExecutionId("exec_" + "d" * 32)
    with_run_only_a = _subject_with_execution_scope(execution_run_id=run_a)
    with_run_only_b = _subject_with_execution_scope(execution_run_id=run_b)
    with_exec_only_a = _subject_with_execution_scope(execution_id=exec_a)
    with_exec_only_b = _subject_with_execution_scope(execution_id=exec_b)
    assert _expectation_key_for_subject(
        with_run_only_a
    ) != _expectation_key_for_subject(with_run_only_b)
    assert _expectation_key_for_subject(
        with_exec_only_a
    ) != _expectation_key_for_subject(with_exec_only_b)


def test_p2r1r1_07_expected_contract_separate_from_subject_identity() -> None:
    subject = _subject()
    key_v1 = _expectation_key_for_subject(subject, contract_version="v1")
    key_v2 = _expectation_key_for_subject(subject, contract_version="v2")
    assert key_v1 != key_v2
    assert key_v1.subject == key_v2.subject
