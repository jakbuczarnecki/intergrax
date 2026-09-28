# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intergrax.integrations.contracts.external_contract_compatibility import (
    DimensionCompatibilityStatus,
    ExternalContractAssessmentWindow,
    ExternalContractCompatibilityAssessment,
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
    ExternalContractPin,
    ExternalContractSchemaEvidenceFact,
    ExternalContractSemanticAssertionResult,
    ExternalContractSemanticEvidenceFact,
    SchemaValidationStatus,
    SemanticAssertionStatus,
)
from intergrax.integrations.examples.custom_memory_kv.compatibility import (
    CustomMemoryKvContractEvidenceProvider,
    CustomMemoryKvContractObservation,
    CustomMemoryKvSchemaCompatibilityEvaluator,
    custom_memory_kv_compatibility_extensions,
)
from intergrax.integrations.examples.custom_memory_kv.integration import (
    CUSTOM_MEMORY_KV_PROVIDER_ID,
)
from intergrax.integrations.external_contract_compatibility_extensions import (
    ExternalContractCompatibilityExtensions,
    assess_external_contract_compatibility,
    collect_external_contract_compatibility_evidence,
    external_contract_compatibility_extensions,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TS = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
_TS_OBSERVED_EARLIER = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
_TS_ASSESSED_LATER = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
_WINDOW = ExternalContractAssessmentWindow(max_age=timedelta(hours=24))
_STALE_WINDOW = ExternalContractAssessmentWindow(max_age=timedelta(hours=1))
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_REF = "integrations/custom_memory_kv/read"
_CONTRACT_VERSION = "v1"


class _AcceptAllPolicy:
    def accepts(
        self,
        evidence: ExternalContractCompatibilityEvidence,
        *,
        assessed_at: datetime,
        window: ExternalContractAssessmentWindow,
    ) -> bool:
        return True


class _MaxAgeEvidencePolicy:
    def accepts(
        self,
        evidence: ExternalContractCompatibilityEvidence,
        *,
        assessed_at: datetime,
        window: ExternalContractAssessmentWindow,
    ) -> bool:
        if window.max_age is None:
            return True
        return assessed_at - evidence.observed_at <= window.max_age


def _subject(*, tenant_id: str = "tenant-a") -> ExternalContractCompatibilitySubject:
    return ExternalContractCompatibilitySubject(
        tenant_id=tenant_id,
        provider_id=CUSTOM_MEMORY_KV_PROVIDER_ID,
        integration_kind="key_value_cache",
        integration_id=f"{CUSTOM_MEMORY_KV_PROVIDER_ID}:key_value_cache",
        external_operation_id="kv.get",
    )


def _reference_expectation(
    *,
    tenant_id: str = "tenant-a",
) -> ExternalContractCompatibilityExpectation:
    return ExternalContractCompatibilityExpectation(
        expectation_id="exp-kv-ref",
        subject=_subject(tenant_id=tenant_id),
        expected_contract=ExternalContractPin(_CONTRACT_REF, _CONTRACT_VERSION),
        required_dimensions=frozenset(
            {
                ExternalContractCompatibilityDimension.SCHEMA,
                ExternalContractCompatibilityDimension.SEMANTIC,
            }
        ),
        schema_expectation_ref="schema/custom_memory_kv/get",
        domain_extension_ref="application.cache-key-policy",
    )


def _passing_schema_observation(
    *,
    subject: ExternalContractCompatibilitySubject | None = None,
    observed_at: datetime = _TS,
) -> CustomMemoryKvContractObservation:
    return CustomMemoryKvContractObservation(
        subject=subject if subject is not None else _subject(),
        observed_at=observed_at,
        observed_contract=ExternalContractPin(_CONTRACT_REF, _CONTRACT_VERSION),
        schema_ref="schema/custom_memory_kv/get",
        schema_fingerprint="fp-observed-abc",
        validation_status=SchemaValidationStatus.PASS,
        evidence_refs=("probe/custom_memory_kv/schema/1",),
    )


def _collection_request(
    expectation: ExternalContractCompatibilityExpectation,
    *,
    assessed_at: datetime = _TS,
    assessment_window: ExternalContractAssessmentWindow = _WINDOW,
) -> ExternalContractEvidenceCollectionRequest:
    return ExternalContractEvidenceCollectionRequest(
        expectation=expectation,
        assessed_at=assessed_at,
        assessment_window=assessment_window,
    )


@dataclass(frozen=True, slots=True)
class _ApplicationSemanticProbe:
    assertion_status: SemanticAssertionStatus
    evidence_refs: tuple[str, ...]


class _ApplicationSemanticEvidenceProvider:
    def __init__(self, probe: _ApplicationSemanticProbe) -> None:
        self._probe = probe

    @property
    def evidence_provider_id(self) -> str:
        return "application.cache_key_policy.evidence"

    def collect(
        self,
        request: ExternalContractEvidenceCollectionRequest,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]:
        probe = self._probe
        return (
            ExternalContractCompatibilityEvidence(
                evidence_id="ev-app-semantic",
                subject=request.subject,
                observed_contract=None,
                dimension=ExternalContractCompatibilityDimension.SEMANTIC,
                observed_at=request.assessed_at,
                authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
                evidence_refs=probe.evidence_refs,
                fact=ExternalContractSemanticEvidenceFact(
                    assertions=(
                        ExternalContractSemanticAssertionResult(
                            assertion_id="application.cache-key-policy",
                            status=probe.assertion_status,
                            authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
                            evidence_refs=probe.evidence_refs,
                        ),
                    )
                ),
            ),
        )


class _ApplicationSemanticEvaluator:
    @property
    def evaluator_id(self) -> str:
        return "application.cache_key_policy.semantic"

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
        semantic_items = tuple(
            item
            for item in evidence
            if item.dimension is ExternalContractCompatibilityDimension.SEMANTIC
        )
        if not semantic_items:
            return (
                ExternalContractCompatibilityFinding(
                    dimension=ExternalContractCompatibilityDimension.SEMANTIC,
                    status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                    reason_code=ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE,
                    evidence_refs=(),
                    evaluator_id=self.evaluator_id,
                    source_authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
                ),
            )
        fact = semantic_items[0].fact
        if type(fact) is not ExternalContractSemanticEvidenceFact:
            return (
                ExternalContractCompatibilityFinding(
                    dimension=ExternalContractCompatibilityDimension.SEMANTIC,
                    status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                    reason_code=ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE,
                    evidence_refs=semantic_items[0].evidence_refs,
                    evaluator_id=self.evaluator_id,
                    source_authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
                ),
            )
        assertion = fact.assertions[0]
        refs = assertion.evidence_refs
        if assertion.status is SemanticAssertionStatus.PASS:
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
        return (
            ExternalContractCompatibilityFinding(
                dimension=ExternalContractCompatibilityDimension.SEMANTIC,
                status=DimensionCompatibilityStatus.INCOMPATIBLE,
                reason_code=ExternalContractCompatibilityReasonCode.SEMANTIC_MISMATCH,
                evidence_refs=refs,
                evaluator_id=self.evaluator_id,
                source_authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
            ),
        )


def _composed_extensions(
    *,
    schema_observation: CustomMemoryKvContractObservation,
    semantic_probe: _ApplicationSemanticProbe,
) -> ExternalContractCompatibilityExtensions:
    provider_bundle = custom_memory_kv_compatibility_extensions(schema_observation)
    return external_contract_compatibility_extensions(
        evidence_providers=provider_bundle.evidence_providers
        + (_ApplicationSemanticEvidenceProvider(semantic_probe),),
        evaluators=provider_bundle.evaluators + (_ApplicationSemanticEvaluator(),),
    )


def _assess_reference(
    extensions: ExternalContractCompatibilityExtensions,
    expectation: ExternalContractCompatibilityExpectation,
) -> ExternalContractCompatibilityAssessment:
    collection_request = ExternalContractEvidenceCollectionRequest(
        expectation=expectation,
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    assessment_request = ExternalContractCompatibilityAssessmentRequest(
        assessment_id="assess-p2r2",
        expectation=expectation,
        evidence=(),
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    return assess_external_contract_compatibility(
        extensions,
        evidence_policy=_AcceptAllPolicy(),
        collection_request=collection_request,
        assessment_request=assessment_request,
    )


def test_p2r2_01_provider_reference_implements_evidence_provider_protocol() -> None:
    provider = CustomMemoryKvContractEvidenceProvider(_passing_schema_observation())
    assert isinstance(provider, ExternalContractEvidenceProvider)


def test_p2r2_02_provider_evaluator_implements_evaluator_protocol() -> None:
    evaluator = CustomMemoryKvSchemaCompatibilityEvaluator()
    assert evaluator.evaluator_id == "custom_memory_kv.schema.v1"
    assert (
        ExternalContractCompatibilityDimension.SCHEMA in evaluator.supported_dimensions
    )


def test_p2r2_03_application_semantic_evaluator_implements_evaluator_protocol() -> None:
    evaluator = _ApplicationSemanticEvaluator()
    assert (
        ExternalContractCompatibilityDimension.SEMANTIC
        in evaluator.supported_dimensions
    )


def test_p2r2_04_provider_and_domain_extensions_compose_without_core_patch() -> None:
    extensions = _composed_extensions(
        schema_observation=_passing_schema_observation(),
        semantic_probe=_ApplicationSemanticProbe(
            assertion_status=SemanticAssertionStatus.PASS,
            evidence_refs=("probe/application/cache-key/1",),
        ),
    )
    assert len(extensions.evidence_providers) == 2
    assert len(extensions.evaluators) == 2


def test_p2r2_05_schema_pass_semantic_pass_compatible() -> None:
    expectation = _reference_expectation()
    extensions = _composed_extensions(
        schema_observation=_passing_schema_observation(),
        semantic_probe=_ApplicationSemanticProbe(
            assertion_status=SemanticAssertionStatus.PASS,
            evidence_refs=("probe/application/cache-key/1",),
        ),
    )
    result = _assess_reference(extensions, expectation)
    assert result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE
    assert result.subject.tenant_id == "tenant-a"
    assert result.expectation_id == expectation.expectation_id
    assert "probe/custom_memory_kv/schema/1" in result.evidence_refs
    assert "probe/application/cache-key/1" in result.evidence_refs


def test_p2r2_06_schema_pass_semantic_fail_semantic_incompatible() -> None:
    expectation = _reference_expectation()
    extensions = _composed_extensions(
        schema_observation=_passing_schema_observation(),
        semantic_probe=_ApplicationSemanticProbe(
            assertion_status=SemanticAssertionStatus.FAIL,
            evidence_refs=("probe/application/cache-key/fail",),
        ),
    )
    result = _assess_reference(extensions, expectation)
    assert result.outcome is ExternalContractCompatibilityOutcome.SEMANTIC_INCOMPATIBLE
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.SEMANTIC_MISMATCH
    )


class _AlternateKvSchemaEvidenceProvider:
    """Structural substitute for :class:`CustomMemoryKvContractEvidenceProvider`."""

    def __init__(self, observation: CustomMemoryKvContractObservation) -> None:
        self._observation = observation

    @property
    def evidence_provider_id(self) -> str:
        return "alternate.kv.contract.evidence"

    def collect(
        self,
        request: ExternalContractEvidenceCollectionRequest,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]:
        observation = self._observation
        if observation.subject != request.subject:
            return ()
        return (
            ExternalContractCompatibilityEvidence(
                evidence_id="ev-alt-schema",
                subject=observation.subject,
                observed_contract=observation.observed_contract,
                dimension=ExternalContractCompatibilityDimension.SCHEMA,
                observed_at=observation.observed_at,
                authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
                evidence_refs=observation.evidence_refs,
                fact=ExternalContractSchemaEvidenceFact(
                    schema_ref=observation.schema_ref,
                    schema_fingerprint=observation.schema_fingerprint,
                    validation_status=observation.validation_status,
                ),
            ),
        )


def test_p2r2_07_alternate_provider_structurally_replaceable() -> None:
    expectation = _reference_expectation()
    observation = _passing_schema_observation()
    ref_extensions = _composed_extensions(
        schema_observation=observation,
        semantic_probe=_ApplicationSemanticProbe(
            assertion_status=SemanticAssertionStatus.PASS,
            evidence_refs=("probe/application/cache-key/1",),
        ),
    )
    alt_extensions = external_contract_compatibility_extensions(
        evidence_providers=(
            _AlternateKvSchemaEvidenceProvider(observation),
            _ApplicationSemanticEvidenceProvider(
                _ApplicationSemanticProbe(
                    assertion_status=SemanticAssertionStatus.PASS,
                    evidence_refs=("probe/application/cache-key/1",),
                )
            ),
        ),
        evaluators=(
            CustomMemoryKvSchemaCompatibilityEvaluator(),
            _ApplicationSemanticEvaluator(),
        ),
    )
    ref_result = _assess_reference(ref_extensions, expectation)
    alt_result = _assess_reference(alt_extensions, expectation)
    assert ref_result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE
    assert alt_result.outcome is ExternalContractCompatibilityOutcome.COMPATIBLE


def test_p2r2_08_no_provider_domain_extension_authority_side_effects() -> None:
    compatibility_path = (
        _REPO_ROOT / "intergrax/integrations/examples/custom_memory_kv/compatibility.py"
    )
    source = compatibility_path.read_text(encoding="utf-8")
    for fragment in (
        "CapabilityGap",
        "ExecutionId",
        "ToolRuntime",
        "grant_permission",
        "IntegrationCatalog",
    ):
        assert fragment not in source
    for token in ("ACQUIRE", "CONFIGURE_EXISTING", "ADAPT"):
        assert re.search(rf"\b{token}\b", source) is None


def test_ten_r2_01_tenant_continuity() -> None:
    expectation = _reference_expectation(tenant_id="tenant-alpha")
    extensions = _composed_extensions(
        schema_observation=_passing_schema_observation(
            subject=_subject(tenant_id="tenant-alpha"),
        ),
        semantic_probe=_ApplicationSemanticProbe(
            assertion_status=SemanticAssertionStatus.PASS,
            evidence_refs=("probe/application/cache-key/1",),
        ),
    )
    collected = collect_external_contract_compatibility_evidence(
        extensions,
        ExternalContractEvidenceCollectionRequest(
            expectation=expectation, assessed_at=_TS, assessment_window=_WINDOW
        ),
    )
    for item in collected:
        assert item.subject.tenant_id == "tenant-alpha"
    result = _assess_reference(extensions, expectation)
    assert result.subject.tenant_id == "tenant-alpha"


class _WrongTenantSchemaEvidenceProvider:
    @property
    def evidence_provider_id(self) -> str:
        return "wrong.tenant.schema"

    def collect(
        self,
        request: ExternalContractEvidenceCollectionRequest,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]:
        wrong_subject = _subject(tenant_id="tenant-b")
        observation = _passing_schema_observation(
            subject=_subject(tenant_id="tenant-a")
        )
        return (
            ExternalContractCompatibilityEvidence(
                evidence_id="ev-wrong-tenant",
                subject=wrong_subject,
                observed_contract=observation.observed_contract,
                dimension=ExternalContractCompatibilityDimension.SCHEMA,
                observed_at=request.assessed_at,
                authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
                evidence_refs=observation.evidence_refs,
                fact=ExternalContractSchemaEvidenceFact(
                    schema_ref=observation.schema_ref,
                    schema_fingerprint=observation.schema_fingerprint,
                    validation_status=observation.validation_status,
                ),
            ),
        )


def test_ten_r2_02_wrong_tenant_provider_evidence_fails_closed() -> None:
    expectation = _reference_expectation(tenant_id="tenant-a")
    extensions = external_contract_compatibility_extensions(
        evidence_providers=(
            _WrongTenantSchemaEvidenceProvider(),
            _ApplicationSemanticEvidenceProvider(
                _ApplicationSemanticProbe(
                    assertion_status=SemanticAssertionStatus.PASS,
                    evidence_refs=("probe/application/cache-key/1",),
                )
            ),
        ),
        evaluators=(
            CustomMemoryKvSchemaCompatibilityEvaluator(),
            _ApplicationSemanticEvaluator(),
        ),
    )
    result = _assess_reference(extensions, expectation)
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH
    )


def test_ten_r2_03_wrong_tenant_domain_evidence_fails_closed() -> None:
    expectation = _reference_expectation(tenant_id="tenant-a")
    tenant_b_subject = _subject(tenant_id="tenant-b")
    wrong_semantic = ExternalContractCompatibilityEvidence(
        evidence_id="ev-semantic-wrong-tenant",
        subject=tenant_b_subject,
        observed_contract=None,
        dimension=ExternalContractCompatibilityDimension.SEMANTIC,
        observed_at=_TS,
        authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
        evidence_refs=("probe/application/cache-key/wrong-tenant",),
        fact=ExternalContractSemanticEvidenceFact(
            assertions=(
                ExternalContractSemanticAssertionResult(
                    assertion_id="application.cache-key-policy",
                    status=SemanticAssertionStatus.PASS,
                    authority=ExternalContractEvidenceAuthority.APPLICATION_INVARIANT,
                    evidence_refs=("probe/application/cache-key/wrong-tenant",),
                ),
            )
        ),
    )
    extensions = _composed_extensions(
        schema_observation=_passing_schema_observation(),
        semantic_probe=_ApplicationSemanticProbe(
            assertion_status=SemanticAssertionStatus.PASS,
            evidence_refs=("probe/application/cache-key/1",),
        ),
    )
    collection_request = ExternalContractEvidenceCollectionRequest(
        expectation=expectation,
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    assessment_request = ExternalContractCompatibilityAssessmentRequest(
        assessment_id="assess-ten-r2-03",
        expectation=expectation,
        evidence=(wrong_semantic,),
        assessed_at=_TS,
        assessment_window=_WINDOW,
    )
    result = assess_external_contract_compatibility(
        extensions,
        evidence_policy=_AcceptAllPolicy(),
        collection_request=collection_request,
        assessment_request=assessment_request,
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert (
        result.reason_code is ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH
    )


def test_ten_r2_04_adapter_never_invents_default_or_global_tenant() -> None:
    provider_factory = CustomMemoryKvContractEvidenceProvider
    for tenant_id in ("tenant-x", "tenant-y"):
        subject = _subject(tenant_id=tenant_id)
        observation = _passing_schema_observation(subject=subject)
        provider = provider_factory(observation)
        expectation = _reference_expectation(tenant_id=tenant_id)
        collected = provider.collect(_collection_request(expectation))
        assert len(collected) == 1
        assert collected[0].subject == subject
        assert collected[0].subject.tenant_id == tenant_id
        assert collected[0].subject.tenant_id not in ("default", "global", "system")


def test_r2r1_01_matching_subject_emits_evidence_with_observation_provenance() -> None:
    subject = _subject(tenant_id="tenant-a")
    observation = _passing_schema_observation(subject=subject)
    provider = CustomMemoryKvContractEvidenceProvider(observation)
    expectation = _reference_expectation(tenant_id="tenant-a")
    collected = provider.collect(_collection_request(expectation))
    assert len(collected) == 1
    assert collected[0].subject == observation.subject


def test_r2r1_02_tenant_mismatch_emits_zero_evidence() -> None:
    observation = _passing_schema_observation(subject=_subject(tenant_id="tenant-a"))
    provider = CustomMemoryKvContractEvidenceProvider(observation)
    expectation = _reference_expectation(tenant_id="tenant-b")
    assert provider.collect(_collection_request(expectation)) == ()
    extensions = external_contract_compatibility_extensions(
        evidence_providers=(provider,),
        evaluators=(CustomMemoryKvSchemaCompatibilityEvaluator(),),
    )
    expectation_schema_only = ExternalContractCompatibilityExpectation(
        expectation_id="exp-tenant-mismatch",
        subject=_subject(tenant_id="tenant-b"),
        expected_contract=ExternalContractPin(_CONTRACT_REF, _CONTRACT_VERSION),
        required_dimensions=frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
        schema_expectation_ref="schema/custom_memory_kv/get",
        domain_extension_ref=None,
    )
    result = assess_external_contract_compatibility(
        extensions,
        evidence_policy=_AcceptAllPolicy(),
        collection_request=_collection_request(expectation_schema_only),
        assessment_request=ExternalContractCompatibilityAssessmentRequest(
            assessment_id="assess-r2r1-02",
            expectation=expectation_schema_only,
            evidence=(),
            assessed_at=_TS,
            assessment_window=_WINDOW,
        ),
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE


def test_r2r1_03_operation_mismatch_emits_zero_evidence() -> None:
    base = _subject(tenant_id="tenant-a")
    observation_subject = ExternalContractCompatibilitySubject(
        tenant_id=base.tenant_id,
        integration_id=base.integration_id,
        provider_id=base.provider_id,
        integration_kind=base.integration_kind,
        external_operation_id="kv.put",
    )
    observation = _passing_schema_observation(subject=observation_subject)
    provider = CustomMemoryKvContractEvidenceProvider(observation)
    expectation = _reference_expectation(tenant_id="tenant-a")
    assert provider.collect(_collection_request(expectation)) == ()


@pytest.mark.parametrize(
    "mutate_subject",
    (
        lambda s: ExternalContractCompatibilitySubject(
            tenant_id=s.tenant_id,
            integration_id=f"{s.provider_id}:other_kind",
            provider_id=s.provider_id,
            integration_kind="other_kind",
            external_operation_id=s.external_operation_id,
        ),
        lambda s: ExternalContractCompatibilitySubject(
            tenant_id=s.tenant_id,
            integration_id="other.provider:key_value_cache",
            provider_id="other.provider",
            integration_kind=s.integration_kind,
            external_operation_id=s.external_operation_id,
        ),
    ),
    ids=("integration_kind", "provider_id"),
)
def test_r2r1_04_provider_or_integration_mismatch_emits_zero_evidence(
    mutate_subject: Callable[
        [ExternalContractCompatibilitySubject], ExternalContractCompatibilitySubject
    ],
) -> None:
    base = _subject(tenant_id="tenant-a")
    observation = _passing_schema_observation(subject=mutate_subject(base))
    provider = CustomMemoryKvContractEvidenceProvider(observation)
    expectation = _reference_expectation(tenant_id="tenant-a")
    assert provider.collect(_collection_request(expectation)) == ()


def test_r2r1_05_observation_timestamp_preserved_not_assessment_time() -> None:
    subject = _subject(tenant_id="tenant-a")
    observation = _passing_schema_observation(
        subject=subject,
        observed_at=_TS_OBSERVED_EARLIER,
    )
    provider = CustomMemoryKvContractEvidenceProvider(observation)
    expectation = _reference_expectation(tenant_id="tenant-a")
    collected = provider.collect(
        _collection_request(expectation, assessed_at=_TS_ASSESSED_LATER)
    )
    assert len(collected) == 1
    assert collected[0].observed_at == _TS_OBSERVED_EARLIER
    assert collected[0].observed_at != _TS_ASSESSED_LATER


def test_r2r1_06_stale_observation_remains_stale_under_evidence_policy() -> None:
    subject = _subject(tenant_id="tenant-a")
    observation = _passing_schema_observation(
        subject=subject,
        observed_at=_TS_OBSERVED_EARLIER,
    )
    extensions = custom_memory_kv_compatibility_extensions(observation)
    expectation = ExternalContractCompatibilityExpectation(
        expectation_id="exp-stale-schema",
        subject=subject,
        expected_contract=ExternalContractPin(_CONTRACT_REF, _CONTRACT_VERSION),
        required_dimensions=frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
        schema_expectation_ref="schema/custom_memory_kv/get",
        domain_extension_ref=None,
    )
    collection_request = _collection_request(
        expectation,
        assessed_at=_TS_ASSESSED_LATER,
        assessment_window=_STALE_WINDOW,
    )
    assessment_request = ExternalContractCompatibilityAssessmentRequest(
        assessment_id="assess-r2r1-06",
        expectation=expectation,
        evidence=(),
        assessed_at=_TS_ASSESSED_LATER,
        assessment_window=_STALE_WINDOW,
    )
    result = assess_external_contract_compatibility(
        extensions,
        evidence_policy=_MaxAgeEvidencePolicy(),
        collection_request=collection_request,
        assessment_request=assessment_request,
    )
    assert result.outcome is ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
    assert result.reason_code is ExternalContractCompatibilityReasonCode.STALE_EVIDENCE


def test_architecture_regression_reference_extension_surface() -> None:
    service_path = (
        _REPO_ROOT / "intergrax/integrations/external_contract_compatibility_service.py"
    )
    extensions_path = (
        _REPO_ROOT
        / "intergrax/integrations/external_contract_compatibility_extensions.py"
    )
    reference_path = (
        _REPO_ROOT / "intergrax/integrations/examples/custom_memory_kv/compatibility.py"
    )
    service_source = service_path.read_text(encoding="utf-8")
    extensions_source = extensions_path.read_text(encoding="utf-8")
    assert "custom_memory_kv.compatibility" not in service_source
    assert "custom_memory_kv.compatibility" not in extensions_source
    reference_source = reference_path.read_text(encoding="utf-8")
    reference_tree = ast.parse(reference_source)
    import_modules = {
        node.module
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden_prefixes = (
        "intergrax.governance",
        "intergrax.runtime.execution",
        "intergrax.runtime.tool_runtime",
    )
    for prefix in forbidden_prefixes:
        assert not any(mod.startswith(prefix) for mod in import_modules)
    assert "ToolRuntime" not in reference_source
    assert "IntegrationCatalog" not in reference_source
    assert "dict[str, Any]" not in reference_source
    assert "Mapping[str, Any]" not in reference_source
    assert ": Any" not in reference_source
