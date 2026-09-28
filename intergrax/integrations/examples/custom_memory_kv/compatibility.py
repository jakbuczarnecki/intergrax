# © Artur Czarnecki. All rights reserved.

"""Reference provider-owned external contract compatibility extensions (example package)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.external_contract_compatibility import (
    DimensionCompatibilityStatus,
    ExternalContractCompatibilityDimension,
    ExternalContractCompatibilityEvaluationContext,
    ExternalContractCompatibilityEvidence,
    ExternalContractCompatibilityExpectation,
    ExternalContractCompatibilityFinding,
    ExternalContractCompatibilityReasonCode,
    ExternalContractEvidenceAuthority,
    ExternalContractEvidenceCollectionRequest,
    ExternalContractPin,
    ExternalContractSchemaEvidenceFact,
    SchemaValidationStatus,
)
from intergrax.integrations.examples.custom_memory_kv.integration import (
    CUSTOM_MEMORY_KV_PROVIDER_ID,
)
from intergrax.integrations.external_contract_compatibility_extensions import (
    ExternalContractCompatibilityExtensions,
    external_contract_compatibility_extensions,
)

_PROVIDER_EVIDENCE_PROVIDER_ID = "custom_memory_kv.contract.evidence"
_SCHEMA_EVALUATOR_ID = "custom_memory_kv.schema.v1"


@dataclass(frozen=True, slots=True)
class CustomMemoryKvContractObservation:
    """Typed provider-side probe result; not copied from platform expectation."""

    observed_contract: ExternalContractPin | None
    schema_ref: str | None
    schema_fingerprint: str | None
    validation_status: SchemaValidationStatus
    evidence_refs: tuple[str, ...]


class CustomMemoryKvContractEvidenceProvider:
    """Reference :class:`ExternalContractEvidenceProvider` for the example KV integration."""

    def __init__(self, observation: CustomMemoryKvContractObservation) -> None:
        self._observation = observation

    @property
    def evidence_provider_id(self) -> str:
        return _PROVIDER_EVIDENCE_PROVIDER_ID

    def collect(
        self,
        request: ExternalContractEvidenceCollectionRequest,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]:
        observation = self._observation
        if not observation.evidence_refs:
            return ()
        return (
            ExternalContractCompatibilityEvidence(
                evidence_id=f"{_PROVIDER_EVIDENCE_PROVIDER_ID}:{observation.evidence_refs[0]}",
                subject=request.subject,
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


class CustomMemoryKvSchemaCompatibilityEvaluator:
    """Provider-owned SCHEMA dimension evaluator for observed KV contract facts."""

    @property
    def evaluator_id(self) -> str:
        return _SCHEMA_EVALUATOR_ID

    @property
    def supported_dimensions(self) -> frozenset[ExternalContractCompatibilityDimension]:
        return frozenset({ExternalContractCompatibilityDimension.SCHEMA})

    def can_evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        *,
        dimension: ExternalContractCompatibilityDimension,
    ) -> bool:
        if dimension is not ExternalContractCompatibilityDimension.SCHEMA:
            return False
        if expectation.subject.provider_id != CUSTOM_MEMORY_KV_PROVIDER_ID:
            return False
        return any(
            item.dimension is ExternalContractCompatibilityDimension.SCHEMA
            and item.authority is ExternalContractEvidenceAuthority.PROVIDER_ADAPTER
            for item in evidence
        )

    def evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        context: ExternalContractCompatibilityEvaluationContext,
    ) -> tuple[ExternalContractCompatibilityFinding, ...]:
        schema_items = tuple(
            item
            for item in evidence
            if item.dimension is ExternalContractCompatibilityDimension.SCHEMA
        )
        if not schema_items:
            return (
                ExternalContractCompatibilityFinding(
                    dimension=ExternalContractCompatibilityDimension.SCHEMA,
                    status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                    reason_code=ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE,
                    evidence_refs=(),
                    evaluator_id=self.evaluator_id,
                    source_authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
                ),
            )
        item = schema_items[0]
        fact = item.fact
        if type(fact) is not ExternalContractSchemaEvidenceFact:
            return (
                ExternalContractCompatibilityFinding(
                    dimension=ExternalContractCompatibilityDimension.SCHEMA,
                    status=DimensionCompatibilityStatus.INSUFFICIENT_EVIDENCE,
                    reason_code=ExternalContractCompatibilityReasonCode.MISSING_REQUIRED_EVIDENCE,
                    evidence_refs=item.evidence_refs,
                    evaluator_id=self.evaluator_id,
                    source_authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
                ),
            )
        refs = item.evidence_refs
        if fact.validation_status is not SchemaValidationStatus.PASS:
            return (
                ExternalContractCompatibilityFinding(
                    dimension=ExternalContractCompatibilityDimension.SCHEMA,
                    status=DimensionCompatibilityStatus.INCOMPATIBLE,
                    reason_code=ExternalContractCompatibilityReasonCode.SCHEMA_MISMATCH,
                    evidence_refs=refs,
                    evaluator_id=self.evaluator_id,
                    source_authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
                ),
            )
        expected = expectation.expected_contract
        observed = item.observed_contract
        if observed is None or (
            observed.contract_ref != expected.contract_ref
            or observed.contract_version != expected.contract_version
        ):
            return (
                ExternalContractCompatibilityFinding(
                    dimension=ExternalContractCompatibilityDimension.SCHEMA,
                    status=DimensionCompatibilityStatus.INCOMPATIBLE,
                    reason_code=ExternalContractCompatibilityReasonCode.SCHEMA_MISMATCH,
                    evidence_refs=refs,
                    evaluator_id=self.evaluator_id,
                    source_authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
                ),
            )
        return (
            ExternalContractCompatibilityFinding(
                dimension=ExternalContractCompatibilityDimension.SCHEMA,
                status=DimensionCompatibilityStatus.COMPATIBLE,
                reason_code=ExternalContractCompatibilityReasonCode.NONE,
                evidence_refs=refs,
                evaluator_id=self.evaluator_id,
                source_authority=ExternalContractEvidenceAuthority.PROVIDER_ADAPTER,
            ),
        )


def custom_memory_kv_compatibility_extensions(
    observation: CustomMemoryKvContractObservation,
) -> ExternalContractCompatibilityExtensions:
    """Build provider reference extension bundle for explicit composition roots."""
    return external_contract_compatibility_extensions(
        evidence_providers=(CustomMemoryKvContractEvidenceProvider(observation),),
        evaluators=(CustomMemoryKvSchemaCompatibilityEvaluator(),),
    )


__all__ = [
    "CustomMemoryKvContractEvidenceProvider",
    "CustomMemoryKvContractObservation",
    "CustomMemoryKvSchemaCompatibilityEvaluator",
    "custom_memory_kv_compatibility_extensions",
]
