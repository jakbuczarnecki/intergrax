# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic enrichment for legacy platform causal evidence (NPSC-5F)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.npsc5f_compatibility import (
    AmbiguousLegacyExecutionIdentityError,
    LegacyCausalEvidenceIncompatibleError,
)
from intergrax.runtime.observability.causal_evidence import (
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.causal_evidence_legacy import (
    LegacyPlatformCausalEvidence,
)
from intergrax.runtime.observability.platform_causal_evidence_codec import (
    DecodedPlatformCausalEvidence,
)


@dataclass(frozen=True, slots=True)
class CanonicalExecutionIdLookupResult:
    outcome: Literal["resolved", "unresolved", "ambiguous"]
    execution_id: ExecutionId | None = None


class CanonicalExecutionIdLookupPort(Protocol):
    """Bounded canonical lookup — observability must not mint via this port."""

    def lookup_execution_id(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        attempt_id: AttemptId,
    ) -> CanonicalExecutionIdLookupResult: ...


@dataclass(frozen=True, slots=True)
class CausalEvidenceEnrichmentOutcome:
    status: Literal["resolved", "legacy_incomplete", "incompatible"]
    complete_v2: PlatformCausalEvidence | None = None
    legacy_v1: LegacyPlatformCausalEvidence | None = None


def enrich_decoded_causal_evidence(
    decoded: DecodedPlatformCausalEvidence,
    *,
    lookup: CanonicalExecutionIdLookupPort | None = None,
) -> CausalEvidenceEnrichmentOutcome:
    if decoded.kind == "complete_v2" and decoded.complete_v2 is not None:
        return CausalEvidenceEnrichmentOutcome(
            status="resolved",
            complete_v2=decoded.complete_v2,
        )
    if decoded.kind != "legacy_incomplete_v1" or decoded.legacy_v1 is None:
        return CausalEvidenceEnrichmentOutcome(status="incompatible")
    legacy = decoded.legacy_v1
    if lookup is None:
        return CausalEvidenceEnrichmentOutcome(
            status="legacy_incomplete", legacy_v1=legacy
        )
    lookup_result = lookup.lookup_execution_id(
        tenant_id=legacy.tenant_id,
        task_id=legacy.target.task_id,
        run_id=legacy.target.run_id,
        attempt_id=legacy.target.attempt_id,
    )
    if lookup_result.outcome == "unresolved":
        raise LegacyCausalEvidenceIncompatibleError(
            "legacy causal evidence missing ExecutionId and canonical lookup unresolved",
        )
    if lookup_result.outcome == "ambiguous":
        raise AmbiguousLegacyExecutionIdentityError(
            "legacy causal evidence missing ExecutionId and canonical lookup ambiguous",
        )
    if lookup_result.execution_id is None:
        raise LegacyCausalEvidenceIncompatibleError(
            "canonical lookup returned resolved without ExecutionId",
        )
    target = RuntimeExecutionRef(
        task_id=legacy.target.task_id,
        run_id=legacy.target.run_id,
        attempt_id=legacy.target.attempt_id,
        execution_id=lookup_result.execution_id,
        tenant_id=legacy.target.tenant_id,
    )
    complete = PlatformCausalEvidence(
        evidence_id=legacy.evidence_id,
        relation_kind=legacy.relation_kind,
        tenant_id=legacy.tenant_id,
        source=legacy.source,
        target=target,
        recorded_at=legacy.recorded_at,
    )
    return CausalEvidenceEnrichmentOutcome(status="resolved", complete_v2=complete)


def require_enriched_v2(
    outcome: CausalEvidenceEnrichmentOutcome,
) -> PlatformCausalEvidence:
    if outcome.status != "resolved" or outcome.complete_v2 is None:
        raise LegacyCausalEvidenceIncompatibleError(
            "complete platform_causal_evidence.v2 required after enrichment",
        )
    return outcome.complete_v2
