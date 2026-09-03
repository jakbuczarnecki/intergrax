# © Artur Czarnecki. All rights reserved.

"""Deterministic functional evidence workload generator for scale qualification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_event_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineArtifactLineageFact,
    PipelineCandidateFact,
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PipelineOutputRelationFact,
    PipelineSelectionFact,
    PipelineValidationLinkFact,
    PlatformFunctionalEvidence,
    ScoreSemantics,
    TypedPipelineScore,
)
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    C1_RAG_EXPECTED_SELECTION_ARTIFACT,
    C1_RAG_QUERY_ID,
    C1_RAG_RETRIEVE_OPERATION_ID,
)
from intergrax.runtime.diagnostics.specifications.q2_tool_selection_functional_diagnostic_specification import (
    Q2_EXPECTED_SEARCH_TOOL_ARTIFACT,
    Q2_TOOL_INVOKE_OPERATION_ID,
    Q2_TOOL_QUERY_ID,
)
from intergrax.runtime.diagnostics.specifications.q3_web_search_functional_diagnostic_specification import (
    Q3_WEB_QUERY_ID,
    Q3_WEB_SEARCH_OPERATION_ID,
)
from intergrax.runtime.diagnostics.specifications.q4_model_routing_functional_diagnostic_specification import (
    Q4_MODEL_GENERATE_OPERATION_ID,
    Q4_MODEL_QUERY_ID,
)
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference
from intergrax.runtime.diagnostics.functional_evidence_record_codec import (
    encode_functional_evidence_record,
)
from tests.system.functional_diagnostics_scale.manifest import (
    ScaleDatasetManifest,
    ScaleExecutionManifestEntry,
)
from tests.system.functional_diagnostics_scale.profile import FunctionalDiagnosticsScaleProfile

_BASE_TIME = datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc)
_ALL_KINDS = tuple(PipelineEvidenceKind)
_WORKLOAD_DOMAINS = ("rag", "tools", "web", "model_routing")


def _digest_hex(*parts: str) -> str:
    material = "|".join(parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _deterministic_id(prefix: str, *parts: str) -> str:
    return f"{prefix}{_digest_hex(*parts)[:32]}"


def _evidence_fingerprint(evidence: PlatformFunctionalEvidence) -> str:
    encoded = encode_functional_evidence_record(evidence)
    canonical = json.dumps(encoded, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ScaleExecutionIdentity:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_index: int
    is_heavy: bool
    analyzer_sample: bool


class FunctionalEvidenceWorkloadGenerator:
    """Deterministic typed workload from seed + profile."""

    def __init__(self, profile: FunctionalDiagnosticsScaleProfile) -> None:
        self._profile = profile

    def tenant_id(self, tenant_index: int) -> str:
        return f"{self._profile.tenant_namespace}-{tenant_index:04d}-seed-{self._profile.seed}"

    def execution_identities(self) -> tuple[ScaleExecutionIdentity, ...]:
        identities: list[ScaleExecutionIdentity] = []
        for tenant_index in range(self._profile.tenant_count):
            tenant_id = self.tenant_id(tenant_index)
            for execution_index in range(self._profile.execution_count_per_tenant):
                is_heavy = (
                    execution_index
                    >= self._profile.execution_count_per_tenant
                    - self._profile.heavy_execution_count_per_tenant
                )
                analyzer_sample = (
                    not is_heavy
                    and execution_index < self._profile.analyzer_sample_executions_per_tenant
                )
                identities.append(
                    ScaleExecutionIdentity(
                        tenant_id=tenant_id,
                        task_id=validate_task_id(
                            _deterministic_id(
                                "task_",
                                str(self._profile.seed),
                                tenant_id,
                                str(execution_index),
                            ),
                        ),
                        run_id=validate_run_id(
                            _deterministic_id(
                                "run_",
                                str(self._profile.seed),
                                tenant_id,
                                str(execution_index),
                            ),
                        ),
                        attempt_id=validate_attempt_id(
                            _deterministic_id(
                                "attempt_",
                                str(self._profile.seed),
                                tenant_id,
                                str(execution_index),
                            ),
                        ),
                        execution_index=execution_index,
                        is_heavy=is_heavy,
                        analyzer_sample=analyzer_sample,
                    ),
                )
        return tuple(identities)

    def evidence_for_execution(
        self,
        identity: ScaleExecutionIdentity,
    ) -> tuple[PlatformFunctionalEvidence, ...]:
        if identity.is_heavy:
            return self._heavy_execution_evidence(identity)
        if identity.analyzer_sample:
            return self._analyzer_sample_evidence(identity)
        return self._typical_execution_evidence(identity)

    def build_manifest(self) -> ScaleDatasetManifest:
        entries: list[ScaleExecutionManifestEntry] = []
        total = 0
        tenant_ids = [
            self.tenant_id(tenant_index)
            for tenant_index in range(self._profile.tenant_count)
        ]
        for identity in self.execution_identities():
            evidence = self.evidence_for_execution(identity)
            fingerprints = tuple(_evidence_fingerprint(item) for item in evidence)
            entries.append(
                ScaleExecutionManifestEntry(
                    tenant_id=identity.tenant_id,
                    task_id=identity.task_id,
                    run_id=identity.run_id,
                    attempt_id=identity.attempt_id,
                    evidence_ids=tuple(str(item.evidence_id) for item in evidence),
                    evidence_fingerprints=fingerprints,
                    is_heavy=identity.is_heavy,
                    analyzer_sample=identity.analyzer_sample,
                ),
            )
            total += len(evidence)
        return ScaleDatasetManifest(
            seed=self._profile.seed,
            profile_name=self._profile.name.value,
            entries=tuple(entries),
            total_evidence=total,
            tenant_ids=tuple(tenant_ids),
        )

    def all_evidence(self) -> tuple[PlatformFunctionalEvidence, ...]:
        items: list[PlatformFunctionalEvidence] = []
        for identity in self.execution_identities():
            items.extend(self.evidence_for_execution(identity))
        return tuple(items)

    def _typical_execution_evidence(
        self,
        identity: ScaleExecutionIdentity,
    ) -> tuple[PlatformFunctionalEvidence, ...]:
        count = self._profile.typical_evidence_per_execution
        items: list[PlatformFunctionalEvidence] = []
        for index in range(count):
            kind = _ALL_KINDS[index % len(_ALL_KINDS)]
            domain = _WORKLOAD_DOMAINS[index % len(_WORKLOAD_DOMAINS)]
            items.append(
                self._build_evidence(
                    identity=identity,
                    index=index,
                    kind=kind,
                    domain=domain,
                    analyzer_rich=False,
                ),
            )
        return tuple(items)

    def _heavy_execution_evidence(
        self,
        identity: ScaleExecutionIdentity,
    ) -> tuple[PlatformFunctionalEvidence, ...]:
        count = self._profile.heavy_evidence_per_execution
        items: list[PlatformFunctionalEvidence] = []
        for index in range(count):
            domain = _WORKLOAD_DOMAINS[index % len(_WORKLOAD_DOMAINS)]
            items.append(
                self._build_evidence(
                    identity=identity,
                    index=index,
                    kind=PipelineEvidenceKind.OPERATION_OUTCOME,
                    domain=domain,
                    analyzer_rich=False,
                ),
            )
        return tuple(items)

    def _analyzer_sample_evidence(
        self,
        identity: ScaleExecutionIdentity,
    ) -> tuple[PlatformFunctionalEvidence, ...]:
        specs: tuple[tuple[PipelineEvidenceKind, str, int], ...] = (
            (PipelineEvidenceKind.OPERATION_OUTCOME, "rag", 0),
            (PipelineEvidenceKind.CANDIDATE_RANK, "rag", 1),
            (PipelineEvidenceKind.SELECTION, "rag", 2),
            (PipelineEvidenceKind.OPERATION_OUTCOME, "tools", 3),
            (PipelineEvidenceKind.CANDIDATE_RANK, "web", 4),
            (PipelineEvidenceKind.SELECTION, "model_routing", 5),
            (PipelineEvidenceKind.ARTIFACT_LINEAGE, "tools", 6),
            (PipelineEvidenceKind.OUTPUT_RELATION, "model_routing", 7),
            (PipelineEvidenceKind.VALIDATION, "web", 8),
        )
        return tuple(
            self._build_evidence(
                identity=identity,
                index=index,
                kind=kind,
                domain=domain,
                analyzer_rich=True,
            )
            for kind, domain, index in specs
        )

    def _build_evidence(
        self,
        *,
        identity: ScaleExecutionIdentity,
        index: int,
        kind: PipelineEvidenceKind,
        domain: str,
        analyzer_rich: bool,
    ) -> PlatformFunctionalEvidence:
        evidence_id = validate_event_id(
            _deterministic_id(
                "evt_",
                str(self._profile.seed),
                identity.tenant_id,
                str(identity.execution_index),
                str(index),
                kind.value,
            ),
        )
        scope = PipelineEvidenceScope(
            tenant_id=identity.tenant_id,
            task_id=identity.task_id,
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
        )
        recorded_at = _BASE_TIME + timedelta(
            seconds=identity.execution_index,
            milliseconds=index,
        )
        operation_id = _domain_operation_id(domain, analyzer_rich=analyzer_rich)
        provenance = PipelineEvidenceProvenance(
            producer_component=f"domain.{domain}",
            operation_id=operation_id,
            recorded_at=recorded_at,
        )
        if kind is PipelineEvidenceKind.ARTIFACT_LINEAGE:
            payload = {
                "artifact_lineage": PipelineArtifactLineageFact(
                    source_artifact_ref=ObservabilityArtifactReference(
                        artifact_ref=f"doc:{domain}:{index}",
                    ),
                    derived_artifact_ref=ObservabilityArtifactReference(
                        artifact_ref=f"chunk:{domain}:{index}",
                    ),
                    lineage_operation="chunk",
                ),
            }
        elif kind is PipelineEvidenceKind.OPERATION_OUTCOME:
            payload = {
                "operation_outcome": PipelineOperationOutcomeFact(
                    operation_name=f"{domain}.op.{index}",
                    status=PipelineOperationStatus.SUCCEEDED,
                ),
            }
        elif kind is PipelineEvidenceKind.CANDIDATE_RANK:
            payload = {
                "candidate": PipelineCandidateFact(
                    query_id=_domain_query_id(domain),
                    candidate_artifact_ref=ObservabilityArtifactReference(
                        artifact_ref=f"candidate:{domain}:{index}",
                    ),
                    score=TypedPipelineScore(
                        raw_value=0.5 + (index % 5) * 0.1,
                        semantics=ScoreSemantics.HIGHER_IS_BETTER,
                    ),
                    rank=1 + (index % 3),
                    selected=index % 2 == 0,
                ),
            }
        elif kind is PipelineEvidenceKind.SELECTION:
            payload = {
                "selection": PipelineSelectionFact(
                    query_id=_domain_query_id(domain),
                    selected_artifact_ref=ObservabilityArtifactReference(
                        artifact_ref=_domain_selection_artifact(domain, analyzer_rich),
                    ),
                    candidate_count=2 + (index % 2),
                    selection_reason="top_score",
                ),
            }
        elif kind is PipelineEvidenceKind.OUTPUT_RELATION:
            payload = {
                "output_relation": PipelineOutputRelationFact(
                    selected_artifact_ref=ObservabilityArtifactReference(
                        artifact_ref=f"selected:{domain}:{index}",
                    ),
                    output_artifact_ref=ObservabilityArtifactReference(
                        artifact_ref=f"output:{domain}:{index}",
                    ),
                    relation_kind="derived_from",
                ),
            }
        else:
            payload = {
                "validation_link": PipelineValidationLinkFact(
                    validation_id=validate_event_id(
                        _deterministic_id(
                            "evt_",
                            "validation",
                            str(self._profile.seed),
                            identity.tenant_id,
                            str(identity.execution_index),
                            str(index),
                        ),
                    ),
                    output_artifact_ref=ObservabilityArtifactReference(
                        artifact_ref=f"output:{domain}:{index}",
                    ),
                ),
            }
        return PlatformFunctionalEvidence(
            evidence_id=evidence_id,
            kind=kind,
            scope=scope,
            provenance=provenance,
            **payload,
        )


def _domain_operation_id(domain: str, *, analyzer_rich: bool) -> str:
    if domain == "rag" and analyzer_rich:
        return C1_RAG_RETRIEVE_OPERATION_ID
    if domain == "tools" and analyzer_rich:
        return Q2_TOOL_INVOKE_OPERATION_ID
    if domain == "web" and analyzer_rich:
        return Q3_WEB_SEARCH_OPERATION_ID
    if domain == "model_routing" and analyzer_rich:
        return Q4_MODEL_GENERATE_OPERATION_ID
    return f"{domain}.operation"


def _domain_query_id(domain: str) -> str:
    if domain == "rag":
        return C1_RAG_QUERY_ID
    if domain == "tools":
        return Q2_TOOL_QUERY_ID
    if domain == "web":
        return Q3_WEB_QUERY_ID
    return Q4_MODEL_QUERY_ID


def _domain_selection_artifact(domain: str, analyzer_rich: bool) -> str:
    if domain == "rag" and analyzer_rich:
        return C1_RAG_EXPECTED_SELECTION_ARTIFACT
    if domain == "tools" and analyzer_rich:
        return Q2_EXPECTED_SEARCH_TOOL_ARTIFACT
    return f"selection:{domain}"


__all__ = [
    "FunctionalEvidenceWorkloadGenerator",
    "ScaleExecutionIdentity",
]
