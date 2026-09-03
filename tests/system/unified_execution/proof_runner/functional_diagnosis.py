# © Artur Czarnecki. All rights reserved.

"""Central functional diagnosis seam for UE-11G-C1 production proof (R4)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Literal

from intergrax.contracts.execution_identity import validate_run_id, validate_task_id
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysis,
    FunctionalDiagnosticCheckStatus,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import FunctionalDiagnosticCheckId
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceScope,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_validation_lookup import FunctionalValidationEvidenceLookup
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    CHECK_C1_CANDIDATES,
    CHECK_C1_OUTPUT_RELATION,
    CHECK_C1_RETRIEVAL_OPERATION,
    CHECK_C1_SELECTION,
    CHECK_C1_VALIDATION,
    build_c1_rag_functional_diagnostic_specification,
)
from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
    wire_in_memory_functional_evidence_runtime,
)
from tests.system.functional_diagnostics_q1.oracle import (
    EXPECTED_INCIDENT_DATE,
    build_independent_validation_evidence,
    evidence_texts_from_lkw_response,
)
from tests.system.unified_execution.proof_runner.contracts import ProofConfig
from tests.system.unified_execution.proof_runner.lkw_client import LkwClientError

_CURSOR_SECRET = "ue-11g-c1-functional-diagnosis-local-only"
_PERSISTENCE_BACKEND = "DocumentStore via qualification API"
_DIAGNOSTIC_IDEMPOTENCY_KEY = "ue-11g-c1-r4"

_FAILURE_STAGE_BY_CHECK: dict[FunctionalDiagnosticCheckId, str] = {
    CHECK_C1_RETRIEVAL_OPERATION: "RETRIEVAL",
    CHECK_C1_CANDIDATES: "CANDIDATE/RANKING",
    CHECK_C1_SELECTION: "SELECTION",
    CHECK_C1_OUTPUT_RELATION: "SYNTHESIS",
    CHECK_C1_VALIDATION: "OUTPUT VALIDATION",
}


@dataclass(frozen=True, slots=True)
class DiagnosticCheckProjection:
    check_id: str
    status: str
    factual_claim: str


@dataclass(frozen=True, slots=True)
class FunctionalDiagnosisReport:
    invocation_status: Literal["PASS", "FAIL", "BLOCKED"]
    persistence_backend: str
    durable: bool
    evidence_kinds: tuple[str, ...]
    evidence_count: int
    validation_id: str | None
    functional_expected: str
    functional_actual_bounded: str
    diagnostic_specification_id: str | None
    diagnostic_specification_version: int | None
    diagnostic_first_proven_failure: str | None
    diagnostic_check_results: tuple[DiagnosticCheckProjection, ...]
    diagnostic_supporting_evidence_refs: tuple[str, ...]
    diagnostic_limitations: tuple[str, ...]
    failure_stage: str | None
    confidence: Literal["PROVEN", "INSUFFICIENT"]
    blocked_reason: str | None = None


def evaluate_r4_result(
    *,
    search_completed: bool,
    oracle_pass: bool,
    diagnosis: FunctionalDiagnosisReport | None,
) -> Literal["PASS", "PARTIAL", "FAIL", "BLOCKED"]:
    if not search_completed:
        return "BLOCKED"
    if diagnosis is None:
        return "FAIL"
    if diagnosis.invocation_status == "BLOCKED":
        return "BLOCKED"
    if diagnosis.invocation_status != "PASS":
        return "FAIL"
    if oracle_pass:
        return "PARTIAL"
    if diagnosis.diagnostic_first_proven_failure is not None:
        return "PASS"
    if diagnosis.confidence == "INSUFFICIENT":
        return "PARTIAL"
    return "FAIL"


def failure_stage_for_check(check_id: FunctionalDiagnosticCheckId | None) -> str | None:
    if check_id is None:
        return None
    return _FAILURE_STAGE_BY_CHECK.get(check_id)


def run_functional_diagnosis(
    config: ProofConfig,
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
    attempt_id: str | None,
    answer: str | None,
    lkw_evidence: dict[str, object] | None,
) -> FunctionalDiagnosisReport:
    functional_actual = _bounded_actual(answer)
    try:
        remote_items = fetch_functional_evidence(
            config,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        )
        evidence_kinds = _evidence_kinds(remote_items)
        pipeline_scope = PipelineEvidenceScope(
            tenant_id=tenant_id,
            task_id=validate_task_id(task_id),
            run_id=validate_run_id(run_id),
            attempt_id=attempt_id,
        )
        evidence_texts = evidence_texts_from_lkw_response(
            answer=answer,
            lkw_evidence=lkw_evidence,
        )
        validation = build_independent_validation_evidence(
            pipeline_scope,
            answer=answer,
            evidence_texts=evidence_texts,
            idempotency_key=_DIAGNOSTIC_IDEMPOTENCY_KEY,
        )
        include_output_relation = PipelineEvidenceKind.OUTPUT_RELATION in {
            item.kind for item in remote_items
        }
        spec = build_c1_rag_functional_diagnostic_specification(
            validation_id=validation.validation_id,
            include_output_relation=include_output_relation,
            include_validation=True,
        )
        wiring = wire_in_memory_functional_evidence_runtime(cursor_secret=_CURSOR_SECRET)
        for item in remote_items:
            wiring.persistence.append(item)
        validations_lookup = FunctionalValidationEvidenceLookup.for_scope(
            tenant_id=tenant_id,
            task_id=pipeline_scope.task_id,
            run_id=pipeline_scope.run_id,
            attempt_id=pipeline_scope.attempt_id,
            validations=(validation,),
        )
        analysis = FunctionalDiagnosticAnalyzer(wiring.persistence).analyze(
            tenant_id=tenant_id,
            task_id=pipeline_scope.task_id,
            run_id=pipeline_scope.run_id,
            attempt_id=pipeline_scope.attempt_id,
            specification=spec,
            validations=validations_lookup,
        )
        return _analysis_to_report(
            analysis=analysis,
            evidence_kinds=evidence_kinds,
            evidence_count=len(remote_items),
            validation_id=str(validation.validation_id),
            functional_actual=functional_actual,
        )
    except (LkwClientError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return FunctionalDiagnosisReport(
            invocation_status="FAIL",
            persistence_backend=_PERSISTENCE_BACKEND,
            durable=False,
            evidence_kinds=(),
            evidence_count=0,
            validation_id=None,
            functional_expected=EXPECTED_INCIDENT_DATE,
            functional_actual_bounded=functional_actual,
            diagnostic_specification_id=None,
            diagnostic_specification_version=None,
            diagnostic_first_proven_failure=None,
            diagnostic_check_results=(),
            diagnostic_supporting_evidence_refs=(),
            diagnostic_limitations=(str(exc),),
            failure_stage=None,
            confidence="INSUFFICIENT",
            blocked_reason=str(exc),
        )


def fetch_functional_evidence(
    config: ProofConfig,
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
) -> tuple[PlatformFunctionalEvidence, ...]:
    query = urllib.parse.urlencode(
        {"tenant_id": tenant_id, "task_id": task_id, "run_id": run_id},
    )
    url = f"{config.base_url.rstrip('/')}/v1/local_workspace/qualification/functional_evidence?{query}"
    request = urllib.request.Request(url, headers={"X-API-Key": config.api_key})
    with urllib.request.urlopen(request, timeout=120.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    items = payload.get("items")
    if not isinstance(items, list):
        raise LkwClientError("functional_evidence_items_missing")
    return tuple(
        PlatformFunctionalEvidence.model_validate(_normalize_remote_evidence_item(item))
        for item in items
    )


def _analysis_to_report(
    *,
    analysis: FunctionalDiagnosticAnalysis,
    evidence_kinds: tuple[str, ...],
    evidence_count: int,
    validation_id: str,
    functional_actual: str,
) -> FunctionalDiagnosisReport:
    first_failure = analysis.first_proven_failure
    confidence: Literal["PROVEN", "INSUFFICIENT"] = (
        "PROVEN" if first_failure is not None else "INSUFFICIENT"
    )
    supporting_refs: list[str] = []
    limitations: list[str] = list(analysis.limitations)
    check_projections: list[DiagnosticCheckProjection] = []
    for result in analysis.check_results:
        check_projections.append(
            DiagnosticCheckProjection(
                check_id=str(result.check_id),
                status=result.status.value,
                factual_claim=result.factual_claim,
            ),
        )
        for ref in result.supporting_evidence_refs:
            if len(supporting_refs) < 20:
                supporting_refs.append(str(ref))
        for limitation in result.limitations:
            if len(limitations) < 20 and limitation not in limitations:
                limitations.append(limitation)
    return FunctionalDiagnosisReport(
        invocation_status="PASS",
        persistence_backend=_PERSISTENCE_BACKEND,
        durable=evidence_count > 0,
        evidence_kinds=evidence_kinds,
        evidence_count=evidence_count,
        validation_id=validation_id,
        functional_expected=EXPECTED_INCIDENT_DATE,
        functional_actual_bounded=functional_actual,
        diagnostic_specification_id=str(analysis.specification_id),
        diagnostic_specification_version=analysis.specification_version,
        diagnostic_first_proven_failure=(
            str(first_failure) if first_failure is not None else None
        ),
        diagnostic_check_results=tuple(check_projections),
        diagnostic_supporting_evidence_refs=tuple(supporting_refs),
        diagnostic_limitations=tuple(limitations[:20]),
        failure_stage=failure_stage_for_check(first_failure),
        confidence=confidence,
    )


def _bounded_actual(answer: str | None) -> str:
    if not answer:
        return ""
    return answer[:200]


def _evidence_kinds(items: tuple[PlatformFunctionalEvidence, ...]) -> tuple[str, ...]:
    kinds: list[str] = []
    for item in items:
        kind = item.kind.value
        if kind not in kinds:
            kinds.append(kind)
    return tuple(kinds)


def _normalize_remote_evidence_item(item: object) -> object:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    provenance = normalized.get("provenance")
    if isinstance(provenance, dict):
        prov = dict(provenance)
        upstream = prov.get("upstream_evidence_ids")
        if isinstance(upstream, list):
            prov["upstream_evidence_ids"] = tuple(upstream)
        normalized["provenance"] = prov
    return normalized


__all__ = [
    "DiagnosticCheckProjection",
    "FunctionalDiagnosisReport",
    "evaluate_r4_result",
    "failure_stage_for_check",
    "fetch_functional_evidence",
    "run_functional_diagnosis",
]
