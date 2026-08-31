# © Artur Czarnecki. All rights reserved.

"""Functional oracle for UE-11G-C1 workspace search."""

from __future__ import annotations

from tests.system.unified_execution.proof_runner.contracts import LkwRunResponse

_EXPECTED_FACT = "2026-08-17"
_SEARCH_QUESTION = "When did Incident Orion occur?"


def search_request_message() -> str:
    return _SEARCH_QUESTION


def expected_fact() -> str:
    return _EXPECTED_FACT


def functional_oracle_passes(response: LkwRunResponse) -> bool:
    if response.answer and _EXPECTED_FACT in response.answer:
        return True
    if response.lkw_evidence is None:
        return False
    search_diag = response.lkw_evidence.diagnostics.get("lkw.search_summary.v1")
    if search_diag is None:
        return False
    evidence_count = search_diag.evidence_count or 0
    num_results = search_diag.num_results or 0
    if evidence_count <= 0 and num_results <= 0:
        return False
    source_refs = search_diag.source_refs or []
    incident_ref = any("incident-report" in ref for ref in source_refs)
    if incident_ref and response.answer and "orion" in response.answer.lower():
        return True
    if response.answer and _EXPECTED_FACT in response.answer:
        return True
    return False
