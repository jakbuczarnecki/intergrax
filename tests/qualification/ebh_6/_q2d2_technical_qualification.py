# © Artur Czarnecki. All rights reserved.

"""Q2-D2 technical Path A eligibility — ADR scope vs production seam (not operator OD)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

AUTHORITATIVE_ADR_PATH: Final[str] = (
    "docs/project/architecture/SCENARIO_1_P0_B_D1_AUTHORITATIVE_DECISION_RESULT_EXPOSURE.md"
)
NEXUS_LOOP_PATH: Final[str] = "intergrax/runtime/nexus/nexus_loop.py"
NEXUS_EXPOSURE_RESOLVER: Final[str] = "intergrax/runtime/nexus/orchestration/nexus_decision_exposure.py"

PATH_A_ELIGIBLE: Final[str] = "PATH A — ELIGIBLE FOR OPERATOR RATIFICATION"
PATH_A_NOT_ELIGIBLE: Final[str] = "PATH A NOT ELIGIBLE"
PATH_B_MANDATORY: Final[str] = "PATH B — MANDATORY REMEDIATION"

PATH_B_PRODUCTION_SCOPE: Final[str] = (
    "OD-EBH6-01: narrow NexusLoop._resolve_authoritative_decision_exposure and TaskResult "
    "projection to AuthoritativeDecisionExposure[object] | None (FRZ-TYP-03); regression via "
    "tests/unit/runtime/nexus/test_authoritative_decision_exposure_propagation.py @ new SHA."
)


@dataclass(frozen=True, slots=True)
class Q2D2AdrAssessment:
    adr_path: str
    adr_exists: bool
    adr_covers_exposure_model: bool
    adr_covers_nexus_object_erasure_debt: bool
    resolver_returns_typed_exposure: bool
    nexus_loop_uses_object_erasure: bool


def assess_q2d2_adr_scope() -> Q2D2AdrAssessment:
    adr_file = _REPO_ROOT / AUTHORITATIVE_ADR_PATH
    adr_text = adr_file.read_text(encoding="utf-8") if adr_file.is_file() else ""
    resolver = (_REPO_ROOT / NEXUS_EXPOSURE_RESOLVER).read_text(encoding="utf-8")
    nexus = (_REPO_ROOT / NEXUS_LOOP_PATH).read_text(encoding="utf-8")
    covers_model = (
        "AuthoritativeDecisionExposure" in adr_text
        and "TaskResult" in adr_text
        and ("decision_flow_result_to_authoritative_exposure" in adr_text or "DecisionFlowResult" in adr_text)
    )
    adr_accepts_erasure = (
        "object | None" in adr_text
        and "nexus_loop" in adr_text.lower()
        and ("accepted" in adr_text.lower() or "deferred" in adr_text.lower())
    )
    resolver_typed = "-> AuthoritativeDecisionExposure[object]" in resolver
    nexus_erasure = "-> object | None:" in nexus and "_resolve_authoritative_decision_exposure" in nexus
    return Q2D2AdrAssessment(
        adr_path=AUTHORITATIVE_ADR_PATH,
        adr_exists=adr_file.is_file(),
        adr_covers_exposure_model=covers_model,
        adr_covers_nexus_object_erasure_debt=adr_accepts_erasure,
        resolver_returns_typed_exposure=resolver_typed,
        nexus_loop_uses_object_erasure=nexus_erasure,
    )


def technical_q2d2_disposition(assessment: Q2D2AdrAssessment) -> str:
    if not assessment.adr_exists or not assessment.adr_covers_exposure_model:
        return PATH_A_NOT_ELIGIBLE
    if assessment.nexus_loop_uses_object_erasure and not assessment.adr_covers_nexus_object_erasure_debt:
        return PATH_A_NOT_ELIGIBLE
    if not assessment.resolver_returns_typed_exposure:
        return PATH_B_MANDATORY
    return PATH_A_ELIGIBLE


def replay_q2d2_fail_closed_behavioral_gates() -> None:
    from tests.unit.runtime.nexus import test_authoritative_decision_exposure_propagation as ade

    ade.test_b13_ambiguous_candidates_fail_closed()
    ade.test_b18_terminal_without_exposure_raises()
    ade.test_r1_n2_gate_configured_execution_failed_before_decision()
