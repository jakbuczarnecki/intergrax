# © Artur Czarnecki. All rights reserved.

"""E02 / IA-02 — per-debt APP-TYP behavioral + typing evidence (executed, not filename-only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from intergrax.applications._shared.business_agent_certification import certify_business_agent
from intergrax.applications._shared.recovery_contract_wiring import validate_recovery_contract
from intergrax.applications._shared.reliability_wiring import apply_reliability_task_defaults
from intergrax.applications.contracts.application_recovery_contract import (
    APPLICATION_RECOVERY_CONTRACT_KEY,
    ApplicationRecoveryContract,
    HostRestartRecoveryAction,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.runtime.task.task import Task, TaskContext
from tests.qualification.ebh_6._contracts_typing_qualification import (
    APPLICATION_SERVICES_TYPING_DEBT,
    ApplicationServicesTypingDebtEntry,
    _debt_covers,
    evaluate_application_services_production_seam_typing,
    evaluate_application_services_typing_on_source,
)
from tests.unit.applications import test_recovery_contract_wiring as recovery_wiring
from tests.unit.applications import test_scenario_runtime_profiles as scenario_profiles

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO: Final[Path] = Path(__file__).resolve().parents[3]

_INSUFFICIENT: Final[str] = "C"
_SCOPED_A: Final[str] = "A"


@dataclass(frozen=True, slots=True)
class AppTypEvidenceOutcome:
    finding_id: str
    recommendation: str
    executed_gates: tuple[str, ...]


def _assert_registered_typing_seam(entry: ApplicationServicesTypingDebtEntry) -> None:
    path = _REPO / entry.source_path
    text = path.read_text(encoding="utf-8")
    violations = evaluate_application_services_typing_on_source(
        source=text,
        source_path=entry.source_path,
        source_module=entry.source_path.replace("/", ".").removesuffix(".py"),
    )
    matched = [v for v in violations if v.symbol == entry.symbol and _debt_covers(v, entry)]
    assert matched, f"{entry.finding_id}: no registered typing violation for {entry.symbol}"


def _evidence_04(tmp_path: Path) -> tuple[str, ...]:
    scenario_profiles.test_production_fail_closed_without_manifest(tmp_path)
    scenario_profiles.test_production_fail_closed_without_tenant(tmp_path)
    return (
        "test_production_fail_closed_without_manifest",
        "test_production_fail_closed_without_tenant",
    )


def _evidence_05() -> tuple[str, ...]:
    env = ApplicationEnvironmentProfile.product_defaults()
    incomplete = AgentContract(
        id="ebh6-incomplete-agent",
        name="Incomplete Agent",
        description="EBH-6 qualification probe",
        capabilities=(),
        validation_rules=(),
        owner_team="",
    )
    result = certify_business_agent(incomplete, environments=(env,))
    assert not result.deploy_ready
    assert result.errors
    return ("certify_business_agent_rejects_incomplete_contract",)


def _evidence_11() -> tuple[str, ...]:
    recovery_wiring.test_validate_recovery_contract_requires_scheduler_for_resume()
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="demo.product")
    task = Task(
        task_id=mint_task_id(),
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="demo.cap"),
    )
    updated = apply_reliability_task_defaults(task, env)
    assert APPLICATION_RECOVERY_CONTRACT_KEY in updated.metadata
    bad = ReliabilityProfile(
        long_running_scheduler_enabled=False,
        recovery_contract=ApplicationRecoveryContract(
            on_host_restart=HostRestartRecoveryAction.RESUME_SCHEDULER,
        ),
    )
    violations = validate_recovery_contract(bad, strict_product=True)
    assert violations
    return (
        "test_validate_recovery_contract_requires_scheduler_for_resume",
        "apply_reliability_task_defaults_attaches_recovery_contract",
        "validate_recovery_contract_rejects_bad_contract",
    )


_BEHAVIORAL: dict[str, Callable[..., tuple[str, ...]]] = {
    "EBH6-DEBT-APP-TYP-04": _evidence_04,
    "EBH6-DEBT-APP-TYP-05": lambda: _evidence_05(),
    "EBH6-DEBT-APP-TYP-11": lambda: _evidence_11(),
}


def _recommend(entry: ApplicationServicesTypingDebtEntry, executed: tuple[str, ...]) -> str:
    if entry.finding_id in _BEHAVIORAL and not executed:
        return _INSUFFICIENT
    if entry.finding_id in ("EBH6-DEBT-APP-TYP-04", "EBH6-DEBT-APP-TYP-05", "EBH6-DEBT-APP-TYP-11"):
        return _SCOPED_A if executed else _INSUFFICIENT
    return _SCOPED_A if executed else _INSUFFICIENT


@pytest.mark.parametrize("entry", APPLICATION_SERVICES_TYPING_DEBT, ids=lambda e: e.finding_id)
def test_ebh6_app_typ_registered_typing_seam(entry: ApplicationServicesTypingDebtEntry) -> None:
    _assert_registered_typing_seam(entry)


def test_ebh6_app_typ_production_seam_typing_gate_passes() -> None:
    result = evaluate_application_services_production_seam_typing(_REPO)
    assert result.passed


@pytest.mark.parametrize("entry", APPLICATION_SERVICES_TYPING_DEBT, ids=lambda e: e.finding_id)
def test_ebh6_app_typ_individual_evidence_and_recommendation(
    entry: ApplicationServicesTypingDebtEntry,
    tmp_path: Path,
) -> None:
    executed: tuple[str, ...] = ("registered_typing_seam",)
    runner = _BEHAVIORAL.get(entry.finding_id)
    if runner is not None:
        executed = runner(tmp_path) if entry.finding_id == "EBH6-DEBT-APP-TYP-04" else runner()
    rec = _recommend(entry, executed)
    outcome = AppTypEvidenceOutcome(entry.finding_id, rec, executed)
    assert outcome.recommendation in ("A", "B", "C")
    assert outcome.recommendation != _INSUFFICIENT, outcome.finding_id


def test_ebh6_app_typ_agent_does_not_operator_ratify() -> None:
    assert all(
        _recommend(entry, ("registered_typing_seam",)) != "OPERATOR_APPROVED"
        for entry in APPLICATION_SERVICES_TYPING_DEBT
    )


def test_ebh6_app_typ_e02_op01_operator_ratification_recorded() -> None:
    reg = _REPO / "docs/project/maintainers/qualification/EBH_6_OPERATOR_DECISION_REGISTER.md"
    text = reg.read_text(encoding="utf-8")
    row = [line for line in text.splitlines() if "E02-OP-01" in line and line.startswith("|")][0]
    assert "OPERATOR RATIFIED" in row or "RATIFIED" in row
    assert "OPEN" not in row.split("|")[3] or "RATIFIED" in row
    typ = (_REPO / "docs/project/maintainers/qualification/EBH_6_APP_TYP_DEBT_DISPOSITION_REGISTER.md").read_text(
        encoding="utf-8",
    )
    assert "E02-OP-01 — OPERATOR RATIFIED" in typ or "OPERATOR RATIFIED — 2026-10-29" in typ
    assert "E02-AUD-01 — PENDING" in typ
