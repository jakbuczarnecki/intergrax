# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-E (F-D5-03, F-D5-02): promotion and gate checks that only ever deny more."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation
from intergrax.runtime.architecture.agent_promotion import (
    PromotionEvidenceBundle,
    PromotionStage,
    evaluate_agent_promotion,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
GATE_SCRIPT = REPO_ROOT / "scripts" / "gates" / "check_mvp_promotion_gates.py"


def _bundle(cert_agent: str = "agent:research", cert_version: str = "1.0.0") -> PromotionEvidenceBundle:
    return PromotionEvidenceBundle(
        agent_id="agent:research",
        agent_version="1.0.0",
        source_stage=PromotionStage.STAGING,
        target_stage=PromotionStage.PRODUCTION,
        certification=AgentCertificationEvaluation(
            agent_id=cert_agent, agent_version=cert_version, eligible=True, reasons=[]
        ),
        evaluation_report_refs=["eval/report.json"],
        rollback_plan_ref="runbook/rollback",
        change_ticket_ref="CHG-1",
    )


def test_matching_certification_still_authorizes() -> None:
    decision = evaluate_agent_promotion(_bundle())
    assert decision.governance_authorized is True
    assert decision.approved is True


@pytest.mark.parametrize(
    ("agent", "version"),
    [("agent:other", "1.0.0"), ("agent:research", "9.9.9"), ("", ""), (" agent:research ", "1.0.0")],
)
def test_certification_of_another_agent_or_version_never_authorizes(agent: str, version: str) -> None:
    decision = evaluate_agent_promotion(_bundle(agent, version))
    assert decision.governance_authorized is False
    assert decision.approved is False
    assert any("certification" in reason and "promoted agent" in reason for reason in decision.reasons)


def _gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("d5r_mvp_gate", GATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G0_FILES = (
    "intergrax/scaffold/cli.py",
    "intergrax/cli/doctor.py",
    "intergrax/cli/run.py",
    "applications/lab_application/host/factory.py",
)
G2_FILES = (
    "intergrax/contracts/resilience_policy.py",
    "intergrax/contracts/autonomy_level.py",
    "intergrax/runtime/policy/autonomy_resolver.py",
    "scripts/maintenance/check_harness_resilience_policy.py",
)


def _hollow_root(tmp_path: Path, files: tuple[str, ...]) -> Path:
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# hollow module\n", encoding="utf-8")
    return tmp_path


def test_g0_rejects_hollow_modules(tmp_path: Path) -> None:
    ok, detail = _gate().gate_g0_runnable(_hollow_root(tmp_path, G0_FILES))
    assert ok is False
    assert "lack required definitions" in detail


def test_g2_rejects_hollow_modules(tmp_path: Path) -> None:
    ok, detail = _gate().gate_g2_policy(_hollow_root(tmp_path, G2_FILES))
    assert ok is False
    assert "lack required definitions" in detail


def test_g0_and_g2_pass_on_the_real_repository() -> None:
    gate = _gate()
    assert gate.gate_g0_runnable(REPO_ROOT)[0] is True
    assert gate.gate_g2_policy(REPO_ROOT)[0] is True


def _ticket(tmp_path: Path, text: str) -> str:
    folder = tmp_path / "governance" / "tickets"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "T-IGX-999-A.md").write_text(text, encoding="utf-8")
    return "governance/tickets/T-IGX-999-A.md"


def test_approved_line_inside_the_section_authorizes(tmp_path: Path) -> None:
    ticket = _ticket(tmp_path, "# T\n\n## Approved\n\nApproved: Operator @ 2026-11-15\n\n## Test contract\n")
    assert _gate().governance_authorization(ticket, tmp_path)[0] is True


def test_approved_line_outside_the_section_does_not_authorize(tmp_path: Path) -> None:
    ticket = _ticket(
        tmp_path,
        "# T\n\n## Approved\n\n(pending)\n\n## Notes\n\nApproved: Operator @ 2026-11-15\n",
    )
    assert _gate().governance_authorization(ticket, tmp_path)[0] is False


def test_ticket_without_an_approved_section_does_not_authorize(tmp_path: Path) -> None:
    ticket = _ticket(tmp_path, "# T\n\nApproved: Operator @ 2026-11-15\n")
    assert _gate().governance_authorization(ticket, tmp_path)[0] is False


def test_placeholder_in_the_section_is_still_refused(tmp_path: Path) -> None:
    ticket = _ticket(tmp_path, "# T\n\n## Approved\n\nApproved: pending\n")
    assert _gate().governance_authorization(ticket, tmp_path)[0] is False


def test_delegation_line_must_be_inside_the_section(tmp_path: Path) -> None:
    outside = _ticket(
        tmp_path,
        "# T\n\n## Approved\n\nApproved: operator-delegate @ 2026-11-15\n\n## Notes\n\nDelegation: x@y\n",
    )
    assert _gate().governance_authorization(outside, tmp_path)[0] is False
    inside = _ticket(
        tmp_path,
        "# T\n\n## Approved\n\nApproved: operator-delegate @ 2026-11-15\nDelegation: x@y\n\n## Notes\n",
    )
    assert _gate().governance_authorization(inside, tmp_path)[0] is True


def test_missing_ticket_and_foreign_path_are_refused(tmp_path: Path) -> None:
    gate = _gate()
    assert gate.governance_authorization(None, tmp_path)[0] is False
    assert gate.governance_authorization("governance/tickets/none.md", tmp_path)[0] is False
