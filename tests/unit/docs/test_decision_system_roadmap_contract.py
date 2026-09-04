# © Artur Czarnecki. All rights reserved.

"""Decision System roadmap and architecture documentation reality gates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]

_PLAN_SYSTEM = REPO_ROOT / "docs/project/maintainers/plans/DECISION_SYSTEM.md"
_PLAN_VERIFICATION = REPO_ROOT / "docs/project/maintainers/plans/DECISION_VERIFICATION.md"
_PLAN_DELIBERATION = REPO_ROOT / "docs/project/maintainers/plans/DECISION_DELIBERATION.md"

_ARCH_SYSTEM = REPO_ROOT / "docs/project/architecture/DECISION_SYSTEM.md"
_ARCH_VERIFICATION = REPO_ROOT / "docs/project/architecture/DECISION_VERIFICATION.md"
_ARCH_DELIBERATION = REPO_ROOT / "docs/project/architecture/DECISION_DELIBERATION.md"

_CANONICAL_ACTIVE_DECISION_DOCS = (
    _PLAN_SYSTEM,
    _PLAN_VERIFICATION,
    _PLAN_DELIBERATION,
    _ARCH_SYSTEM,
    _ARCH_VERIFICATION,
    _ARCH_DELIBERATION,
)

_STALE_CRITIC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Implementation\s+NOT\s+STARTED", re.I), "Implementation NOT STARTED"),
    (re.compile(r"NOT\s+YET\s+MIGRATED", re.I), "NOT YET MIGRATED"),
    (re.compile(r"no\s+Decision\s+System\s+(runtime\s+)?classes\s+shipped", re.I), "no Decision classes shipped"),
    (re.compile(r"production\s+remains\s+CVL\s*/\s*Critic", re.I), "production remains CVL / Critic"),
    (re.compile(r"production\s+uses\s+Critic", re.I), "production uses Critic"),
    (re.compile(r"CriticOrchestrator.*\bCURRENT\b", re.I), "CriticOrchestrator CURRENT"),
    (re.compile(r"\bCURRENT\b.*CriticOrchestrator", re.I), "CURRENT CriticOrchestrator"),
    (re.compile(r"intergrax/runtime/critic/\*\*\s+until", re.I), "runtime/critic until"),
    (re.compile(r"Production\s+path\s*\|\s*\*\*CURRENT\*\*.*critic", re.I), "Production path CURRENT critic"),
    (re.compile(r"Production\s+path\s+remains\s+CVL\s*/\s*Critic", re.I), "Production path remains CVL / Critic"),
    (re.compile(r"Council\s+runtime\s+NOT\s+STARTED", re.I), "Council runtime NOT STARTED"),
)

_STALE_CRITIC_EXCEPTIONS: dict[str, tuple[str, ...]] = {
    _PLAN_SYSTEM.name: (
        "Legacy Critic exists only in historical migration evidence",
        "Critic → Decision disposition matrix",
        "Decision/Critic parity table",
        "intergrax/runtime/migration/**",
    ),
    _ARCH_SYSTEM.name: (
        "CRITIC_VERIFICATION",
        "migration disposition",
        "historical migration",
    ),
    _ARCH_VERIFICATION.name: (
        "replacing the monolithic Critic model",
        "not `CriticOrchestrator` monolith",
        "CRITIC_VERIFICATION",
    ),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _phase_index_status(plan_text: str, phase_label: str) -> str | None:
    escaped = re.escape(phase_label).replace(r"\ ", r"\s+")
    pattern = rf"\|\s*\*?\*?{escaped}\*?\*?(?:\s*\([^)]*\))?\s*\|\s*([^|]+)\|"
    match = re.search(pattern, plan_text)
    if not match:
        return None
    return match.group(1).strip()


def test_decision_plan_parent_child_phase_consistency() -> None:
    system_plan = _read(_PLAN_SYSTEM)
    verification_plan = _read(_PLAN_VERIFICATION)
    deliberation_plan = _read(_PLAN_DELIBERATION)

    ver_parent = _phase_index_status(system_plan, "DS-VER-PIPE / DS-VER-STAGES")
    assert ver_parent is not None
    assert "PLANNED" not in ver_parent.upper(), (
        "parent DS-VER must not be PLANNED when verification plan phases are DONE"
    )
    assert all(
        "DONE" in line.upper()
        for line in re.findall(
            r"^\|\s*\*\*DS-VER-(?:PIPE|STAGES|PROD-COMP)\*\*\s*\|.*$",
            verification_plan,
            re.MULTILINE,
        )
    )

    delib_parent = _phase_index_status(system_plan, "DS-DELIB / DS-COUNCIL")
    assert delib_parent is not None
    assert not re.match(r"^\s*PLANNED\s*$", delib_parent, re.I), (
        "parent DS-DELIB cannot be PLANNED when deliberation foundation is DONE"
    )
    delib_child = _phase_index_status(deliberation_plan, "DS-DELIB")
    assert delib_child is not None
    assert "DONE" in delib_child.upper()

    mig_parent = _phase_index_status(system_plan, "DS-MIG")
    assert mig_parent is not None
    assert "PLANNED" not in mig_parent.upper(), (
        "parent DS-MIG must not be PLANNED after DS-MIG-05 ENTERPRISE CLOSED"
    )
    assert "COMPLETE" in mig_parent.upper()


def test_decision_docs_do_not_contain_stale_critic_current_claims() -> None:
    violations: list[str] = []
    for path in _CANONICAL_ACTIVE_DECISION_DOCS:
        text = _read(path)
        exceptions = _STALE_CRITIC_EXCEPTIONS.get(path.name, ())
        for pattern, label in _STALE_CRITIC_PATTERNS:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 120)
                end = min(len(text), match.end() + 120)
                context = text[start:end]
                if any(exc in context for exc in exceptions):
                    continue
                rel = path.relative_to(REPO_ROOT).as_posix()
                violations.append(f"{rel}: stale claim {label!r} near: {match.group(0)!r}")
    assert not violations, "stale Critic/current-runtime claims found:\n" + "\n".join(violations)


def test_decision_system_plan_declares_implemented_runtime_authority() -> None:
    plan = _read(_PLAN_SYSTEM)
    assert "Canonical Decision System runtime is" in plan and "implemented" in plan
    assert "Legacy Critic production authority has been" in plan and "fully retired" in plan
    assert "Path to complete" in plan


def test_decision_architecture_declares_decision_system_current_authority() -> None:
    arch = _read(_ARCH_SYSTEM)
    assert "CURRENT decision authority = Decision System" in arch
    assert "Critic runtime retired" in arch
    assert "Production qualification of full Decision System still pending DS-E2E" in arch


def test_decision_verification_architecture_declares_migrated_pipeline() -> None:
    arch = _read(_ARCH_VERIFICATION)
    assert "Verification Pipeline implementation migrated and active" in arch
    assert "Legacy Critic verification runtime retired" in arch
