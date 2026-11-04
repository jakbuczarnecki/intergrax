# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
ACP = ROOT / "governance" / "agent-control-plane"
GOV = ROOT / "scripts" / "governance"

# Normative caps from DCP-CLOSURE-BACKLOG.md agent execution protocol (keep in sync with registry).
_EXPECTED_SUBAGENTS: dict[str, dict[str, object]] = {
    "explore": {"maps_to": "architect-planner", "max_parallel": 3, "read_only": True},
    "generalPurpose": {"maps_to": "orchestrator", "max_parallel": 2, "read_only": False},
    "ci-investigator": {"maps_to": "tester", "max_parallel": 1, "read_only": True},
    "bugbot": {"maps_to": "reviewer", "max_parallel": 1, "read_only": True},
    "security-review": {"maps_to": "independent-quality-auditor", "max_parallel": 1, "read_only": True},
    "best-of-n-runner": {
        "maps_to": "implementer",
        "max_parallel": 2,
        "read_only": False,
        "requires_worktree": True,
    },
}


def _load_roles() -> dict:
    doc = yaml.safe_load((ACP / "role_registry.yaml").read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def test_subagent_role_map_matches_backlog_invariants() -> None:
    doc = _load_roles()
    sub = doc["subagent_role_map"]
    roles = doc["roles"]
    assert set(sub.keys()) == set(_EXPECTED_SUBAGENTS.keys())
    for name, expected in _EXPECTED_SUBAGENTS.items():
        meta = sub[name]
        for key, value in expected.items():
            assert meta.get(key) == value, f"{name}.{key}"
        assert meta["maps_to"] in roles


def test_subagent_maps_to_role_skills_exist() -> None:
    doc = _load_roles()
    skill_doc = yaml.safe_load((ACP / "skill_routing_registry.yaml").read_text(encoding="utf-8"))
    skill_ids = set(skill_doc.get("skills", {}).keys())
    for _name, meta in doc["subagent_role_map"].items():
        role = doc["roles"][meta["maps_to"]]
        for skill in role.get("allowed_skills", []):
            assert skill in skill_ids, f"missing skill route for {skill}"


def test_independent_auditor_cannot_declare_g6_pass() -> None:
    doc = _load_roles()
    auditor = doc["roles"]["independent-quality-auditor"]
    assert auditor.get("declares_g6_pass") is False


def test_verify_ticket_lifecycle_valid_ticket(tmp_path: Path) -> None:
    ticket = tmp_path / "ticket.md"
    ticket.write_text(
        "## Lifecycle state\n\nin_progress\n\nApproved: op @ 2026-01-01\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(GOV / "verify_ticket_lifecycle.py"), str(ticket)],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_verify_ticket_lifecycle_rejects_missing_approved(tmp_path: Path) -> None:
    ticket = tmp_path / "ticket.md"
    ticket.write_text("## Lifecycle state\n\nplan_approved\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(GOV / "verify_ticket_lifecycle.py"), str(ticket)],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 1
