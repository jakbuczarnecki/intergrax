# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]

_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("Policy manifest", "governance/GOVERNANCE-MANIFEST.yaml"),
    ("Policy hierarchy", "governance/POLICY-HIERARCHY.md"),
    ("Policy registry", "governance/policy-registry.yaml"),
    ("Tickets SSOT", "governance/tickets/"),
    ("Commit Ref gate", "scripts/governance/verify_commit_ticket_refs.py"),
    ("Quality CLI", "scripts/governance/igx_gov/"),
    ("Role registry", "governance/agent-control-plane/role_registry.yaml"),
    ("Skill routing", "governance/agent-control-plane/skill_routing_registry.yaml"),
    ("Task lifecycle", "governance/agent-control-plane/task_lifecycle.yaml"),
    ("Plan approvers", "governance/agent-control-plane/plan_approvers.yaml"),
    ("Cursor skills", ".cursor/skills/igx-devos-*/"),
    ("Agent rules", ".cursor/rules/intergrax-agent-roles.mdc"),
    ("Independent audit rule", ".cursor/rules/intergrax-independent-audit.mdc"),
    ("Worktrees workflow", "governance/workflows/WORKTREES.md"),
    ("DevOS docs hub", "docs/project/technical/devos/README.md"),
    ("Push gate", "scripts/ci/run_push_gate.py"),
    ("Pre-commit gate", "scripts/ci/run_pre_commit_gate.py"),
)


def print_inventory() -> None:
    print("InterGRAX Delivery DevOS inventory (Part II subset):\n")
    for label, rel in _COMPONENTS:
        path = _REPO / rel.rstrip("/")
        ok = path.exists()
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {label}: {rel}")
