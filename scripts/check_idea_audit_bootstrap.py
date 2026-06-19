# © Artur Czarnecki. All rights reserved.
"""Verify Mode I idea-audit bootstrap and orchestrator consistency."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "docs" / "bootstrap" / "07_idea_audit.txt"
ORCHESTRATOR = ROOT / "docs" / "audit" / "IDEA_AUDIT_ORCHESTRATOR.md"
HUB = ROOT / "docs" / "intergrax_runtime_architecture.md"
AUDIT_MAP = ROOT / "docs" / "guides" / "INTEGRAX_HARNESS_AUDIT_MAP.md"
BOOTSTRAP_README = ROOT / "docs" / "bootstrap" / "README.md"

REQUIRED_BOOTSTRAP_MARKERS = (
    "IDEA_LABEL=",
    "IDEA_TYPE=",
    "MODE=",
    "IDEA_DESCRIPTION:",
    "IDEA_AUDIT_ORCHESTRATOR.md",
    "Do not** write idea-audit artifact files",
    "Duplicate scan (live)",
    "already_implemented",
    "partial_overlap",
    "needs_clarification",
    "deferred_product",
)

REQUIRED_ORCHESTRATOR_MARKERS = (
    "07_idea_audit.txt",
    "Step 0",
    "Step 7",
    "audit-only",
    "audit-and-apply-docs",
    "No idea-audit artifact files",
)

REQUIRED_WORK_CYCLE_STEPS = tuple(f"### Step {n}" for n in range(8))


def main() -> int:
    errors: list[str] = []

    for path in (BOOTSTRAP, ORCHESTRATOR, HUB, AUDIT_MAP, BOOTSTRAP_README):
        if not path.is_file():
            errors.append(f"missing file: {path.relative_to(ROOT)}")

    if errors:
        _report(errors)
        return 1

    bootstrap_text = BOOTSTRAP.read_text(encoding="utf-8")
    orchestrator_text = ORCHESTRATOR.read_text(encoding="utf-8")
    hub_text = HUB.read_text(encoding="utf-8")
    audit_map_text = AUDIT_MAP.read_text(encoding="utf-8")
    bootstrap_readme_text = BOOTSTRAP_README.read_text(encoding="utf-8")

    for marker in REQUIRED_BOOTSTRAP_MARKERS:
        if marker not in bootstrap_text:
            errors.append(f"bootstrap missing marker: {marker!r}")

    for marker in REQUIRED_ORCHESTRATOR_MARKERS:
        if marker not in orchestrator_text:
            errors.append(f"orchestrator missing marker: {marker!r}")

    for step in REQUIRED_WORK_CYCLE_STEPS:
        if step not in orchestrator_text:
            errors.append(f"orchestrator missing work-cycle heading: {step}")

    if "docs/audit/IDEA_AUDIT_ORCHESTRATOR.md" not in bootstrap_text:
        errors.append("bootstrap must reference docs/audit/IDEA_AUDIT_ORCHESTRATOR.md")

    if "../bootstrap/07_idea_audit.txt" not in orchestrator_text:
        errors.append("orchestrator must link to bootstrap/07_idea_audit.txt")

    if "07_idea_audit.txt" not in hub_text:
        errors.append("hub must reference bootstrap/07_idea_audit.txt")

    if "Mode I" not in audit_map_text or "07_idea_audit.txt" not in audit_map_text:
        errors.append("audit map must index Mode I and 07_idea_audit.txt")

    if "No" not in bootstrap_readme_text or "init_architecture_audit_run" not in bootstrap_readme_text:
        errors.append("bootstrap README must clarify init script scope for Mode I")

    _report(errors)
    return 1 if errors else 0


def _report(errors: list[str]) -> None:
    if errors:
        print("check_idea_audit_bootstrap: FAILED", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return
    print("check_idea_audit_bootstrap: OK")


if __name__ == "__main__":
    raise SystemExit(main())
