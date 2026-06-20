# © Artur Czarnecki. All rights reserved.
"""Verify AGENTS stub split (F2), bootstrap F3/READ_BUDGET, and CURSOR_TOKEN_SETUP."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "docs" / "bootstrap"
CURSOR_SETUP = ROOT / "docs" / "guides" / "CURSOR_TOKEN_SETUP.md"
AGENT_INSTRUCTIONS = ROOT / "docs" / "guides" / "AGENT_INSTRUCTIONS.md"
SYMBOL_INDEX = ROOT / "docs" / "guides" / "SYMBOL_INDEX.md"
AGENTS_STUB = ROOT / "AGENTS.md"
ITERATION_RULE = ROOT / ".cursor" / "rules" / "intergrax-iteration.mdc"

SESSION_MARKER = "ONE_DOMAIN_ONE_CHAT"
READ_BUDGET_MARKER = "READ_BUDGET"
OUTPUT_BUDGET_MARKER = "OUTPUT_BUDGET"
O1_MARKER = "O1"
STUB_MARKER = "AGENT_INSTRUCTIONS.md"
STUB_MAX_LINES = 35
FULL_MIN_LINES = 150
AUDIT_BOOTSTRAPS = (
    "01_audit_all_domains.txt",
    "02_audit_one_domain.txt",
    "03_implement_plan_all_domains.txt",
    "04_implement_plan_one_domain.txt",
    "05_closeout_all_domains.txt",
    "06_interactive_layer_by_layer_audit.txt",
)


def main() -> int:
    errors: list[str] = []

    if not CURSOR_SETUP.is_file():
        errors.append("missing docs/guides/CURSOR_TOKEN_SETUP.md")
    elif "O1" not in CURSOR_SETUP.read_text(encoding="utf-8"):
        errors.append("CURSOR_TOKEN_SETUP.md must document O1 terse output policy")

    if not SYMBOL_INDEX.is_file():
        errors.append("missing docs/guides/SYMBOL_INDEX.md")

    if not AGENT_INSTRUCTIONS.is_file():
        errors.append("missing docs/guides/AGENT_INSTRUCTIONS.md")
    elif len(AGENT_INSTRUCTIONS.read_text(encoding="utf-8").splitlines()) < FULL_MIN_LINES:
        errors.append(f"AGENT_INSTRUCTIONS.md must be full reference (>={FULL_MIN_LINES} lines)")
    elif "Operator communication (O1" not in AGENT_INSTRUCTIONS.read_text(encoding="utf-8"):
        errors.append("AGENT_INSTRUCTIONS.md must include Operator communication (O1) section")

    if not AGENTS_STUB.is_file():
        errors.append("missing root AGENTS.md stub")
    else:
        stub = AGENTS_STUB.read_text(encoding="utf-8")
        stub_lines = len(stub.splitlines())
        if stub_lines > STUB_MAX_LINES:
            errors.append(f"AGENTS.md stub too large ({stub_lines} lines; max {STUB_MAX_LINES})")
        if STUB_MARKER not in stub:
            errors.append("AGENTS.md stub must link to AGENT_INSTRUCTIONS.md")
        if "O1" not in stub and "terse" not in stub.lower():
            errors.append("AGENTS.md stub must mention O1 terse output policy")
        if "## Task routing" in stub or "## Verification" in stub:
            errors.append("AGENTS.md stub must not contain full routing/verification sections")

    iteration = ITERATION_RULE.read_text(encoding="utf-8")
    if "AGENT_INSTRUCTIONS.md" not in iteration:
        errors.append("intergrax-iteration.mdc must reference AGENT_INSTRUCTIONS.md (F2 stub split)")
    if SESSION_MARKER not in iteration and "ONE DOMAIN = ONE NEW CHAT" not in iteration:
        errors.append("intergrax-iteration.mdc must include F3 session protocol")
    if "SYMBOL_INDEX" not in iteration:
        errors.append("intergrax-iteration.mdc must reference SYMBOL_INDEX.md (F5)")
    if O1_MARKER not in iteration or "terse default" not in iteration.lower():
        errors.append("intergrax-iteration.mdc must include O1 terse output policy")

    agents_ref = ROOT / ".cursor" / "rules" / "intergrax-agents-reference.mdc"
    if agents_ref.is_file():
        errors.append("remove redundant .cursor/rules/intergrax-agents-reference.mdc (stub replaces it)")

    for name in AUDIT_BOOTSTRAPS:
        path = BOOTSTRAP / name
        if not path.is_file():
            errors.append(f"missing bootstrap {name}")
            continue
        text = path.read_text(encoding="utf-8")
        if SESSION_MARKER not in text:
            errors.append(f"{name}: missing {SESSION_MARKER}")
        if READ_BUDGET_MARKER not in text:
            errors.append(f"{name}: missing {READ_BUDGET_MARKER}")
        if OUTPUT_BUDGET_MARKER not in text:
            errors.append(f"{name}: missing {OUTPUT_BUDGET_MARKER}")

    if errors:
        print("check_cursor_token_setup: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("check_cursor_token_setup: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
