# © Artur Czarnecki. All rights reserved.
"""Verify bootstrap files enforce ONE_DOMAIN_ONE_CHAT (F3) and CURSOR_TOKEN_SETUP exists (F2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "docs" / "bootstrap"
CURSOR_SETUP = ROOT / "docs" / "guides" / "CURSOR_TOKEN_SETUP.md"
ITERATION_RULE = ROOT / ".cursor" / "rules" / "intergrax-iteration.mdc"
AGENTS_REF_RULE = ROOT / ".cursor" / "rules" / "intergrax-agents-reference.mdc"

SESSION_MARKER = "ONE_DOMAIN_ONE_CHAT"
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

    if not AGENTS_REF_RULE.is_file():
        errors.append("missing .cursor/rules/intergrax-agents-reference.mdc")

    iteration = ITERATION_RULE.read_text(encoding="utf-8")
    if "AGENTS.md is NOT auto-loaded" not in iteration and "NOT auto-loaded" not in iteration:
        errors.append("intergrax-iteration.mdc must state AGENTS.md is not auto-loaded (F2)")
    if SESSION_MARKER not in iteration and "ONE DOMAIN = ONE NEW CHAT" not in iteration:
        errors.append("intergrax-iteration.mdc must include F3 session protocol")

    agents_ref = AGENTS_REF_RULE.read_text(encoding="utf-8")
    if "alwaysApply: false" not in Path(AGENTS_REF_RULE).read_text(encoding="utf-8"):
        pass  # frontmatter checked below
    if agents_ref.startswith("---"):
        if "alwaysApply: false" not in agents_ref.split("---", 2)[1]:
            errors.append("intergrax-agents-reference.mdc must have alwaysApply: false")

    for name in AUDIT_BOOTSTRAPS:
        path = BOOTSTRAP / name
        if not path.is_file():
            errors.append(f"missing bootstrap {name}")
            continue
        text = path.read_text(encoding="utf-8")
        if SESSION_MARKER not in text:
            errors.append(f"{name}: missing {SESSION_MARKER}")

    if errors:
        print("check_cursor_token_setup: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("check_cursor_token_setup: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
