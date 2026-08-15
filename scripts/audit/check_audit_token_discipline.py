# © Artur Czarnecki. All rights reserved.
"""CI gate H1: domain audit prompts enforce token-discipline (no bulk doc load)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs" / "project" / "maintainers" / "audit"
ARCH = ROOT / "docs" / "project" / "architecture"
BOOTSTRAP = ROOT / "docs" / "project" / "maintainers" / "bootstrap"

DOMAIN_AUDITS = {p.stem for p in ARCH.glob("*.md")}

ORCHESTRATOR_EXCLUDE = {
    "README",
    "ORCHESTRATOR",
    "IDEA_AUDIT_ORCHESTRATOR",
    "IMPLEMENT_ORCHESTRATOR",
    "LAYER_COMPLETION_ORCHESTRATOR",
    "TEMPLATE_DOMAIN_RESULT",
}

AUDIT_BOOTSTRAPS = (
    "01_audit_all_domains.txt",
    "02_audit_one_domain.txt",
    "03_implement_plan_all_domains.txt",
    "04_implement_plan_one_domain.txt",
    "05_closeout_all_domains.txt",
    "06_interactive_layer_by_layer_audit.txt",
)

FORBIDDEN_IN_AUDIT = (
    re.compile(r"load full (plan|architecture)", re.I),
    re.compile(r"read entire (plan|architecture|IDEAL)", re.I),
    re.compile(r"load (IDEAL_HARNESS|INTEGRAX_HARNESS_AUDIT_MAP) in full", re.I),
)

REQUIRED_IN_DOMAIN_AUDIT = (
    "audit_slices",
    "Context budget",
)

MAX_PROMPT_BODY_TOKENS = 2_400
BEGIN = "---BEGIN PROMPT---"
END = "---END PROMPT---"


def extract_prompt_body(text: str) -> str | None:
    """Last BEGIN/END pair — ignore instructional mention in 'How to use'."""
    start = text.rfind(BEGIN)
    end = text.rfind(END)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start + len(BEGIN) : end]


def main() -> int:
    errors: list[str] = []

    for path in sorted(AUDIT.glob("*.md")):
        stem = path.stem
        if stem in ORCHESTRATOR_EXCLUDE:
            continue
        if stem not in DOMAIN_AUDITS:
            continue
        text = path.read_text(encoding="utf-8")
        for pat in FORBIDDEN_IN_AUDIT:
            if pat.search(text):
                errors.append(f"{path.relative_to(ROOT)}: forbidden bulk-load phrase: {pat.pattern}")
        for req in REQUIRED_IN_DOMAIN_AUDIT:
            if req not in text:
                errors.append(f"{path.relative_to(ROOT)}: missing required marker {req!r}")
        body = extract_prompt_body(text)
        if body is not None:
            tok = len(body) // 4
            if tok > MAX_PROMPT_BODY_TOKENS:
                errors.append(
                    f"{path.relative_to(ROOT)}: prompt body ~{tok} tok exceeds max {MAX_PROMPT_BODY_TOKENS}"
                )

    for name in AUDIT_BOOTSTRAPS:
        path = BOOTSTRAP / name
        if not path.is_file():
            errors.append(f"missing bootstrap {name}")
            continue
        text = path.read_text(encoding="utf-8")
        if "ONE_DOMAIN_ONE_CHAT" not in text:
            errors.append(f"{name}: missing ONE_DOMAIN_ONE_CHAT")
        if "READ_BUDGET" not in text:
            errors.append(f"{name}: missing READ_BUDGET")
        if "OUTPUT_BUDGET" not in text:
            errors.append(f"{name}: missing OUTPUT_BUDGET")

    if errors:
        print("check_audit_token_discipline: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("check_audit_token_discipline: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
