# © Artur Czarnecki. All rights reserved.
"""CI gate H1: domain audit prompts enforce token-discipline (no bulk doc load)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs" / "audit"
ARCH = ROOT / "docs" / "architecture"
BOOTSTRAP = ROOT / "docs" / "bootstrap"

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
