#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — TIER3_APPLICATION_ENVIRONMENT audit prompt freshness (APP-CON-DX.2)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "docs" / "project" / "technical" / "guides" / "audit" / "TIER3_APPLICATION_ENVIRONMENT.md"

REQUIRED_MARKERS = (
    "APP-EVOL-7",
    "APP-OPS-4",
    "EnvironmentHealthScore",
    "ApplicationRegistry",
    "check_application_production_gates.py",
    "APPLICATION_CREATION_GUIDE.md",
    "§24–§51",
)


def main() -> int:
    violations: list[str] = []

    if not AUDIT_PATH.is_file():
        violations.append(f"missing audit prompt: {AUDIT_PATH}")
        return _report(violations)

    actual = AUDIT_PATH.read_text(encoding="utf-8")
    for marker in REQUIRED_MARKERS:
        if marker not in actual:
            violations.append(f"audit prompt missing marker: {marker}")

    gen = runpy.run_path(str(REPO_ROOT / "scripts" / "generate_domain_audit_prompts.py"))
    domains: list[dict] = gen["DOMAINS"]
    render = gen["render"]
    tier3 = next(item for item in domains if item["id"] == "TIER3_APPLICATION_ENVIRONMENT")
    expected = render(tier3)
    if actual != expected:
        violations.append(
            "audit prompt out of date — run: uv run python scripts/audit/generate_domain_audit_prompts.py",
        )

    return _report(violations)


def _report(violations: list[str]) -> int:
    if violations:
        print("tier3 audit prompt gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1
    print("tier3 audit prompt gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
