#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Audit Tier-3 hosts resolve catalogs through registry wiring (Phase REG-3)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

APPLICATIONS_ROOT = "applications"
SHARED_WIRING_ROOT = "intergrax/applications/_shared"

REQUIRED_WIRING_MARKERS = (
    "wire_application_environment",
    "build_harness_host_runtime",
)

FORBIDDEN_DIRECT_REGISTRY = re.compile(
    r"\b(?:ToolRegistry|SkillRegistry)\s*\(",
)

ALLOWLIST_PREFIXES = (
    f"{SHARED_WIRING_ROOT}/",
    "tests/",
)


def _grandfathered(rel: str) -> bool:
    return any(rel.startswith(prefix) for prefix in ALLOWLIST_PREFIXES)


def _check_host_wiring(repo_root: Path) -> list[str]:
    violations: list[str] = []
    apps_root = repo_root / APPLICATIONS_ROOT
    if not apps_root.is_dir():
        return [f"missing applications root: {APPLICATIONS_ROOT}"]

    for path in apps_root.glob("*_application/host/wiring.py"):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        if not any(marker in text for marker in REQUIRED_WIRING_MARKERS):
            violations.append(
                f"{rel}: must call wire_application_environment or build_harness_host_runtime"
            )
    return violations


def _check_direct_registry_construction(repo_root: Path) -> list[str]:
    violations: list[str] = []
    apps_root = repo_root / APPLICATIONS_ROOT
    if not apps_root.is_dir():
        return violations

    for path in apps_root.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(repo_root).as_posix()
        if _grandfathered(rel):
            continue
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN_DIRECT_REGISTRY.search(text):
            violations.append(f"{rel}: direct ToolRegistry/SkillRegistry construction is forbidden")
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    violations = _check_host_wiring(repo_root) + _check_direct_registry_construction(repo_root)
    if violations:
        print("harness registry resolution audit failed:")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1
    print("harness registry resolution audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
