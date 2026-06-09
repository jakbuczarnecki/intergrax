# © Artur Czarnecki. All rights reserved.
"""Verify docs architecture/plan 1:1 domain pair structure."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ARCH = DOCS / "architecture"
PLAN = DOCS / "plan"
HUB = DOCS / "intergrax_runtime_architecture.md"
GUIDES = DOCS / "guides"

REQUIRED_GUIDES = {
    "INTERGRAX_DEVELOPMENT_STRATEGY.md",
    "IDEAL_HARNESS_AI_ARCHITECTURE.md",
    "INTEGRAX_HARNESS_AUDIT_MAP.md",
}

FORBIDDEN = [
    DOCS / "INTERGRAX_IMPLEMENTATION_PLAN.md",
    ARCH / "README.md",
    PLAN / "README.md",
    PLAN / "phases",
    PLAN / "appendices",
]

# Cross-domain implementation registers (hub-linked; no 1:1 architecture pair).
PLAN_ONLY_HUBS = {
    "IDEAL_HARNESS_L3",
    "AUDIT_IDEAL_2026",
}


def main() -> int:
    errors: list[str] = []

    if not HUB.is_file():
        errors.append("missing docs/intergrax_runtime_architecture.md hub")

    extra_docs_root = [p.name for p in DOCS.glob("*.md") if p.name != "intergrax_runtime_architecture.md"]
    if extra_docs_root:
        errors.append(f"unexpected files in docs/ root: {extra_docs_root}")

    for p in FORBIDDEN:
        if p.exists():
            errors.append(f"forbidden path exists: {p.relative_to(ROOT)}")

    for g in REQUIRED_GUIDES:
        if not (GUIDES / g).is_file():
            errors.append(f"missing guides/{g}")

    arch_files = sorted(ARCH.glob("*.md"))
    plan_files = sorted(PLAN.glob("*.md"))

    arch_names = {p.stem for p in arch_files}
    plan_names = {p.stem for p in plan_files} - PLAN_ONLY_HUBS

    if arch_names != plan_names:
        only_arch = sorted(arch_names - plan_names)
        only_plan = sorted(plan_names - arch_names)
        if only_arch:
            errors.append(f"architecture without plan: {only_arch}")
        if only_plan:
            errors.append(f"plan without architecture: {only_plan}")

    for name in sorted(arch_names):
        arch_text = (ARCH / f"{name}.md").read_text(encoding="utf-8")
        plan_text = (PLAN / f"{name}.md").read_text(encoding="utf-8")
        if f"plan/{name}.md" not in arch_text:
            errors.append(f"architecture/{name}.md missing link to plan/{name}.md")
        if f"architecture/{name}.md" not in plan_text:
            errors.append(f"plan/{name}.md missing link to architecture/{name}.md")
        if "Architecture (1:1)" not in plan_text and "architecture/" not in plan_text:
            errors.append(f"plan/{name}.md missing Architecture (1:1) header")

    hub = HUB.read_text(encoding="utf-8") if HUB.is_file() else ""
    for name in arch_names:
        if f"architecture/{name}.md" not in hub or f"plan/{name}.md" not in hub:
            errors.append(f"hub missing pair entry for {name}")

    if errors:
        print("check_docs_domain_pairs: FAILED")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"check_docs_domain_pairs: OK ({len(arch_names)} domain pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
