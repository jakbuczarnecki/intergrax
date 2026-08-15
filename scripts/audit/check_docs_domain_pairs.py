# © Artur Czarnecki. All rights reserved.
"""Verify docs architecture/plan 1:1 domain and multi-layer feature pair structure."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "project"
ARCH = DOCS / "architecture"
PLAN = DOCS / "maintainers" / "plans"
FEATURES = DOCS / "capabilities"
FEATURE_ARCH = FEATURES / "architecture"
FEATURE_PLAN = FEATURES / "plan"
HUB = ARCH / "intergrax_runtime_architecture.md"
GUIDES = DOCS / "technical" / "guides"

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
    FEATURES / "satellites",
]

# Cross-domain implementation registers (hub-linked; no 1:1 architecture pair).
PLAN_ONLY_HUBS = {
    "IDEAL_HARNESS_L3",
    "AUDIT_IDEAL_2026",
    "HARNESS_EVIDENCE_PACK",
    "ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search",
    "PROVIDER_CATEGORY_CONTRACTS",
}

# Meta-architecture governance canon (hub-linked; no 1:1 plan pair).
ARCH_ONLY_GOVERNANCE = {
    "INTERGRAX_ARCHITECTURE_PRINCIPLES",
}

ALLOWED_DOCS_ROOT = {
    "README.md",
    "intergrax_runtime_architecture.md",
    "DOCUMENTATION_MAP.md",
}


def _check_domain_pairs(errors: list[str]) -> int:
    arch_files = sorted(ARCH.glob("*.md"))
    plan_files = sorted(PLAN.glob("*.md"))

    arch_names = {p.stem for p in arch_files} - ARCH_ONLY_GOVERNANCE
    plan_names = {p.stem for p in plan_files} - PLAN_ONLY_HUBS

    if arch_names != plan_names:
        only_arch = sorted(arch_names - plan_names)
        only_plan = sorted(plan_names - arch_names)
        if only_arch:
            errors.append(f"architecture without plan: {only_arch}")
        if only_plan:
            errors.append(f"plan without architecture: {only_plan}")

    for name in sorted(arch_names & plan_names):
        arch_text = (ARCH / f"{name}.md").read_text(encoding="utf-8")
        plan_text = (PLAN / f"{name}.md").read_text(encoding="utf-8")
        if f"maintainers/plans/{name}.md" not in arch_text:
            errors.append(f"architecture/{name}.md missing link to maintainers/plans/{name}.md")
        if f"architecture/{name}.md" not in plan_text:
            errors.append(f"plan/{name}.md missing link to architecture/{name}.md")
        if "Architecture (1:1)" not in plan_text and "architecture/" not in plan_text:
            errors.append(f"plan/{name}.md missing Architecture (1:1) header")

    hub = HUB.read_text(encoding="utf-8") if HUB.is_file() else ""
    for name in arch_names:
        if f"architecture/{name}.md" not in hub or f"maintainers/plans/{name}.md" not in hub:
            errors.append(f"hub missing pair entry for {name}")

    return len(arch_names)


def _check_feature_pairs(errors: list[str]) -> int:
    if not FEATURES.exists():
        return 0

    if not (FEATURES / "README.md").is_file():
        errors.append("docs/project/capabilities exists but docs/project/capabilities/README.md is missing")
    if not FEATURE_ARCH.is_dir():
        errors.append("docs/project/capabilities exists but docs/project/capabilities/architecture/ is missing")
    if not FEATURE_PLAN.is_dir():
        errors.append("docs/project/capabilities exists but docs/project/capabilities/plan/ is missing")

    if not FEATURE_ARCH.is_dir() or not FEATURE_PLAN.is_dir():
        return 0

    feature_arch_files = sorted(FEATURE_ARCH.glob("*.md"))
    feature_plan_files = sorted(FEATURE_PLAN.glob("*.md"))
    feature_arch_names = {p.stem for p in feature_arch_files}
    feature_plan_names = {p.stem for p in feature_plan_files}

    if feature_arch_names != feature_plan_names:
        only_arch = sorted(feature_arch_names - feature_plan_names)
        only_plan = sorted(feature_plan_names - feature_arch_names)
        if only_arch:
            errors.append(f"feature architecture without feature plan: {only_arch}")
        if only_plan:
            errors.append(f"feature plan without feature architecture: {only_plan}")

    for name in sorted(feature_arch_names & feature_plan_names):
        arch_text = (FEATURE_ARCH / f"{name}.md").read_text(encoding="utf-8")
        plan_text = (FEATURE_PLAN / f"{name}.md").read_text(encoding="utf-8")
        if f"../plan/{name}.md" not in arch_text and f"plan/{name}.md" not in arch_text:
            errors.append(
                f"capabilities/architecture/{name}.md missing link to capabilities/plan/{name}.md"
            )
        if f"../architecture/{name}.md" not in plan_text and f"architecture/{name}.md" not in plan_text:
            errors.append(
                f"capabilities/plan/{name}.md missing link to capabilities/architecture/{name}.md"
            )

    return len(feature_arch_names)


def main() -> int:
    errors: list[str] = []

    if not HUB.is_file():
        errors.append("missing docs/project/architecture/intergrax_runtime_architecture.md hub")

    extra_docs_root = [
        p.name for p in DOCS.glob("*.md") if p.name not in ALLOWED_DOCS_ROOT
    ]
    if extra_docs_root:
        errors.append(f"unexpected files in docs/project/ root: {extra_docs_root}")

    for p in FORBIDDEN:
        if p.exists():
            errors.append(f"forbidden path exists: {p.relative_to(ROOT)}")

    for g in REQUIRED_GUIDES:
        if not (GUIDES / g).is_file():
            errors.append(f"missing guides/{g}")

    domain_count = _check_domain_pairs(errors)
    feature_count = _check_feature_pairs(errors)

    if errors:
        print("check_docs_domain_pairs: FAILED")
        for e in errors:
            print(f"  - {e}")
        return 1

    suffix = f"; {feature_count} feature pairs" if feature_count else ""
    print(f"check_docs_domain_pairs: OK ({domain_count} domain pairs{suffix})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
