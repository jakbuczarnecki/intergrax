# © Artur Czarnecki. All rights reserved.
"""Verify docs architecture/plan 1:1 domain and multi-layer feature pair structure."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docs_domain_common import DOMAIN_ORDER, REPO_ROOT, canonical_domain_ids

DOCS = REPO_ROOT / "docs" / "project"
ARCH = DOCS / "architecture"
PLAN = DOCS / "maintainers" / "plans"
FEATURES = DOCS / "capabilities"
FEATURE_ARCH = FEATURES / "architecture"
FEATURE_PLAN = FEATURES / "plan"
CAPABILITIES_README = FEATURES / "README.md"
HUB = ARCH / "intergrax_runtime_architecture.md"
GUIDES = DOCS / "technical" / "guides"

REQUIRED_GUIDES = {
    "INTERGRAX_DEVELOPMENT_STRATEGY.md",
    "IDEAL_HARNESS_AI_ARCHITECTURE.md",
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

_FEATURE_INDEX_ROW = re.compile(r"^\| `([A-Z][A-Z0-9_]+)` \|", re.MULTILINE)


def _canonical_feature_ids() -> list[str]:
    text = CAPABILITIES_README.read_text(encoding="utf-8")
    start = text.find("## Current multi-layer features")
    end = text.find("**Satellites", start)
    if start < 0 or end < 0:
        return []
    return _FEATURE_INDEX_ROW.findall(text[start:end])


def _has_path_reference(text: str, *candidates: str) -> bool:
    return any(candidate in text for candidate in candidates)


def _check_domain_pairs(errors: list[str]) -> int:
    canonical = list(canonical_domain_ids())
    canonical_set = set(canonical)

    if tuple(canonical) != DOMAIN_ORDER:
        errors.append(
            "DOMAIN_ORDER != hub domain ids; "
            f"hub={canonical}; common={list(DOMAIN_ORDER)}",
        )

    overlap_plan_only = sorted(canonical_set & PLAN_ONLY_HUBS)
    overlap_arch_only = sorted(canonical_set & ARCH_ONLY_GOVERNANCE)
    if overlap_plan_only:
        errors.append(f"canonical domains overlap plan-only hubs: {overlap_plan_only}")
    if overlap_arch_only:
        errors.append(f"canonical domains overlap arch-only governance: {overlap_arch_only}")

    for name in PLAN_ONLY_HUBS:
        if not (PLAN / f"{name}.md").is_file():
            errors.append(f"plan-only hub missing maintainers/plans/{name}.md")
    for name in ARCH_ONLY_GOVERNANCE:
        if not (ARCH / f"{name}.md").is_file():
            errors.append(f"arch-only governance missing architecture/{name}.md")

    for name in canonical:
        arch_path = ARCH / f"{name}.md"
        plan_path = PLAN / f"{name}.md"
        if not arch_path.is_file():
            errors.append(f"canonical domain missing architecture/{name}.md")
            continue
        if not plan_path.is_file():
            errors.append(f"canonical domain missing maintainers/plans/{name}.md")
            continue

        arch_text = arch_path.read_text(encoding="utf-8")
        plan_text = plan_path.read_text(encoding="utf-8")
        if not _has_path_reference(
            arch_text,
            f"maintainers/plans/{name}.md",
            f"../maintainers/plans/{name}.md",
        ):
            errors.append(f"architecture/{name}.md missing link to maintainers/plans/{name}.md")
        if not _has_path_reference(
            plan_text,
            f"architecture/{name}.md",
            f"../architecture/{name}.md",
        ):
            errors.append(f"plan/{name}.md missing link to architecture/{name}.md")
        if "Architecture (1:1)" not in plan_text and "architecture/" not in plan_text:
            errors.append(f"plan/{name}.md missing Architecture (1:1) header")

    return len(canonical)


def _check_feature_pairs(errors: list[str]) -> int:
    if not FEATURES.exists():
        return 0

    if not CAPABILITIES_README.is_file():
        errors.append("docs/project/capabilities exists but docs/project/capabilities/README.md is missing")
    if not FEATURE_ARCH.is_dir():
        errors.append("docs/project/capabilities exists but docs/project/capabilities/architecture/ is missing")
    if not FEATURE_PLAN.is_dir():
        errors.append("docs/project/capabilities exists but docs/project/capabilities/plan/ is missing")

    feature_ids = _canonical_feature_ids()
    if FEATURES.exists() and CAPABILITIES_README.is_file() and not feature_ids:
        errors.append("capabilities/README.md missing canonical feature index rows")

    if not FEATURE_ARCH.is_dir() or not FEATURE_PLAN.is_dir():
        return 0

    feature_set = set(feature_ids)
    for name in feature_ids:
        arch_path = FEATURE_ARCH / f"{name}.md"
        plan_path = FEATURE_PLAN / f"{name}.md"
        if not arch_path.is_file():
            errors.append(f"canonical feature missing capabilities/architecture/{name}.md")
            continue
        if not plan_path.is_file():
            errors.append(f"canonical feature missing capabilities/plan/{name}.md")
            continue

        arch_text = arch_path.read_text(encoding="utf-8")
        plan_text = plan_path.read_text(encoding="utf-8")
        if not _has_path_reference(
            arch_text,
            f"../plan/{name}.md",
            f"plan/{name}.md",
            f"capabilities/plan/{name}.md",
        ):
            errors.append(
                f"capabilities/architecture/{name}.md missing link to capabilities/plan/{name}.md",
            )
        if not _has_path_reference(
            plan_text,
            f"../architecture/{name}.md",
            f"architecture/{name}.md",
            f"capabilities/architecture/{name}.md",
        ):
            errors.append(
                f"capabilities/plan/{name}.md missing link to capabilities/architecture/{name}.md",
            )

    return len(feature_set)


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
            errors.append(f"forbidden path exists: {p.relative_to(REPO_ROOT)}")

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
