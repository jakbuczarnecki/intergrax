# © Artur Czarnecki. All rights reserved.

"""Platform documentation must not depend on application or agent paths.

Higher layers may reference platform contracts. Platform provider documentation
must not cite concrete repository paths under applications/ or agents/.

Generic wording such as "consuming application" or "agent integration" is allowed.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_CANONICAL_PLATFORM_DOC_ROOTS = (
    REPO / "docs" / "project",
    REPO / "docs" / "project" / "architecture",
    REPO / "docs" / "project" / "technical",
    REPO / "docs" / "project" / "integrations",
)

# Concrete downward path dependencies (not generic terminology).
_FORBIDDEN_DOWNWARD_PATHS = (
    re.compile(r"\]\([^)]*applications/"),
    re.compile(r"\]\([^)]*agents/"),
    re.compile(r"`applications/[^`]+`"),
    re.compile(r"`agents/[^`]+`"),
    # Exclude platform packages intergrax/applications/ and intergrax/agents/.
    re.compile(r"(?<!intergrax/)(?<![A-Za-z0-9_])applications/[A-Za-z0-9_.\-/]+"),
    re.compile(r"(?<!intergrax/)(?<![A-Za-z0-9_])agents/[A-Za-z0-9_.\-/]+"),
)


def _provider_usage_files() -> list[Path]:
    root = REPO / "intergrax" / "integrations" / "providers"
    if not root.is_dir():
        return []
    return sorted(root.rglob("USAGE.md"))


def _iter_violations(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    found: list[tuple[str, str]] = []
    for pattern in _FORBIDDEN_DOWNWARD_PATHS:
        for match in pattern.finditer(text):
            found.append((pattern.pattern, match.group(0)))
    return found


def test_provider_usage_docs_have_no_application_or_agent_paths() -> None:
    """Provider-local USAGE.md must not link or cite applications/ or agents/."""
    files = _provider_usage_files()
    assert files, "expected provider USAGE.md files under intergrax/integrations/providers"
    violations: list[str] = []
    for path in files:
        for pattern, matched in _iter_violations(path):
            rel = path.relative_to(REPO).as_posix()
            violations.append(
                f"source={rel} pattern={pattern!r} matched={matched!r}"
            )
    assert not violations, "platform provider docs cite applications/ or agents/:\n" + "\n".join(
        violations
    )


def test_platform_doc_roots_exist_for_boundary_scope() -> None:
    """Enforce canonical documentation roots and reject the pre-migration roots."""
    missing = [path for path in _CANONICAL_PLATFORM_DOC_ROOTS if not path.is_dir()]
    assert not missing, f"missing canonical platform documentation roots: {missing}"

    technical_map = (
        REPO / "docs" / "project" / "technical" / "DOCUMENTATION_MAP.md"
    ).read_text(encoding="utf-8")
    public_architecture = (
        REPO
        / "docs"
        / "project"
        / "maintainers"
        / "public-adoption"
        / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
    ).read_text(encoding="utf-8")
    stale_topology_language = (
        "Sole `docs/` root file",
        "root `PROOFS.md`",
        "root proof dashboard",
        "applications/*/docs/",
    )
    for phrase in stale_topology_language:
        assert phrase not in technical_map
        assert phrase not in public_architecture

    assert "docs/project/architecture/" in technical_map
    assert "docs/project/proofs/PROOFS.md" in public_architecture
    assert "docs/project/technical/applications/" in public_architecture
    assert "docs/project/technical/agents/" in public_architecture

    assert _provider_usage_files()
