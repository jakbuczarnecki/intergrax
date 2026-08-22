# © Artur Czarnecki. All rights reserved.

"""Documentation contract tests for PP-2 Platform Proof System."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_PROOFS = REPO_ROOT / "platform_proofs"
MAP_PATH = PLATFORM_PROOFS / "PLATFORM_PROOF_MAP.md"
PROTOCOL_PATH = PLATFORM_PROOFS / "PLATFORM_PROOF_PROTOCOL.md"
README_PATH = PLATFORM_PROOFS / "README.md"
AUTHORING_PATH = PLATFORM_PROOFS / "PLATFORM_PROOF_AUTHORING_GUIDE.md"
MANIFEST_PATH = REPO_ROOT / "scripts" / "proof" / "intergrax_proof_manifest.py"
RUNNER_PATH = REPO_ROOT / "scripts" / "proof" / "intergrax_proof_runner.py"
PUBLIC_PROOFS = REPO_ROOT / "docs" / "project" / "proofs" / "PROOFS.md"

_DOCS_SCRIPTS = REPO_ROOT / "scripts" / "docs"
if str(_DOCS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_DOCS_SCRIPTS))
from docs_domain_common import canonical_domain_ids  # noqa: E402

_ADDITIONAL_DOMAINS = (
    "GOVERNED_EXECUTION",
    "AGENT_DISTRIBUTION",
    "PLATFORM_PLUGINS",
    "PROOF_RECEIPTS",
)
ALL_CANONICAL_DOMAINS: tuple[str, ...] = canonical_domain_ids() + _ADDITIONAL_DOMAINS

_MAP_DOMAIN_ROW = re.compile(
    r"^\| `(?P<domain>[A-Z][A-Z0-9_]*)` \|",
    re.MULTILINE,
)

@pytest.fixture
def map_text() -> str:
    return MAP_PATH.read_text(encoding="utf-8")


@pytest.fixture
def protocol_text() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


@pytest.fixture
def readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def test_platform_proofs_canonical_docs_exist() -> None:
    required = (
        README_PATH,
        PROTOCOL_PATH,
        MAP_PATH,
        AUTHORING_PATH,
    )
    missing = [path for path in required if not path.is_file()]
    assert not missing, f"missing platform_proofs canonical docs: {missing}"


def test_platform_proof_map_contains_all_28_domains_exactly_once(
    map_text: str,
) -> None:
    found = _MAP_DOMAIN_ROW.findall(map_text)
    domain_section = map_text.split("## Feature proof coverage", 1)[0]
    found_in_domain_table = _MAP_DOMAIN_ROW.findall(domain_section)
    assert len(found_in_domain_table) == 28
    assert len(set(found_in_domain_table)) == 28
    assert tuple(found_in_domain_table) == ALL_CANONICAL_DOMAINS


def test_scenario_design_root_documented(map_text: str, readme_text: str) -> None:
    assert "platform_proofs/scenarios/" in map_text or "scenarios/" in readme_text
    assert "ai_incident_investigation" in readme_text


def test_lkw_not_listed_as_platform_domain(map_text: str) -> None:
    domain_section = map_text.split("## Feature proof coverage", 1)[0]
    assert "`LKW`" not in domain_section
    assert "local_workspace_application" not in domain_section.lower() or "not" in domain_section.lower()
    lkw_domain_row = re.search(r"^\| `LKW` \|", map_text, re.MULTILINE)
    assert lkw_domain_row is None


def test_applications_product_only_ownership_in_protocol_and_readme(
    protocol_text: str,
    readme_text: str,
) -> None:
    normative = (
        "product-specific execution **MUST NOT** substitute for an independently owned Platform Proof"
    )
    assert normative in protocol_text
    assert normative in readme_text
    assert "applications/" in protocol_text
    assert "product proofs" in protocol_text.lower()
    assert "LKW" in protocol_text
    assert "applications/" in readme_text


def test_scripts_proof_remains_canonical_execution_infrastructure(
    protocol_text: str,
    readme_text: str,
) -> None:
    assert MANIFEST_PATH.is_file()
    assert RUNNER_PATH.is_file()
    assert "scripts/proof/intergrax_proof_manifest.py" in protocol_text
    assert "scripts/proof/intergrax_proof_runner.py" in protocol_text
    assert "SuiteReceipt" in protocol_text
    assert "scripts/proof/" in readme_text


def test_public_proofs_dashboard_ownership_preserved(protocol_text: str) -> None:
    assert PUBLIC_PROOFS.is_file()
    assert "docs/project/proofs/PROOFS.md" in protocol_text
    assert "PUBLIC_PROOF_AND_CLAIMS_MODEL" in protocol_text


def test_no_duplicate_proof_runner_or_manifest_paths() -> None:
    """PP-2 must not introduce competing manifest/runner under platform_proofs/."""
    platform_tree = PLATFORM_PROOFS
    forbidden_names = {
        "intergrax_proof_manifest.py",
        "intergrax_proof_runner.py",
        "intergrax_proof_contracts.py",
        "proof_manifest.py",
        "proof_runner.py",
    }
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in platform_tree.rglob("*.py")
        if path.name in forbidden_names
    ]
    assert not offenders, f"duplicate proof infrastructure under platform_proofs/: {offenders}"

    for name in ("intergrax_proof_manifest.py", "intergrax_proof_runner.py"):
        matches = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in REPO_ROOT.rglob(name)
            if path.relative_to(REPO_ROOT).as_posix() != f"scripts/proof/{name}"
        ]
        assert not matches, f"competing canonical {name} outside scripts/proof/: {matches}"
