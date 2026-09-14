# © Artur Czarnecki. All rights reserved.

"""HARDENING-9.1 — canonical ``dev-unit-cert`` optional-extra contract (import graph for ``tests/unit``)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

CANONICAL_EXTRA = "dev-unit-cert"

# Packages required to clear the 17 known collection failures (see UNIT_TEST_CERTIFICATION_ENVIRONMENT.md).
_REQUIRED_DISTRIBUTIONS = frozenset(
    {
        "fastmcp",
        "langchain-core",
        "qdrant-client",
        "anthropic",
        "tiktoken",
        "pgvector",
        "torch",
        "sentence-transformers",
        "transformers",
    }
)

# Heavy / external-only surfaces that must not be pulled into the unit-cert profile.
_FORBIDDEN_DISTRIBUTIONS = frozenset(
    {
        "docling",
        "unstructured",
        "neo4j",
        "e2b",
        "chromadb",
        "pinecone",
        "openai-whisper",
    }
)

_CI_SYNC_PATTERN = re.compile(
    r"uv sync .*--extra dev-ci .*--extra dev-unit-cert",
    re.MULTILINE,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _requirement_distribution_name(requirement: str) -> str:
    token = requirement.strip()
    for separator in ("==", ">=", "<=", "!=", "~=", ">", "<", "[", ";"):
        if separator in token:
            token = token.split(separator, 1)[0].strip()
    return token.casefold()


def _optional_extra_packages(extra_name: str) -> frozenset[str]:
    pyproject = _repo_root() / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    optional = data.get("project", {}).get("optional-dependencies", {})
    if extra_name not in optional:
        return frozenset()
    return frozenset(
        _requirement_distribution_name(requirement)
        for requirement in optional[extra_name]
        if requirement.strip()
    )


def test_canonical_unit_cert_extra_is_declared() -> None:
    packages = _optional_extra_packages(CANONICAL_EXTRA)
    assert packages, f"missing [project.optional-dependencies].{CANONICAL_EXTRA}"


def test_unit_cert_extra_covers_required_import_graph() -> None:
    packages = _optional_extra_packages(CANONICAL_EXTRA)
    missing = sorted(
        name for name in _REQUIRED_DISTRIBUTIONS if name.casefold() not in packages
    )
    assert not missing, f"{CANONICAL_EXTRA} missing distributions: {missing}"


def test_unit_cert_extra_excludes_forbidden_heavy_integrations() -> None:
    packages = _optional_extra_packages(CANONICAL_EXTRA)
    present = sorted(name for name in _FORBIDDEN_DISTRIBUTIONS if name.casefold() in packages)
    assert not present, f"{CANONICAL_EXTRA} must not include: {present}"


def test_nightly_regression_gate_syncs_unit_cert_profile() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "unit-tests.yml").read_text(
        encoding="utf-8"
    )
    assert _CI_SYNC_PATTERN.search(workflow), (
        "unit-tests.yml gate job must run: uv sync --extra dev-ci --extra dev-unit-cert"
    )


def test_unit_cert_documentation_matches_extra_name() -> None:
    doc_path = (
        _repo_root()
        / "docs"
        / "project"
        / "maintainers"
        / "quality"
        / "UNIT_TEST_CERTIFICATION_ENVIRONMENT.md"
    )
    text = doc_path.read_text(encoding="utf-8")
    assert CANONICAL_EXTRA in text
    assert "uv sync --all-extras" in text
    assert "anti-pattern" in text.casefold() or "Anti-pattern" in text
