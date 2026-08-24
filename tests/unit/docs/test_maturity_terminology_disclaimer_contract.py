# © Artur Czarnecki. All rights reserved.

"""Regression guard for public maturity/qualification terminology (PROMO-P1-1B)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
MATURITY_TAXONOMY = REPO_ROOT / "docs/project/technical/guides/MATURITY_TAXONOMY.md"
RAG = REPO_ROOT / "docs/project/architecture/RAG.md"
README = REPO_ROOT / "README.md"
PROOFS = REPO_ROOT / "docs/project/proofs/PROOFS.md"


def test_maturity_taxonomy_internal_classification_disclaimer() -> None:
    text = MATURITY_TAXONOMY.read_text(encoding="utf-8")
    assert "internal Intergrax engineering maturity classification" in text
    assert "not" in text and "third-party certification" in text


def test_rag_first_contact_internal_qualification_disclaimer() -> None:
    first_contact = RAG.read_text(encoding="utf-8").split("## At a glance", 1)[0]
    assert "internal engineering qualification status" in first_contact
    assert "third-party certification" in first_contact


def test_readme_no_hybrid_ask_certification_wording() -> None:
    assert "Hybrid Ask certification" not in README.read_text(encoding="utf-8")


def test_proofs_no_public_technical_certification_phrases() -> None:
    text = PROOFS.read_text(encoding="utf-8")
    assert "Hybrid Ask certification" not in text
    assert "all-provider certification" not in text
