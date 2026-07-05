# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-3G: public token optimization claim guardrail tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CLAIMS_DOC = _REPO_ROOT / "docs" / "public-adoption" / "TOKEN_OPTIMIZATION_CLAIMS.md"
_PUBLIC_ADOPTION_README = _REPO_ROOT / "docs" / "public-adoption" / "README.md"
_LKW_PLATFORM_PROOF = _REPO_ROOT / "docs" / "public-adoption" / "LKW_PLATFORM_PROOF.md"

_PERCENT_PATTERN = re.compile(r"\d+\s*%")
_FORBIDDEN_CONTEXT_MARKERS = (
    "do not say",
    "forbidden",
    "unless a future",
    "by x%",
    "x%",
)


def _read_claims_doc() -> str:
    return _CLAIMS_DOC.read_text(encoding="utf-8")


def _read_public_doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _line_is_forbidden_example_context(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in _FORBIDDEN_CONTEXT_MARKERS)


# --- Document existence and structure ---


def test_claim_guardrail_doc_exists() -> None:
    assert _CLAIMS_DOC.is_file()


def test_doc_contains_approved_public_wording_section() -> None:
    content = _read_claims_doc()
    assert "## Approved public wording" in content


def test_doc_contains_forbidden_wording_section() -> None:
    content = _read_claims_doc()
    assert "## Forbidden wording" in content


def test_doc_contains_evidence_checklist_before_publishing_numbers() -> None:
    content = _read_claims_doc()
    assert "## Evidence checklist before publishing numbers" in content


# --- Required explicit boundaries ---


def test_doc_explicitly_says_char_level_metrics() -> None:
    content = _read_claims_doc().lower()
    assert "char-level" in content or "character-level" in content


def test_doc_explicitly_says_synthetic_corpus() -> None:
    content = _read_claims_doc().lower()
    assert "synthetic" in content and "corpus" in content


def test_doc_explicitly_says_no_provider_aware_tokenizer() -> None:
    content = _read_claims_doc().lower()
    assert "no provider-aware tokenizer" in content or "provider-aware tokenizer" in content


def test_doc_explicitly_says_no_token_accurate_savings_claim() -> None:
    content = _read_claims_doc().lower()
    assert "no token-accurate savings claim" in content or "token-accurate savings" in content


# --- Forbidden phrase documentation ---


def test_doc_forbids_reduces_token_usage_by_x_percent() -> None:
    content = _read_claims_doc().lower()
    assert "reduces token usage by x%" in content


def test_doc_forbids_production_proven_token_savings() -> None:
    content = _read_claims_doc().lower()
    assert "production-proven token savings" in content


def test_doc_forbids_token_accurate_optimizer() -> None:
    content = _read_claims_doc().lower()
    assert "token-accurate optimizer" in content


def test_doc_does_not_contain_unqualified_numeric_percentage_claims() -> None:
    content = _read_claims_doc()
    offenders: list[str] = []
    for line in content.splitlines():
        if not _PERCENT_PATTERN.search(line):
            continue
        if _line_is_forbidden_example_context(line):
            continue
        offenders.append(line.strip())
    assert offenders == [], f"Unqualified percentage claims found: {offenders}"


# --- Cross-document links ---


def test_public_adoption_readme_links_to_claim_guardrails() -> None:
    content = _read_public_doc(_PUBLIC_ADOPTION_README)
    assert "TOKEN_OPTIMIZATION_CLAIMS.md" in content


def test_lkw_platform_proof_links_to_claim_guardrails() -> None:
    content = _read_public_doc(_LKW_PLATFORM_PROOF)
    assert "TOKEN_OPTIMIZATION_CLAIMS.md" in content


# --- Additional structure required by acceptance ---


def test_doc_contains_conditional_wording_section() -> None:
    content = _read_claims_doc()
    assert "## Conditional wording" in content


def test_doc_contains_reviewer_checklist_section() -> None:
    content = _read_claims_doc()
    assert "## Reviewer checklist" in content
