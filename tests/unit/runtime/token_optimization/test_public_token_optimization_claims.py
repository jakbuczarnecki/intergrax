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
_TOKEN_OPT_ARCH = _REPO_ROOT / "docs" / "features" / "architecture" / "TOKEN_OPTIMIZATION.md"
_UCL_ARCH = _REPO_ROOT / "docs" / "architecture" / "UNIFIED_CONTEXT_LIFECYCLE.md"

_PERCENT_PATTERN = re.compile(r"\d+\s*%")
_FORBIDDEN_CONTEXT_MARKERS = (
    "do not say",
    "forbidden",
    "unless a future",
    "by x%",
    "x%",
)

# CTX-UCL-ARCH-1-R1: ownership regression guardrails (canonical TOKEN-10E / UCL docs).
_OWNERSHIP_FORBIDDEN_PHRASES = (
    "application-owned persistence and activation",
    "application owns where context versions are persisted",
    "application owns how an accepted context version becomes active",
    "rollback execution remains application-owned",
    "platform owner: intergrax.runtime.token_optimization",
)

# CTX-UCL-ARCH-1-R3: reuse-before-create and artifact lifecycle guardrails.
_REUSE_REQUIRED_CONCEPTS = (
    "reuse-before-create",
    "REUSE_ARTIFACT",
    "CREATE_ARTIFACT",
    "ArtifactLookupKey",
    "source_content_hash",
    "validation_contract_version",
)

_REGENERATION_FORBIDDEN_PHRASES = (
    "generate a summary before every model call",
    "always invoke the summarizer",
    "always invoke summarizer",
)

_ARTIFACT_OWNERSHIP_FORBIDDEN_PHRASES = (
    "token optimization owns artifact persistence",
    "application owns the summary cache",
)

_UCL_OWNERSHIP_FORBIDDEN_PHRASES = _OWNERSHIP_FORBIDDEN_PHRASES

_BYPASS_FLOW_FORBIDDEN_PATTERNS = (
    re.compile(
        r"application\s+context.*cacheawaretokenoptimizationruntime.*application-owned\s+activation",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"application-owned\s+persistence\s+and\s+activation",
        re.IGNORECASE,
    ),
)

_ALLOWED_APPLICATION_BOUNDARY_PHRASES = (
    "application-owned authorization",
    "application-owned review ux",
    "application-owned rollback ux",
    "application-selected persistence adapter",
    "application host authorizes",
    "persistence adapter selection and wiring",
)


def _read_claims_doc() -> str:
    return _CLAIMS_DOC.read_text(encoding="utf-8")


def _read_public_doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _token_optimization_section_810() -> str:
    content = _read_public_doc(_TOKEN_OPT_ARCH)
    start = content.index("### 8.10 Policy-governed in-cache compaction (TOKEN-10E)")
    end = content.index("## 9. Protected region policy")
    return content[start:end]


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


# --- CTX-UCL-ARCH-1-R1: UCL / TOKEN-10E ownership guardrails ---


@pytest.mark.parametrize("phrase", _OWNERSHIP_FORBIDDEN_PHRASES)
def test_token_optimization_section_810_rejects_old_ownership_phrases(
    phrase: str,
) -> None:
    section = _token_optimization_section_810().lower()
    assert phrase not in section, f"Forbidden ownership phrase in §8.10: {phrase!r}"


@pytest.mark.parametrize("phrase", _UCL_OWNERSHIP_FORBIDDEN_PHRASES)
def test_ucl_architecture_rejects_old_ownership_phrases(phrase: str) -> None:
    content = _read_public_doc(_UCL_ARCH).lower()
    assert phrase not in content, f"Forbidden ownership phrase in UCL arch: {phrase!r}"


def test_token_optimization_section_810_rejects_direct_bypass_activation_flow() -> None:
    section = _token_optimization_section_810()
    normalized = re.sub(r"\s+", " ", section.lower())
    for pattern in _BYPASS_FLOW_FORBIDDEN_PATTERNS:
        assert not pattern.search(normalized), (
            f"Forbidden bypass flow pattern: {pattern.pattern!r}"
        )


def test_ucl_architecture_documents_memory_session_activation_boundary() -> None:
    content = _read_public_doc(_UCL_ARCH).lower()
    assert "activecontextrevisionpointer" in content
    assert "memory/session cas" in content or "compare-and-swap" in content


def test_token_optimization_section_810_links_canonical_ucl() -> None:
    section = _token_optimization_section_810()
    assert "UNIFIED_CONTEXT_LIFECYCLE.md" in section
    assert "sole canonical source" in section.lower()


def test_token_optimization_section_810_has_valid_safe_reporting_marker() -> None:
    section = _token_optimization_section_810()
    assert "raw_content_included = false" in section
    assert not re.search(r"(?<!r)aw_content_included\s*=\s*false", section)
    assert "`\\text" not in section


def test_allowed_application_boundary_phrases_remain_valid_in_ucl() -> None:
    content = _read_public_doc(_UCL_ARCH).lower()
    assert any(phrase in content for phrase in _ALLOWED_APPLICATION_BOUNDARY_PHRASES)


# --- CTX-UCL-ARCH-1-R3: reuse-before-create guardrails ---


@pytest.mark.parametrize("concept", _REUSE_REQUIRED_CONCEPTS)
def test_ucl_architecture_documents_reuse_before_create_concepts(concept: str) -> None:
    content = _read_public_doc(_UCL_ARCH)
    assert concept in content, f"Missing required UCL concept: {concept!r}"


def test_ucl_architecture_documents_summary_regeneration_prohibition() -> None:
    content = _read_public_doc(_UCL_ARCH).lower()
    assert "must not" in content and "llm summarizer" in content
    assert "identical" in content and "artifactlookupkey" in content.replace("_", "")


@pytest.mark.parametrize("phrase", _REGENERATION_FORBIDDEN_PHRASES)
def test_ucl_architecture_rejects_regenerate_every_call_wording(phrase: str) -> None:
    content = _read_public_doc(_UCL_ARCH).lower()
    assert phrase not in content, f"Forbidden regeneration wording in UCL arch: {phrase!r}"


@pytest.mark.parametrize("phrase", _ARTIFACT_OWNERSHIP_FORBIDDEN_PHRASES)
def test_ucl_architecture_rejects_artifact_ownership_regressions(phrase: str) -> None:
    content = _read_public_doc(_UCL_ARCH).lower()
    assert phrase not in content, f"Forbidden artifact ownership phrase in UCL arch: {phrase!r}"


def test_ucl_architecture_documents_llm_transform_invariant_on_reuse() -> None:
    content = _read_public_doc(_UCL_ARCH).lower()
    assert "llm_transform_invoked" in content
    assert "reuse_artifact" in content
