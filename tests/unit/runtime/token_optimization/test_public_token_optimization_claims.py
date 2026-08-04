# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-3G: public token optimization claim guardrail tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CLAIMS_DOC = _REPO_ROOT / "docs" / "public-adoption" / "TOKEN_OPTIMIZATION_CLAIMS.md"
_PROOFS_DOC = _REPO_ROOT / "PROOFS.md"
_PUBLIC_ADOPTION_README = _REPO_ROOT / "docs" / "public-adoption" / "README.md"
_PUBLIC_PROOF_MODEL = (
    _REPO_ROOT
    / "docs"
    / "public-adoption"
    / "PUBLIC_PROOF_AND_CLAIMS_MODEL.md"
)
_LKW_PLATFORM_PROOF = _REPO_ROOT / "docs" / "public-adoption" / "LKW_PLATFORM_PROOF.md"
_TOKEN_OPT_README = _REPO_ROOT / "docs" / "features" / "token_optimization" / "README.md"
_TOKEN_OPT_ARCH = _REPO_ROOT / "docs" / "features" / "architecture" / "TOKEN_OPTIMIZATION.md"
_UCL_ARCH = _REPO_ROOT / "docs" / "architecture" / "UNIFIED_CONTEXT_LIFECYCLE.md"
_UCL_PLAN = _REPO_ROOT / "docs" / "plan" / "UNIFIED_CONTEXT_LIFECYCLE.md"
_UCL_ADR = _REPO_ROOT / "docs" / "adr" / "entries" / "2026-08-01" / "ADR-UCL-001.md"

_CANONICAL_STATUS_LABELS = (
    "IMPLEMENTED",
    "BOUNDED PROOF",
    "PARTIAL",
    "PLANNED",
    "NOT CLAIMABLE",
)

_PERCENT_PATTERN = re.compile(r"\d+\s*%")
_FORBIDDEN_CONTEXT_MARKERS = (
    "do not say",
    "do not claim",
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

# CTX-UCL-ARCH-1-R4: internal-call boundary, single-flight creation, repository delivery guardrails.
_R4_REQUIRED_CONCEPTS = (
    "PRIMARY_MODEL_CALL",
    "INTERNAL_OPTIMIZATION_CALL",
    "OptimizationExecutionGuard",
    "ArtifactCreationReservation",
    "single-flight",
    "InMemoryOptimizationArtifactRepository",
    "OPTIMIZATION_RECURSION_BLOCKED",
    "ARTIFACT_CREATION_IN_PROGRESS",
)

_R4_FORBIDDEN_PHRASES = (
    "internal summarizer traverses the full ucl lifecycle",
    "allow duplicate summarization and deduplicate afterward",
    "content addressing alone prevents duplicate llm calls",
    "token optimization owns the artifact repository",
    "application-local mutex coordinates summary creation",
)

_ARTIFACT_OWNERSHIP_FORBIDDEN_PHRASES = (
    "token optimization owns artifact persistence",
    "token optimization owns the artifact repository",
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


def _normalize_public_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\*+", "", normalized)
    normalized = re.sub(r"[_`]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


_UCL_PUBLIC_SURFACES = (
    ("PROOFS.md", _PROOFS_DOC),
    ("token_optimization/README.md", _TOKEN_OPT_README),
    ("PUBLIC_PROOF_AND_CLAIMS_MODEL.md", _PUBLIC_PROOF_MODEL),
    ("TOKEN_OPTIMIZATION_CLAIMS.md", _CLAIMS_DOC),
)

_CTX_UCL_6_MILESTONE = re.compile(r"ctx-ucl-6(?!a)", re.IGNORECASE)


def _ucl_status_window(normalized: str, milestone: str) -> str:
    milestone_lower = milestone.lower()
    if milestone == "CTX-UCL-6":
        match = _CTX_UCL_6_MILESTONE.search(normalized)
        if match is None:
            return ""
        return normalized[match.start() : match.start() + 200]
    idx = normalized.find(milestone_lower)
    if idx == -1:
        return ""
    return normalized[idx : idx + 200]


def _ucl_status_segment(normalized: str, milestone: str, *, width: int = 80) -> str:
    milestone_lower = milestone.lower()
    if milestone == "CTX-UCL-6":
        match = _CTX_UCL_6_MILESTONE.search(normalized)
        if match is None:
            return ""
        return normalized[match.start() : match.start() + width]
    idx = normalized.find(milestone_lower)
    if idx == -1:
        return ""
    return normalized[idx : idx + width]


def _assert_required_ucl_statuses(normalized: str) -> None:
    ucl5 = _ucl_status_window(normalized, "CTX-UCL-5")
    assert "accepted" in ucl5 and "closed" in ucl5, "CTX-UCL-5 must be accepted/closed"

    ucl6 = _ucl_status_window(normalized, "CTX-UCL-6")
    assert "accepted" in ucl6 and "closed" in ucl6, "CTX-UCL-6 must be accepted/closed"

    closeout = _ucl_status_window(normalized, "CTX-UCL-CLOSEOUT-1")
    assert (
        ("accepted" in closeout and "closed" in closeout)
        or "ready for final review" in closeout
        or "pending independent acceptance" in closeout
    ), "CTX-UCL-CLOSEOUT-1 must be accepted/closed or ready for review"

    token10e = _ucl_status_window(normalized, "TOKEN-10E-1")
    assert token10e, "TOKEN-10E-1 status must be present"
    assert (
        "ready for review" in token10e
        or "ready_for_review" in token10e
        or "blocked" in token10e
    ), "TOKEN-10E-1 must be ready for review or blocked"


_CTX_UCL_6_NOT_STARTED = re.compile(
    r"ctx-ucl-6(?!a)\b[^;|]{0,40}\bnot started\b",
    re.IGNORECASE,
)


def _assert_no_stale_ucl_wording(normalized: str) -> None:
    ucl5_segment = _ucl_status_segment(normalized, "CTX-UCL-5")
    assert not (
        ("ready for review" in ucl5_segment or "in review" in ucl5_segment)
        and "accepted" not in ucl5_segment
    ), "CTX-UCL-5 must not be described as ready for review or in review"

    stale_patterns = (
        _CTX_UCL_6_NOT_STARTED,
        re.compile(r"ctx-ucl-6(?!a).{0,60}\bin progress\b", re.IGNORECASE),
        re.compile(r"ctx-ucl-6a.{0,60}\bready for review\b", re.IGNORECASE),
        re.compile(r"ctx-ucl-6b.{0,60}\bready for review\b", re.IGNORECASE),
        re.compile(r"ctx-ucl-closeout-1.{0,60}\bnot started\b", re.IGNORECASE),
        re.compile(r"token-10e.{0,60}\bimplementation started\b", re.IGNORECASE),
        re.compile(
            r"token-10e(?:\s+is|\s+runtime|\s+implementation)?\s+implemented\b",
            re.IGNORECASE,
        ),
    )
    for pattern in stale_patterns:
        assert not pattern.search(normalized), (
            f"Stale UCL wording matched pattern: {pattern.pattern!r}"
        )


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


def test_proofs_doc_exists() -> None:
    assert _PROOFS_DOC.is_file()


@pytest.mark.parametrize("label", _CANONICAL_STATUS_LABELS)
def test_proofs_doc_contains_all_canonical_status_labels(label: str) -> None:
    content = _read_public_doc(_PROOFS_DOC)
    assert label in content


def test_proofs_doc_links_to_lkw_platform_proof() -> None:
    content = _read_public_doc(_PROOFS_DOC)
    assert "LKW_PLATFORM_PROOF.md" in content


def test_proofs_doc_links_to_token_optimization_readme() -> None:
    content = _read_public_doc(_PROOFS_DOC)
    assert "features/token_optimization/README.md" in content


def test_proofs_doc_links_to_token_optimization_claims() -> None:
    content = _read_public_doc(_PROOFS_DOC)
    assert "TOKEN_OPTIMIZATION_CLAIMS.md" in content


def test_proofs_doc_does_not_contain_unqualified_numeric_percentage_claims() -> None:
    content = _read_public_doc(_PROOFS_DOC)
    offenders: list[str] = []
    for line in content.splitlines():
        if not _PERCENT_PATTERN.search(line):
            continue
        if _line_is_forbidden_example_context(line):
            continue
        offenders.append(line.strip())
    assert offenders == [], f"Unqualified percentage claims found in PROOFS.md: {offenders}"


def test_proofs_doc_states_real_user_and_commercial_validation_incomplete() -> None:
    content = _read_public_doc(_PROOFS_DOC).lower()
    assert "real-user" in content or "real user" in content
    assert "commercial validation" in content
    assert "incomplete" in content or "not currently claim" in content


def test_lkw_platform_proof_links_to_proofs_doc() -> None:
    content = _read_public_doc(_LKW_PLATFORM_PROOF)
    assert "PROOFS.md" in content


def test_token_optimization_readme_links_to_proofs_doc() -> None:
    content = _read_public_doc(_TOKEN_OPT_README)
    assert "PROOFS.md" in content


def test_claims_doc_contains_neutral_discovery_vs_performance_promotion_boundary() -> None:
    content = _read_claims_doc()
    assert "## README discovery and promotion boundary" in content
    assert "Neutral discovery allowed now" in content
    assert "Performance promotion remains gated" in content


def test_claims_doc_mentions_token_10g_and_10h_performance_gates() -> None:
    content = _read_claims_doc()
    assert "TOKEN-10G" in content
    assert "TOKEN-10H" in content


def test_ucl_adr_has_no_utf8_bom() -> None:
    raw = _UCL_ADR.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")
    assert text.startswith("# ADR-UCL-001")


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


# --- CTX-UCL-ARCH-1-R4: internal-call and single-flight guardrails ---


@pytest.mark.parametrize("concept", _R4_REQUIRED_CONCEPTS)
def test_ucl_architecture_documents_r4_required_concepts(concept: str) -> None:
    content = _read_public_doc(_UCL_ARCH)
    assert concept in content, f"Missing required UCL R4 concept: {concept!r}"


@pytest.mark.parametrize("phrase", _R4_FORBIDDEN_PHRASES)
def test_ucl_architecture_rejects_r4_regression_phrases(phrase: str) -> None:
    content = _read_public_doc(_UCL_ARCH).lower()
    assert phrase not in content, f"Forbidden R4 regression phrase in UCL arch: {phrase!r}"


def test_claims_doc_does_not_claim_runtime_single_flight_or_recursion_protection() -> None:
    content = _read_claims_doc()
    forbidden_runtime_claims = (
        "single-flight is implemented",
        "artifact reservations are operational",
        "inmemoryoptimizationartifactrepository exists",
        "internal summarizer recursion is prevented in runtime",
    )
    for claim in forbidden_runtime_claims:
        offenders = [
            line.strip()
            for line in content.splitlines()
            if claim in line.lower() and not _line_is_forbidden_example_context(line)
        ]
        assert offenders == [], f"Forbidden runtime claim in claims doc: {claim!r} -> {offenders}"


# --- UCL public status synchronization guardrails ---


@pytest.mark.parametrize("surface_name, surface_path", _UCL_PUBLIC_SURFACES)
def test_ucl_public_surface_reports_current_status(
    surface_name: str,
    surface_path: Path,
) -> None:
    normalized = _normalize_public_text(_read_public_doc(surface_path))
    _assert_required_ucl_statuses(normalized)


@pytest.mark.parametrize("surface_name, surface_path", _UCL_PUBLIC_SURFACES)
def test_ucl_public_surface_rejects_stale_status_wording(
    surface_name: str,
    surface_path: Path,
) -> None:
    normalized = _normalize_public_text(_read_public_doc(surface_path))
    _assert_no_stale_ucl_wording(normalized)


def test_public_adoption_reading_order_is_zero_through_nine_unique() -> None:
    content = _read_public_doc(_PUBLIC_ADOPTION_README)
    start = content.index("## Recommended reading order")
    section_end = content.index("\n## ", start + 1)
    section = content[start:section_end]
    steps = [int(match.group(1)) for match in re.finditer(r"^\|\s*(\d+)\s*\|", section, re.MULTILINE)]
    assert steps == list(range(10))


# --- CTX-UCL-CLOSEOUT-1: final register integrity guards ---


def _extract_markdown_section(content: str, start_heading: str, end_heading: str) -> str:
    start = content.index(start_heading)
    end = content.index(end_heading, start + len(start_heading))
    return content[start:end]


def _count_markdown_table_data_rows(section: str) -> list[str]:
    lines = section.splitlines()
    rows: list[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table and rows:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        headerish = all(set(cell) <= {"-", ":", " "} for cell in cells)
        if headerish:
            in_table = True
            continue
        if in_table:
            rows.append(stripped)
    return rows


def _historical_resolution_table_section(content: str) -> str:
    section = _extract_markdown_section(
        content,
        "### 3.2 Final resolution of the 19 historical mechanisms",
        "### 3.3 Additional closure surfaces",
    )
    marker = "| Mechanism | Final classification |"
    return section[section.index(marker) :]


_HISTORICAL_3_2_MARKERS = (
    "ConversationalMemory._trim_if_needed",
    "InMemorySessionStorage",
    "SqliteConversationalMemoryStore",
    "SessionManager.append_message",
    "fragments_from_session_history",
    "builtin.session_history",
    "DefaultContextFormatter",
    "HistoryLayer",
    "HistoryCompressionStrategy",
    "ContextCompiler",
    "TOKENIZER_HARD_TRIM",
    "ContextManager",
    "verify_context_preflight",
    "Token Optimization pipeline",
    "protected_regions.py",
    "budget_aware_packing",
    "semantic_compression_enabled",
    "SessionMemoryConsolidationService",
    "ConversationalMemory.get_for_model",
)


def test_ucl_architecture_section_32_has_exactly_19_resolution_rows() -> None:
    content = _read_public_doc(_UCL_ARCH)
    table_section = _historical_resolution_table_section(content)
    rows = _count_markdown_table_data_rows(table_section)
    assert len(rows) == 19, f"Expected 19 historical resolution rows, found {len(rows)}"


@pytest.mark.parametrize("marker", _HISTORICAL_3_2_MARKERS)
def test_ucl_architecture_section_32_resolves_each_historical_mechanism(
    marker: str,
) -> None:
    content = _read_public_doc(_UCL_ARCH)
    table_section = _historical_resolution_table_section(content)
    assert marker in table_section, f"Missing historical mechanism resolution: {marker!r}"


def test_ucl_architecture_context_manager_not_canonical_ucl() -> None:
    content = _read_public_doc(_UCL_ARCH)
    table_section = _historical_resolution_table_section(content)
    context_manager_row = next(
        line for line in table_section.splitlines() if "`ContextManager`" in line
    )
    lowered = context_manager_row.lower()
    normalized = lowered.replace("_", "").replace("-", "")
    assert "canonicalucl" not in normalized
    assert (
        "model_presentation_only" in lowered.replace("-", "_")
        or "legacy_compatibility_presentation" in lowered.replace("-", "_")
    )
    assert "outside ucl-managed" in lowered


@pytest.mark.parametrize(
    "marker",
    (
        "`protected_regions.py`",
        "`budget_aware_packing` / `context_pack.py`",
    ),
)
def test_generic_pipeline_components_are_not_canonical_ucl(
    marker: str,
) -> None:
    content = _read_public_doc(_UCL_ARCH)
    table_section = _historical_resolution_table_section(content)
    row = next(
        line
        for line in table_section.splitlines()
        if marker in line
    )
    normalized = row.lower().replace("_", "").replace("-", "")

    assert "separatetokenoptimizationpipeline" in normalized
    assert "canonicalucl" not in normalized


def test_ucl_architecture_section_33_includes_history_summary_diag() -> None:
    arch = _read_public_doc(_UCL_ARCH)
    section_33 = _extract_markdown_section(
        arch,
        "### 3.3 Additional closure surfaces",
        "**UCL-managed compile integrity fence:**",
    )
    assert "HistorySummaryDiagV1" in section_33


def test_ucl_architecture_executor_distinction() -> None:
    arch = _read_public_doc(_UCL_ARCH)
    table_section = _historical_resolution_table_section(arch)
    pipeline_row = next(
        line for line in table_section.splitlines() if "Token Optimization pipeline" in line
    )
    assert "separate_token_optimization_pipeline" in pipeline_row.lower().replace(
        "-", "_"
    )

    section_33 = _extract_markdown_section(
        arch,
        "### 3.3 Additional closure surfaces",
        "**UCL-managed compile integrity fence:**",
    )
    executor_row = next(
        line
        for line in section_33.splitlines()
        if line.strip().startswith("| `MessageSequenceArtifactExecutor`")
    )
    assert "canonical_ucl" in executor_row.lower().replace("-", "_")
    assert "sole conversation-summary executor" in executor_row.lower()


def test_ucl_plan_deferred_work_precedes_next_step() -> None:
    plan = _read_public_doc(_UCL_PLAN)
    deferred_index = plan.index("## Deferred work")
    next_index = plan.index("## Next step")
    assert deferred_index < next_index

    deferred_section = plan[deferred_index:next_index]
    deferred_rows = _count_markdown_table_data_rows(deferred_section)
    assert len(deferred_rows) >= 1, "Deferred work table must have at least one data row"