# © Artur Czarnecki. All rights reserved.

"""TOKEN-1B / TOKEN-1B-R: protected-region detection and validation tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    ProtectedRegionValidationStatus,
)
from intergrax.runtime.token_optimization.protected_regions import (
    MAX_ENV_PROTECTED_TERMS,
    MAX_PROTECTED_TERM_LENGTH,
    PROTECTED_TERMS_ENV_VAR,
    detect_protected_regions,
    parse_protected_terms,
    validate_protected_regions,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PREVIEW_MAX_LEN = 40


def test_detects_fenced_code_blocks() -> None:
    content = "Before\n```python\nprint('x')\n```\nAfter"
    regions = detect_protected_regions(content)
    code_blocks = [r for r in regions if r.kind is ProtectedRegionKind.CODE_BLOCK]
    assert len(code_blocks) == 1
    assert "```python" in code_blocks[0].value
    assert "print('x')" in code_blocks[0].value


def test_detects_inline_code() -> None:
    content = "Use `foo_bar` in the handler."
    regions = detect_protected_regions(content)
    inline = [r for r in regions if r.kind is ProtectedRegionKind.INLINE_CODE]
    assert any(r.value == "`foo_bar`" for r in inline)


def test_detects_urls() -> None:
    content = "See https://example.com/docs for details."
    regions = detect_protected_regions(content)
    urls = [r for r in regions if r.kind is ProtectedRegionKind.URL]
    assert any("https://example.com/docs" in r.value for r in urls)


def test_detects_paths() -> None:
    content = "Config at /etc/app/config.yaml and C:\\Projects\\app\\main.py"
    regions = detect_protected_regions(content)
    paths = [r for r in regions if r.kind is ProtectedRegionKind.PATH]
    assert any("/etc/app/config.yaml" in r.value for r in paths)
    assert any("C:\\Projects\\app\\main.py" in r.value for r in paths)


def test_detects_env_vars() -> None:
    content = "Set OPENAI_API_KEY and DATABASE_URL before start."
    regions = detect_protected_regions(content)
    env_vars = [r for r in regions if r.kind is ProtectedRegionKind.ENV_VAR]
    values = {r.value for r in env_vars}
    assert "OPENAI_API_KEY" in values
    assert "DATABASE_URL" in values


def test_common_acronyms_are_not_detected_as_env_var_by_default() -> None:
    content = "LKW uses RAG and API with JSON over HTTP via CLI."
    regions = detect_protected_regions(content)
    env_vars = [r for r in regions if r.kind is ProtectedRegionKind.ENV_VAR]
    assert env_vars == []


def test_builtin_openai_api_key_detected() -> None:
    content = "Export OPENAI_API_KEY before boot."
    regions = detect_protected_regions(content)
    env_vars = [r for r in regions if r.kind is ProtectedRegionKind.ENV_VAR]
    assert any(r.value == "OPENAI_API_KEY" for r in env_vars)


def test_builtin_database_url_detected() -> None:
    content = "Set DATABASE_URL before boot."
    regions = detect_protected_regions(content)
    env_vars = [r for r in regions if r.kind is ProtectedRegionKind.ENV_VAR]
    assert any(r.value == "DATABASE_URL" for r in env_vars)


def test_env_provided_polish_terms_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PROTECTED_TERMS_ENV_VAR, "HASLO_BAZY_DANYCH,KLUCZ_API")
    content = "Configure HASLO_BAZY_DANYCH and KLUCZ_API in prod."
    regions = detect_protected_regions(content)
    env_vars = {r.value for r in regions if r.kind is ProtectedRegionKind.ENV_VAR}
    assert "HASLO_BAZY_DANYCH" in env_vars
    assert "KLUCZ_API" in env_vars


def test_env_terms_extend_builtin_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PROTECTED_TERMS_ENV_VAR, "HASLO_BAZY_DANYCH")
    content = "Needs OPENAI_API_KEY and HASLO_BAZY_DANYCH."
    regions = detect_protected_regions(content)
    env_vars = {r.value for r in regions if r.kind is ProtectedRegionKind.ENV_VAR}
    assert "OPENAI_API_KEY" in env_vars
    assert "HASLO_BAZY_DANYCH" in env_vars


def test_explicit_protected_terms_without_env() -> None:
    content = "Set MY_CUSTOM_SECRET before deploy."
    regions = detect_protected_regions(
        content,
        protected_terms=("MY_CUSTOM_SECRET",),
        include_env_protected_terms=False,
    )
    env_vars = [r for r in regions if r.kind is ProtectedRegionKind.ENV_VAR]
    assert any(r.value == "MY_CUSTOM_SECRET" for r in env_vars)


def test_include_env_protected_terms_false_keeps_builtin_and_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PROTECTED_TERMS_ENV_VAR, "HASLO_BAZY_DANYCH")
    content = "OPENAI_API_KEY and HASLO_BAZY_DANYCH and MY_TERM."
    regions = detect_protected_regions(
        content,
        protected_terms=("MY_TERM",),
        include_env_protected_terms=False,
    )
    env_vars = {r.value for r in regions if r.kind is ProtectedRegionKind.ENV_VAR}
    assert "OPENAI_API_KEY" in env_vars
    assert "MY_TERM" in env_vars
    assert "HASLO_BAZY_DANYCH" not in env_vars


def test_parse_protected_terms_supports_separators() -> None:
    assert parse_protected_terms("A,B;C\nD") == ("A", "B", "C", "D")


def test_parse_protected_terms_trims_and_ignores_empty() -> None:
    assert parse_protected_terms("  FOO  , , ; ; \n  BAR  ") == ("FOO", "BAR")


def test_parse_protected_terms_ignores_overlong_entries() -> None:
    long_term = "X" * (MAX_PROTECTED_TERM_LENGTH + 1)
    assert parse_protected_terms(f"OK_TERM,{long_term},ANOTHER") == ("OK_TERM", "ANOTHER")


def test_parse_protected_terms_caps_env_count() -> None:
    terms = ",".join(f"TERM_{i}" for i in range(MAX_ENV_PROTECTED_TERMS + 50))
    parsed = parse_protected_terms(terms)
    assert len(parsed) == MAX_ENV_PROTECTED_TERMS


def test_validation_does_not_fail_when_acronyms_disappear() -> None:
    original = "LKW uses RAG and API with JSON over HTTP."
    optimized = "local workspace uses retrieval."
    result = validate_protected_regions(original, optimized)
    assert result.status is ProtectedRegionValidationStatus.NOT_APPLICABLE


def test_detects_hashes_dates_versions() -> None:
    content = (
        "sha abcdef0123456789abcdef0123456789 on 2024-06-15 using version 1.2.3"
    )
    regions = detect_protected_regions(content)
    kinds = {r.kind for r in regions}
    assert ProtectedRegionKind.HASH in kinds
    assert ProtectedRegionKind.DATE in kinds
    assert ProtectedRegionKind.VERSION in kinds


def test_detects_exact_errors_and_commands() -> None:
    content = (
        'Failure: ValueError("disk full")\n'
        "$ uv run pytest tests/unit -q\n"
        "run_a1b2c3d4e5f67890"
    )
    regions = detect_protected_regions(content)
    kinds = {r.kind for r in regions}
    assert ProtectedRegionKind.EXACT_ERROR in kinds
    assert ProtectedRegionKind.COMMAND in kinds
    assert ProtectedRegionKind.IDENTIFIER in kinds


def test_fenced_block_supersedes_nested_inline_fragments() -> None:
    content = "```\n`nested` https://in.block\n```"
    regions = detect_protected_regions(content)
    assert any(r.kind is ProtectedRegionKind.CODE_BLOCK for r in regions)
    assert not any(r.kind is ProtectedRegionKind.INLINE_CODE for r in regions)
    assert not any(r.kind is ProtectedRegionKind.URL for r in regions)


def test_validation_returns_not_applicable_when_no_regions() -> None:
    result = validate_protected_regions("plain text", "plain text")
    assert result.status is ProtectedRegionValidationStatus.NOT_APPLICABLE
    assert result.regions_checked == 0


def test_validation_returns_passed_when_all_preserved() -> None:
    original = "Path /var/log/app.log and https://api.example.com/v1"
    optimized = "summary; /var/log/app.log; https://api.example.com/v1"
    result = validate_protected_regions(original, optimized)
    assert result.status is ProtectedRegionValidationStatus.PASSED
    assert result.regions_failed == 0
    assert result.regions_preserved == result.regions_checked


def test_validation_returns_failed_when_value_missing() -> None:
    original = "See https://example.com/secret and OPENAI_API_KEY"
    optimized = "See https://example.com/secret"
    result = validate_protected_regions(original, optimized)
    assert result.status is ProtectedRegionValidationStatus.FAILED
    assert result.regions_failed >= 1
    assert result.regions_preserved < result.regions_checked
    assert result.failures


def test_validation_accepts_explicitly_provided_regions() -> None:
    regions = (
        ProtectedRegion(
            kind=ProtectedRegionKind.URL,
            value="https://keep.me",
            start=0,
            end=16,
        ),
    )
    result = validate_protected_regions(
        "ignored original",
        "prefix https://keep.me suffix",
        regions=regions,
    )
    assert result.status is ProtectedRegionValidationStatus.PASSED
    assert result.regions_checked == 1


def test_validation_failure_does_not_include_large_raw_content() -> None:
    long_value = "x" * 500
    regions = (
        ProtectedRegion(
            kind=ProtectedRegionKind.HASH,
            value=long_value,
            start=0,
            end=500,
        ),
    )
    result = validate_protected_regions("orig", "opt", regions=regions)
    assert result.status is ProtectedRegionValidationStatus.FAILED
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert long_value not in failure
    assert "..." in failure
    assert len(failure) < len(long_value)
