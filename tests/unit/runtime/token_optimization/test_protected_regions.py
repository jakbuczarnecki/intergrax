# © Artur Czarnecki. All rights reserved.

"""TOKEN-1B: protected-region detection and validation tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    ProtectedRegionValidationStatus,
)
from intergrax.runtime.token_optimization.protected_regions import (
    detect_protected_regions,
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
