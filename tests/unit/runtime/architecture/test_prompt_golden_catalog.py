# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from intergrax.runtime.architecture.prompt_golden_catalog import (
    PromptGoldenExpectation,
    load_golden_expectations,
    verify_prompt_golden_catalog,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GOLDEN_EXPECTATIONS = _REPO_ROOT / "tests" / "fixtures" / "prompt_golden" / "expectations.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_prompt_golden_catalog_passes_matching_fixture(tmp_path: Path) -> None:
    prompt_path = tmp_path / "harness.echo" / "1.yaml"
    prompt_path.parent.mkdir(parents=True)
    content = "id: harness.echo\nversion: 1\n"
    prompt_path.write_text(content, encoding="utf-8")
    expectations = (
        PromptGoldenExpectation(
            prompt_id="harness.echo",
            version=1,
            content_sha256=_sha256(content),
        ),
    )
    report = verify_prompt_golden_catalog(catalog_dir=tmp_path, expectations=expectations)
    assert report.passed is True
    assert report.checked == 1


def test_prompt_golden_catalog_fails_on_hash_mismatch(tmp_path: Path) -> None:
    prompt_path = tmp_path / "harness.echo" / "1.yaml"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("changed", encoding="utf-8")
    expectations = (
        PromptGoldenExpectation(
            prompt_id="harness.echo",
            version=1,
            content_sha256="0" * 64,
        ),
    )
    report = verify_prompt_golden_catalog(catalog_dir=tmp_path, expectations=expectations)
    assert report.passed is False
    assert report.failures


@pytest.mark.gate
def test_repo_prompt_golden_catalog_matches_expectations() -> None:
    expectations = load_golden_expectations(_GOLDEN_EXPECTATIONS)
    report = verify_prompt_golden_catalog(
        catalog_dir=_REPO_ROOT / "prompts",
        expectations=expectations,
    )
    assert report.passed is True, report.failures
