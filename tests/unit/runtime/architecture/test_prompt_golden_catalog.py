# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from intergrax.runtime.architecture.prompt_golden_catalog import (
    PromptGoldenExpectation,
    verify_prompt_golden_catalog,
)

pytestmark = pytest.mark.unit


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_prompt_golden_catalog_passes_matching_fixture(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "harness.echo" / "v1.yaml"
    prompt_dir.parent.mkdir(parents=True)
    content = "id: harness.echo\nversion: 1\n"
    prompt_dir.write_text(content, encoding="utf-8")
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
    prompt_dir = tmp_path / "harness.echo" / "v1.yaml"
    prompt_dir.parent.mkdir(parents=True)
    prompt_dir.write_text("changed", encoding="utf-8")
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
