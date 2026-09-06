# © Artur Czarnecki. All rights reserved.

"""Directory content digest hardening tests (P2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agent_distribution.runtime_context_staging import directory_content_digest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_directory_digest_ignores_pycache(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    baseline = directory_content_digest(root)

    pycache = root / "__pycache__"
    pycache.mkdir()
    (pycache / "module.cpython-312.pyc").write_bytes(b"compiled")

    assert directory_content_digest(root) == baseline


def test_directory_digest_changes_when_authoritative_source_changes(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    source = root / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    baseline = directory_content_digest(root)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert directory_content_digest(root) != baseline
