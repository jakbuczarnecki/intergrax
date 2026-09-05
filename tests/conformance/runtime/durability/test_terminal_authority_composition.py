# © Artur Czarnecki. All rights reserved.

"""P0C-8A — production composition roots share one terminal authority."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_worker_bootstrap_does_not_construct_checkpoint_terminal_when_kv_authority_exists() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/task/worker_bootstrap.py").read_text(
        encoding="utf-8",
    )
    assert "execution_terminal=admission.execution_terminal" in source


def test_nexus_worker_runtime_receives_explicit_execution_terminal() -> None:
    source = (
        _REPO_ROOT / "intergrax/runtime/task/nexus_worker_execution.py"
    ).read_text(encoding="utf-8")
    assert "execution_terminal=execution_terminal" in source


def test_nexus_factory_uses_terminal_provider_resolver() -> None:
    source = (_REPO_ROOT / "intergrax/applications/_shared/nexus_factory.py").read_text(
        encoding="utf-8",
    )
    assert "resolved_execution_terminal" in source
    assert "resolve_execution_terminal_store" in source
