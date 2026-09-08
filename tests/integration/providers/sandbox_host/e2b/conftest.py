# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for E2B physical provider qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.proof.intergrax_proof_environment import bootstrap_process_environment

_REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_e2b_qualification_process_environment() -> None:
    """Load repository ``.env`` into the process without overwriting existing vars."""
    bootstrap_process_environment(
        proof_package_dir=_REPO_ROOT,
        repository_root=_REPO_ROOT,
    )
