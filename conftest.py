# © Artur Czarnecki. All rights reserved.

"""Repository-wide pytest fixtures (tests/, applications/, agents/)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent
_BUILD_DIR = _REPO_ROOT / "build"

GATE_HARNESS_API_KEY = "gate-test-harness-key"


def pytest_configure(config: pytest.Config) -> None:
    """Ensure gitignored ``build/`` exists before xdist basetemp/cache setup."""
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def harness_auth_headers() -> dict[str, str]:
    """Headers for product hosts with harness API-key middleware enabled."""
    return {"X-Api-Key": os.environ.get("INTERGRAX_HARNESS_API_KEY", GATE_HARNESS_API_KEY)}


@pytest.fixture
def product_harness_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set harness API key for product Tier-3 host startup (identity_profile.require_api_key)."""
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", GATE_HARNESS_API_KEY)
    return GATE_HARNESS_API_KEY
