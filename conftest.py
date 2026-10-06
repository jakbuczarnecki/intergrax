# © Artur Czarnecki. All rights reserved.

"""Repository-wide pytest fixtures (tests/, applications/, agents/)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from testing_support.pytest_temp_root import apply_invocation_pytest_basetemp

_REPO_ROOT = Path(__file__).resolve().parent
_BUILD_DIR = _REPO_ROOT / "build"

GATE_HARNESS_API_KEY = "gate-test-harness-key"


def pytest_configure(config: pytest.Config) -> None:
    """Ensure gitignored ``build/`` exists and assign invocation-owned basetemp."""
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    apply_invocation_pytest_basetemp(config, _REPO_ROOT)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    from tests.qualification.trace_x._trace_x_p3_r1_pass1_session import (
        pytest_runtest_logreport as _p3_r1_pass1_logreport,
    )

    _p3_r1_pass1_logreport(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    from tests.qualification.trace_x._trace_x_p3_r1_pass1_session import (
        pytest_sessionfinish as _p3_r1_pass1_sessionfinish,
    )

    _p3_r1_pass1_sessionfinish(session, exitstatus)


@pytest.fixture
def harness_auth_headers() -> dict[str, str]:
    """Headers for product hosts with harness API-key middleware enabled."""
    return {"X-Api-Key": os.environ.get("INTERGRAX_HARNESS_API_KEY", GATE_HARNESS_API_KEY)}


@pytest.fixture
def product_harness_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set harness API key for product Tier-3 host startup (identity_profile.require_api_key)."""
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", GATE_HARNESS_API_KEY)
    return GATE_HARNESS_API_KEY
