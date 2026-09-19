# © Artur Czarnecki. All rights reserved.

"""Fixtures for Tier-3 application integration tests."""

from __future__ import annotations

import pytest

from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret

_PRODUCT_HARNESS_API_KEY = "gate-test-harness-key"


@pytest.fixture(autouse=True)
def _product_host_harness_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", _PRODUCT_HARNESS_API_KEY)


@pytest.fixture(autouse=True)
def _diagnostic_problem_list_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
