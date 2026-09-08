# © Artur Czarnecki. All rights reserved.

"""Unit tests for E2B credential discovery — no physical provider required."""

from __future__ import annotations

import pytest

from tests.integration.providers.sandbox_host.e2b.qualification.credentials import (
    E2bCredentialStatus,
    resolve_e2b_credentials,
)

pytestmark = pytest.mark.qualification


def test_resolve_e2b_credentials_unavailable_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_E2B_API_KEY", raising=False)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    assert resolve_e2b_credentials() is E2bCredentialStatus.UNAVAILABLE


def test_resolve_e2b_credentials_available_from_intergrax_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("INTERGRAX_E2B_API_KEY", "test-intergrax-key")
    assert resolve_e2b_credentials() is E2bCredentialStatus.AVAILABLE


def test_resolve_e2b_credentials_available_from_e2b_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_E2B_API_KEY", raising=False)
    monkeypatch.setenv("E2B_API_KEY", "test-e2b-key")
    assert resolve_e2b_credentials() is E2bCredentialStatus.AVAILABLE


def test_intergrax_key_priority_over_e2b_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_E2B_API_KEY", "preferred-key")
    monkeypatch.setenv("E2B_API_KEY", "fallback-key")
    from intergrax.integrations.providers.sandbox_host.e2b.config import E2bSandboxHostConfig

    assert E2bSandboxHostConfig.from_env().resolved_api_key() == "preferred-key"
