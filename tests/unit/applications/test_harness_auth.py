# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from intergrax.applications._shared.harness_auth import (
    is_harness_api_key_valid,
    require_harness_api_key,
    resolve_harness_api_key,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_resolve_harness_api_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    assert resolve_harness_api_key() is None
    assert is_harness_api_key_valid() is True


def test_resolve_harness_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "secret")
    assert resolve_harness_api_key() == "secret"
    assert is_harness_api_key_valid(x_api_key="secret") is True
    assert is_harness_api_key_valid(x_api_key="wrong") is False
    assert is_harness_api_key_valid(authorization="Bearer secret") is True


def test_require_harness_api_key_dependency_allows_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(require_harness_api_key)])
    def probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/probe").status_code == 200


def test_require_harness_api_key_dependency_rejects_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "lab-key")
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(require_harness_api_key)])
    def probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/probe").status_code == 401
    assert client.get("/probe", headers={"X-Api-Key": "lab-key"}).status_code == 200
