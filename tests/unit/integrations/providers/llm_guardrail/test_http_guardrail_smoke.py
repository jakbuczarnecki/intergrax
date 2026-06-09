# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.providers.llm_guardrail._factory import create_guardrail_backend

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lakera_http_adapter_parses_block_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_LAKERA_BASE_URL", "https://lakera.test")
    monkeypatch.setenv("INTERGRAX_LAKERA_API_KEY", "test-key")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"allowed": False, "categories": ["prompt_injection"]}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _Response())

    backend = create_guardrail_backend("lakera")
    result = backend.scan_input("jailbreak attempt", context=None)
    assert result.allowed is False
    assert "prompt_injection" in result.categories
