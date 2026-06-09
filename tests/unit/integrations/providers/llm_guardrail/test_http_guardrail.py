# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.providers.llm_guardrail._vendor_opens import http_guardrail_scan

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_http_guardrail_returns_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_LAKERA_API_KEY", raising=False)
    monkeypatch.delenv("INTERGRAX_LAKERA_BASE_URL", raising=False)
    assert (
        http_guardrail_scan(
            slug="lakera",
            text="hello",
            mode="input",
            env_prefix="INTERGRAX_LAKERA",
            default_path="/v1/guard",
        )
        is None
    )
