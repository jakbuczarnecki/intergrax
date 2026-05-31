# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector, record_llm_call, set_metrics_enabled
from intergrax.llm_adapters.tracking.prometheus_push import push_llm_metrics_to_gateway

pytestmark = pytest.mark.unit


def test_push_disabled_without_url() -> None:
    assert push_llm_metrics_to_gateway(url="") is False


def test_push_sends_prometheus_body() -> None:
    set_metrics_enabled(True)
    get_llm_metrics_collector().reset()
    record_llm_call(
        provider="openai",
        model="m",
        run_id="r",
        input_tokens=1,
        output_tokens=1,
        duration_ms=1,
        success=True,
    )
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as urlopen:
        ok = push_llm_metrics_to_gateway(url="http://pg:9091", job="intergrax_llm")
    assert ok is True
    assert urlopen.called
    set_metrics_enabled(False)
