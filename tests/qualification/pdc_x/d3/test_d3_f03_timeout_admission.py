# © Artur Czarnecki. All rights reserved.

"""PDC-D3 F-D3-03 (documentation honesty) — ``timeout_sec`` is an admission budget, not an SDK timeout.

Enforcing ``timeout_sec`` on the provider call is an excluded finding (it needs per-provider SDK timeout
mapping that cannot be verified offline). These tests pin what the setting really does today so that a
future enforcement ticket has to change them deliberately.
"""

from __future__ import annotations

from typing import Any

import pytest
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.runtime.execution.deadline_provider_guard import ExecutionProtectedWorkDeniedError
from tests.qualification.pdc_x.d3.fakes import BufferedOnlyAdapter

pytestmark = pytest.mark.unit


def _adapter(timeout_sec: float | None) -> BufferedOnlyAdapter:
    adapter = BufferedOnlyAdapter()
    adapter.call_config = LLMCallConfig(timeout_sec=timeout_sec)
    return adapter


def test_an_exhausted_active_deadline_budget_is_rejected_before_the_sdk_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import intergrax.runtime.execution.deadline_provider_guard as guard

    monkeypatch.setattr(guard, "resolve_active_provider_timeout_seconds", lambda configured: 0.0)
    sdk_calls: list[str] = []

    with pytest.raises(ExecutionProtectedWorkDeniedError):
        _adapter(30.0)._execute(lambda: sdk_calls.append("called"))

    assert sdk_calls == []


def test_the_effective_budget_is_resolved_from_timeout_sec_and_the_active_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import intergrax.runtime.execution.deadline_provider_guard as guard

    seen: list[Any] = []

    def _resolve(configured: Any) -> Any:
        seen.append(configured)
        return configured

    monkeypatch.setattr(guard, "resolve_active_provider_timeout_seconds", _resolve)
    result = _adapter(12.5)._execute(lambda: "ok")

    assert result == "ok"
    assert seen == [12.5]


def test_timeout_sec_is_not_forwarded_to_the_sdk_callable() -> None:
    """The callable passed to ``_execute`` receives no timeout; documented in ``LLMCallConfig``."""
    received: list[tuple[Any, ...]] = []

    def _sdk_call(*args: Any) -> str:
        received.append(args)
        return "ok"

    assert _adapter(0.01)._execute(_sdk_call) == "ok"
    assert received == [()]


def test_the_contract_docstring_states_that_timeout_sec_is_an_admission_budget() -> None:
    doc = " ".join((LLMCallConfig.__doc__ or "").split())
    assert "timeout_sec" in doc
    assert "admission" in doc
    assert "not forwarded" in doc
