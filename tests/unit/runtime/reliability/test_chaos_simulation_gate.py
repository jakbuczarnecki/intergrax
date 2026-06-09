# © Artur Czarnecki. All rights reserved.

"""IDEAL-22.5 / IDEAL-26.2 — chaos simulation catalog gate."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.errors.classifier import ErrorClassifier
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.reliability.compensation import CompensationFlow, CompensationStep

pytestmark = pytest.mark.gate


@pytest.mark.parametrize(
    "exc,expected",
    [
        (ConnectionError("down"), RuntimeErrorCode.DEPENDENCY_ERROR),
        (TimeoutError("slow"), RuntimeErrorCode.TIMEOUT),
    ],
)
def test_chaos_dependency_failure_taxonomy(exc: Exception, expected: RuntimeErrorCode) -> None:
    assert ErrorClassifier.classify(exc) is expected


@pytest.mark.asyncio
async def test_compensation_flow_invokes_handler() -> None:
    calls: list[str] = []

    async def handler(step_id: str, context: dict) -> None:
        calls.append(step_id)

    flow = CompensationFlow(
        steps=[CompensationStep(step_id="s1", handler_id="rollback")],
        handlers={"rollback": handler},
    )
    executed = await flow.run("s1", {})
    assert executed == ["rollback"]
    assert calls == ["s1"]
