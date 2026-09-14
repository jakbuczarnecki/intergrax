# © Artur Czarnecki. All rights reserved.

"""EE-B2-FINAL — partial fan-out / recovery slot containment (NPSC-5B + 5E alignment)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_POLICY = ConcurrentExecutionWorkPolicy(max_concurrency=3)
_REPO = Path(__file__).resolve().parents[4]
_NPSC5E_PARTIAL = (
    _REPO
    / "tests"
    / "unit"
    / "runtime"
    / "architecture"
    / "test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py"
)


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _req(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


class _SlotPort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def execute(self, request: ExecutionRequest[str, _WorkResult]) -> _WorkResult:
        label = request.input
        self.counts[label] = self.counts.get(label, 0) + 1
        if label == "B":
            raise RuntimeError("slot_b_fault")
        return _WorkResult(value=label)


@pytest.mark.asyncio
async def test_ee_b2_final_partial_fanout_abc_slot_b_fails_once_each() -> None:
    port = _SlotPort()
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_req("A"), _req("B"), _req("C")),
        policy=_POLICY,
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[2].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert port.counts == {"A": 1, "B": 1, "C": 1}


def test_ee_b2_final_npsc5e_partial_recovery_gate_module_present() -> None:
    assert _NPSC5E_PARTIAL.is_file()
