# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from lkw_shared.diagnostics import index_diagnostic_from_output


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kernel_propagates_typed_diagnostic_payloads() -> None:
    output = {
        "ingest_summary": {
            "accepted_paths": ["a.txt"],
            "rejected_paths": [],
            "ingested": [{"status": "success"}],
            "num_chunks": 1,
        }
    }
    outcome = StepOutcome.complete(
        output,
        diagnostic_payloads=[index_diagnostic_from_output(output)],
    )
    step_ctx = AgentStepContext(step_index=0, metadata={"step_id": "local_indexer_step"})
    kernel_ctx = StepKernelContext(
        agent_id="local_indexer",
        run_id="run-diag",
        allow_permissive_missing_policy=True,
    )
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    diagnostics = record.step_record.diagnostics
    assert diagnostics is not None
    assert "lkw.index_summary.v1" in diagnostics
    assert diagnostics["lkw.index_summary.v1"]["accepted_count"] == 1
