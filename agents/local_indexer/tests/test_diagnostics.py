# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.authoring.diagnostic_serialization import merge_diagnostic_payloads
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from local_indexer.diagnostics import (
    IndexSummaryDiagnostic,
    index_diagnostic_from_output,
)


@pytest.mark.unit
def test_index_summary_schema_id_and_required_fields() -> None:
    payload = IndexSummaryDiagnostic(
        accepted_count=2,
        rejected_count=1,
        ingested_count=2,
        chunk_count=5,
        source_count=3,
    )
    assert payload.schema_id() == "lkw.index_summary.v1"
    data = payload.to_dict()
    assert data["accepted_count"] == 2
    assert data["ingested_count"] == 2
    assert data["chunk_count"] == 5


@pytest.mark.unit
def test_index_diagnostic_from_output_maps_ingest_summary() -> None:
    output = {
        "ingest_summary": {
            "accepted_paths": ["a.txt"],
            "rejected_paths": [{"path": "b.txt", "reason": "source_not_found"}],
            "ingested": [{"status": "success", "num_chunks": 2}],
            "num_chunks": 2,
        }
    }
    payload = index_diagnostic_from_output(output)
    assert isinstance(payload, IndexSummaryDiagnostic)
    assert payload.accepted_count == 1
    assert payload.rejected_count == 1
    assert payload.ingested_count == 1
    assert payload.chunk_count == 2
    assert payload.rejected_reasons == ("source_not_found",)


@pytest.mark.unit
def test_merge_diagnostic_payloads_keys_by_schema_id() -> None:
    payload = IndexSummaryDiagnostic(
        accepted_count=1,
        rejected_count=0,
        ingested_count=1,
        chunk_count=1,
        source_count=1,
    )
    merged = merge_diagnostic_payloads(None, [payload])
    assert "lkw.index_summary.v1" in merged
    assert merged["lkw.index_summary.v1"]["accepted_count"] == 1


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
