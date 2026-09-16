# © Artur Czarnecki. All rights reserved.

"""ExecutionReconstructionReader contract gates (OBS-DIAG-CONFORMANCE-R1)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_reconstruction import (
    ExecutionReconstruction,
    ExecutionReconstructionReader,
)
from intergrax.runtime.observability.reconstruction import (
    ExecutionReconstruction as LegacyExecutionReconstruction,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]


def test_default_execution_reconstructor_satisfies_reader_contract() -> None:
    reconstructor = ExecutionReconstructor(
        runtime_events=InMemoryRuntimeEventStore(),
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    assert isinstance(reconstructor, ExecutionReconstructionReader)


def test_default_reconstructor_returns_canonical_contract_reconstruction_type() -> None:
    reconstructor = ExecutionReconstructor(
        runtime_events=InMemoryRuntimeEventStore(),
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    from intergrax.contracts.execution_identity import mint_run_id, mint_task_id

    result = reconstructor.reconstruct_execution(
        "tenant-a",
        mint_task_id(),
        mint_run_id(),
    )
    assert type(result) is ExecutionReconstruction
    assert LegacyExecutionReconstruction is ExecutionReconstruction
