# © Artur Czarnecki. All rights reserved.

"""ExecutionReconstructionReader contract gates (OBS-DIAG-CONFORMANCE-R1)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_reconstruction import ExecutionReconstructionReader
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
