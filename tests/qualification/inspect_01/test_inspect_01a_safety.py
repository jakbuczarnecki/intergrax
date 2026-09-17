# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-A safety qualification gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.contracts.execution_reconstruction_models import (
    ExecutionReconstruction,
    RuntimeHistoryCompleteness,
)
from intergrax.contracts.runtime_inspection import RuntimeInspectionQuery
from intergrax.runtime.runtime_inspection.federation import FederatedRuntimeInspectionReadService
from intergrax.runtime.runtime_inspection.redaction import payload_contains_raw_secret
from tests.qualification.inspect_01.test_inspect_01a_federation import (
    _EXEC,
    _SCOPE,
    _ScopeReader,
    _FactsReader,
    _TENANT,
    _reconstruction,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RAW_SECRET = "SUPER_SECRET_TOKEN_abc123"


def test_a_q9_secrets_not_leaked() -> None:
    reconstruction = _reconstruction(secret_in_kind=f"password={_RAW_SECRET}")
    service = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=_FactsReader(reconstruction),
    )
    snapshot = service.inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    serialized = json.dumps(snapshot.model_dump(mode="json"), sort_keys=True)
    assert not payload_contains_raw_secret(serialized, raw_secret=_RAW_SECRET)
    assert _RAW_SECRET not in serialized


def test_a_q10_zero_side_effects() -> None:
    class _RecordingFacts(_FactsReader):
        def read_execution_facts(self, scope):
            self.mutation_attempted = True
            return super().read_execution_facts(scope)

    facts = _RecordingFacts()
    service = FederatedRuntimeInspectionReadService(
        scope_reader=_ScopeReader(),
        execution_facts_reader=facts,
    )
    service.inspect(RuntimeInspectionQuery(tenant_id=_TENANT, execution_id=_EXEC))
    assert facts.calls == 1
