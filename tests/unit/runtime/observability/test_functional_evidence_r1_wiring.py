# © Artur Czarnecki. All rights reserved.

"""OBS-FUNCTIONAL-CONTRACTS-1-R1 explicit wiring and external provider pluginability."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.contracts.functional_evidence import PlatformFunctionalEvidence
from intergrax.contracts.functional_evidence.persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidenceQueryPage,
    FunctionalEvidenceQueryRequest,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.observability.functional_evidence.functional_evidence_persistence_conformance import (
    sample_functional_evidence,
)
from intergrax.runtime.observability.functional_evidence_recorder import recorder_from_exec_ctx
from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
    FunctionalEvidenceRuntimeWiring,
    attach_functional_evidence_recorder_from_tool_wiring,
    functional_evidence_wiring_extra_key,
    wire_functional_evidence_runtime,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@dataclass
class _ExternalFunctionalEvidencePersistence(FunctionalEvidencePersistence):
    _items: list[PlatformFunctionalEvidence] = field(default_factory=list)

    def append(self, evidence: PlatformFunctionalEvidence) -> PlatformFunctionalEvidence:
        self._items.append(evidence)
        return evidence

    def query_evidence(self, request: FunctionalEvidenceQueryRequest) -> FunctionalEvidenceQueryPage:
        filtered = [
            item
            for item in self._items
            if item.scope.tenant_id == request.tenant_id
            and item.scope.task_id == request.task_id
            and item.scope.run_id == request.run_id
        ]
        return FunctionalEvidenceQueryPage(
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            run_id=request.run_id,
            items=tuple(filtered),
            next_cursor=None,
        )


def test_external_provider_can_be_wired_without_core_changes() -> None:
    external = _ExternalFunctionalEvidencePersistence()
    wiring = wire_functional_evidence_runtime(persistence=external)
    scope = sample_functional_evidence().scope
    exec_ctx = _minimal_exec_ctx(scope)
    wiring.recorder.record_candidate_rank(
        scope=wiring.recorder.scope_from_exec_ctx(exec_ctx, tenant_id=scope.tenant_id),
        operation_id="op-1",
        query_id="q-1",
        candidate_artifact_ref="artifact://candidate",
        rank=1,
        selected=True,
    )
    assert external._items


def test_explicit_tool_wiring_attaches_recorder_without_runtime_state_traversal() -> None:
    wiring = wire_functional_evidence_runtime(cursor_secret=b"k" * 32)
    tool_ctx = ToolWiringContext()
    tool_ctx.extras[functional_evidence_wiring_extra_key()] = wiring
    exec_ctx = _minimal_exec_ctx(sample_functional_evidence().scope)
    attach_functional_evidence_recorder_from_tool_wiring(exec_ctx, tool_ctx)
    assert recorder_from_exec_ctx(exec_ctx) is wiring.recorder


def _minimal_exec_ctx(scope) -> RuntimeExecutionContext:
    from intergrax.contracts.agent_contract_meta import AgentContract
    from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

    contract = AgentContract(
        id="agent.test",
        name="agent.test",
        description="test agent",
        version="1",
        allowed_tools=[],
    )
    request = RuntimeRequest(
        agent_id="agent.test",
        tenant_id=scope.tenant_id,
        user_id="user",
        session_id="session",
        message="hello",
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    return RuntimeExecutionContext(
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        execution_id=scope.execution_id,
        agent_id="agent.test",
        contract=contract,
        request=request,
    )
