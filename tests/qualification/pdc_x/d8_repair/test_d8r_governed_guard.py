# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-H (F-D8-08, F-D8-12, F-D8-16): the governed port honours decisions and audits failures."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from intergrax.contracts.agent_runtime_governance import (
    ToolAuthorizationDecision,
    ToolAuthorizationDecisionState,
)
from intergrax.contracts.artifact_ref import ArtifactClass
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactAccessDenied,
    ArtifactReadRequest,
    GovernedArtifactContent,
    GovernedArtifactReadPort,
)
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource
from tests.qualification.pdc_x.d8._d8_support import (
    TENANT_A,
    TENANT_B,
    MutableClock,
    make_artifact,
    make_context,
    make_governance,
    reader_grants,
)
from tests.qualification.pdc_x.d8_repair._support import make_clocked_port
from tests.qualification.pdc_x.d8_repair._support import request as make_request

pytestmark = pytest.mark.unit


class _ReturningGovernance:
    """A governance port that returns a decision instead of raising."""

    def __init__(self, state: ToolAuthorizationDecisionState) -> None:
        self._state = state

    def authorize_tool(self, request: Any, *, policy_context: Any = None) -> ToolAuthorizationDecision:
        return ToolAuthorizationDecision(decision=self._state, reason="test")


def _store() -> InMemoryArtifactStore:
    store = InMemoryArtifactStore()
    store.put(make_artifact())
    return store


def _checks(harness: Any) -> list[str]:
    return [
        result.policy_id
        for event in harness.sink.events
        if event.capability.endswith(".outcome")
        for result in event.policy_results
    ]


def test_a_returned_deny_decision_denies_the_read() -> None:
    port, harness = make_clocked_port(_store(), governance=_ReturningGovernance(ToolAuthorizationDecisionState.DENY))
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.read(make_context(), make_request())
    assert exc.value.reason_code == "not_authorized"
    assert "artifact.port.not_authorized" in _checks(harness)


def test_a_returned_approval_required_decision_denies_the_read() -> None:
    port, _ = make_clocked_port(
        _store(), governance=_ReturningGovernance(ToolAuthorizationDecisionState.REQUIRE_APPROVAL)
    )
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.read(make_context(), make_request())
    assert exc.value.reason_code == "approval_required"


def test_a_returned_allow_decision_still_reads() -> None:
    port, _ = make_clocked_port(_store(), governance=_ReturningGovernance(ToolAuthorizationDecisionState.ALLOW))
    assert port.read(make_context(), make_request()).content == b"payload"


class _ExplodingSource:
    def read(self, request: ArtifactReadRequest) -> GovernedArtifactContent | None:
        raise RuntimeError("backend down")


def test_an_unexpected_backend_failure_leaves_an_outcome_event_and_propagates() -> None:
    harness = make_governance(granted=reader_grants())
    port = GovernedArtifactReadPort(
        governance=harness.boundary,
        audit=harness.recorder,
        sources={ArtifactClass.EXECUTION_EVIDENCE: _ExplodingSource()},
        clock=MutableClock(),
    )
    with pytest.raises(RuntimeError, match="backend down"):
        port.read(make_context(), make_request())
    assert "artifact.port.backend_error" in _checks(harness)
    outcome = [e for e in harness.sink.events if e.capability == "artifact.read.outcome"][0]
    assert "backend down" not in repr(outcome)  # the evidence names the error type, never its message


class _HostileSource:
    """Returns the reference of another tenant for whatever is asked."""

    def __init__(self, store: InMemoryArtifactStore) -> None:
        self._inner = ExecutionArtifactSource(store)

    def read(self, request: ArtifactReadRequest) -> GovernedArtifactContent | None:
        found = self._inner.read(replace(request, tenant_id=TENANT_B))
        assert found is not None
        return found


def test_a_source_returning_another_tenants_reference_is_denied() -> None:
    store = InMemoryArtifactStore()
    store.put(make_artifact(tenant_id=TENANT_B))
    harness = make_governance(granted=reader_grants())
    port = GovernedArtifactReadPort(
        governance=harness.boundary,
        audit=harness.recorder,
        sources={ArtifactClass.EXECUTION_EVIDENCE: _HostileSource(store)},
        clock=MutableClock(),
    )
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.read(make_context(tenant_id=TENANT_A), make_request(tenant_id=TENANT_A))
    assert exc.value.reason_code == "tenant_mismatch"
