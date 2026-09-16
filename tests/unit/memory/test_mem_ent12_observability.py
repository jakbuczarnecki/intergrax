# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-12: memory observability foundation and boundary diagnostics."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_control import (
    MemoryControlGovernanceDenied,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticEvent,
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
    MemoryDiagnosticPhase,
    NoOpMemoryObservabilitySink,
    RecordingMemoryObservabilitySink,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemorySecurityContext,
    MemorySecurityStrategySet,
)
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.strategies.defaults.memory_security_governance import (
    build_default_memory_security_strategy_set,
)
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane

pytestmark = pytest.mark.gate

_REPO = Path(__file__).resolve().parents[3]
_MEMORY_ROOT = _REPO / "intergrax" / "memory"
_TENANT = "tenant-12"
_USER = "user-12"


def _identity() -> RequestIdentity:
    return RequestIdentity(tenant_id=_TENANT, user_id=_USER)


def _scope():
    return user_memory_scope(_identity())


def test_contract_recording_and_noop_sinks() -> None:
    noop = NoOpMemoryObservabilitySink()
    recording = RecordingMemoryObservabilitySink()
    emitter = MemoryDiagnosticEmitter(_sink=recording)
    event = MemoryDiagnosticEvent(
        event_id=emitter.new_event_id(),
        operation=MemoryDiagnosticOperation.REMEMBER,
        phase=MemoryDiagnosticPhase.TERMINAL,
        outcome=MemoryDiagnosticOutcome.SUCCESS,
        component=__import__(
            "intergrax.memory.contracts.memory_observability",
            fromlist=["MemoryDiagnosticComponent"],
        ).MemoryDiagnosticComponent.CONTROL_PLANE,
        reference_time_iso=emitter.reference_time_iso(),
        tenant_id=_TENANT,
    )
    emitter.emit(event)
    noop.record(event)
    assert len(recording.events) == 1


def test_emitter_isolates_exploding_sink() -> None:
    class _ExplodingSink:
        def record(self, event: MemoryDiagnosticEvent) -> None:
            raise RuntimeError("sink down")

    emitter = MemoryDiagnosticEmitter(_sink=_ExplodingSink())
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store=store, diagnostic_emitter=emitter, tenant_id=_TENANT)
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
        diagnostic_emitter=emitter,
    )

    async def _run() -> None:
        result = await plane.remember(
            _identity(),
            _scope(),
            MemoryControlRememberRequest(content="observable fact"),
        )
        assert result.entry_id

    import asyncio

    asyncio.run(_run())


def test_governance_deny_emits_without_successful_remember() -> None:
    recording = RecordingMemoryObservabilitySink()
    emitter = MemoryDiagnosticEmitter(_sink=recording)
    denied = MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        policy_id="test.deny",
        policy_version="1",
        operation=MemoryGovernanceOperation.REMEMBER,
    )

    class DenyAuthorization:
        policy_id = "test.deny"
        policy_version = "1"

        def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
            return denied

    strategies = build_default_memory_security_strategy_set()
    custom = MemorySecurityStrategySet(
        authorization=DenyAuthorization(),
        trust=strategies.trust,
        admission=strategies.admission,
        governance=strategies.governance,
        retention=strategies.retention,
    )
    governance = MemorySecurityGovernanceService(
        strategies=custom,
        diagnostic_emitter=emitter,
    )
    store = InMemoryUserProfileStore()
    manager = UserProfileManager(store=store, tenant_id=_TENANT)
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
        security_governance=governance,
        diagnostic_emitter=emitter,
    )

    async def _run() -> None:
        with pytest.raises(MemoryControlGovernanceDenied):
            await plane.remember(
                _identity(),
                _scope(),
                MemoryControlRememberRequest(content="blocked"),
            )

    import asyncio

    asyncio.run(_run())
    governance_events = [
        e
        for e in recording.events
        if e.phase is MemoryDiagnosticPhase.GOVERNANCE
    ]
    assert governance_events
    assert any(e.outcome is MemoryDiagnosticOutcome.DENIED for e in governance_events)
    assert not any(
        e.operation is MemoryDiagnosticOperation.REMEMBER
        and e.outcome is MemoryDiagnosticOutcome.SUCCESS
        for e in recording.events
    )


def test_policy_failure_visible() -> None:
    recording = RecordingMemoryObservabilitySink()
    emitter = MemoryDiagnosticEmitter(_sink=recording)

    class BrokenPolicy:
        policy_id = "broken"
        policy_version = "1"

        def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
            raise RuntimeError("policy exploded")

    strategies = build_default_memory_security_strategy_set()
    custom = MemorySecurityStrategySet(
        authorization=BrokenPolicy(),
        trust=strategies.trust,
        admission=strategies.admission,
        governance=strategies.governance,
        retention=strategies.retention,
    )
    governance = MemorySecurityGovernanceService(
        strategies=custom,
        diagnostic_emitter=emitter,
    )
    request = MemoryGovernanceEvaluationRequest(
        context=MemorySecurityContext(
            identity=_identity(),
            scope=_scope(),
            operation=MemoryGovernanceOperation.REMEMBER,
        ),
    )
    decision = governance.evaluate(request)
    assert decision.reason_code is MemoryGovernanceReasonCode.POLICY_FAILURE
    assert any(
        e.reason_code is MemoryGovernanceReasonCode.POLICY_FAILURE for e in recording.events
    )


def test_remember_success_terminal_event() -> None:
    recording = RecordingMemoryObservabilitySink()
    emitter = MemoryDiagnosticEmitter(_sink=recording)
    plane = build_default_memory_control_plane(
        user_profile_manager=UserProfileManager(
            store=InMemoryUserProfileStore(),
            tenant_id=_TENANT,
        ),
        memory_diagnostic_emitter=emitter,
    )

    async def _run() -> str:
        result = await plane.remember(
            _identity(),
            _scope(),
            MemoryControlRememberRequest(content="hello memory"),
        )
        return result.entry_id

    import asyncio

    entry_id = asyncio.run(_run())
    success = [
        e
        for e in recording.events
        if e.operation is MemoryDiagnosticOperation.REMEMBER
        and e.outcome is MemoryDiagnosticOutcome.SUCCESS
    ]
    assert success
    assert success[0].memory_id == entry_id


def test_diagnostic_event_has_no_raw_content_fields() -> None:
    forbidden = {"content", "prompt", "message", "summary", "body", "text"}
    for field in fields(MemoryDiagnosticEvent):
        assert field.name not in forbidden


def test_shared_sink_injection_via_control_plane_wiring() -> None:
    recording = RecordingMemoryObservabilitySink()
    plane = build_default_memory_control_plane(
        user_profile_manager=UserProfileManager(
            store=InMemoryUserProfileStore(),
            tenant_id=_TENANT,
        ),
        memory_observability_sink=recording,
    )
    assert isinstance(plane, DefaultMemoryControlPlane)


_VENDOR_PATTERNS = ("opentelemetry", "prometheus_client", "datadog", "newrelic")


def test_memory_domain_no_vendor_observability_imports() -> None:
    violations: list[str] = []
    for path in _MEMORY_ROOT.rglob("*.py"):
        if "contracts" in path.parts and path.name == "memory_observability.py":
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _VENDOR_PATTERNS:
                        violations.append(f"{path}:{node.lineno}:{alias.name}")
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in _VENDOR_PATTERNS:
                    violations.append(f"{path}:{node.lineno}:{node.module}")
    assert not violations


def test_observability_contracts_no_untyped_payload_fields() -> None:
    source = inspect.getsource(
        __import__(
            "intergrax.memory.contracts.memory_observability",
            fromlist=["MemoryDiagnosticEvent"],
        )
    )
    assert "dict[str" not in source
    assert ": Any" not in source
    assert ": object" not in source
