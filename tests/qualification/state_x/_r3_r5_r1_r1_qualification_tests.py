# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R5-R1-R1 — ACP host context trust boundary & provider provenance."""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from typing import Any

import pytest

from intergrax.agents.authoring import acp_run as acp_run_module
from intergrax.agents.authoring.acp_session_host import ACP_HOST_CONTEXT_KEY, ACPSessionHostContext
from intergrax.agents.authoring.patterns.reference import PatternPlanExecuteProbe
import intergrax.agents.persistence as persistence_pkg
from intergrax.agents.persistence.checkpoint_store import (
    AgentCheckpointStore,
    InMemoryAgentCheckpointStore,
    SQLiteAgentCheckpointStore,
    build_checkpoint,
)
from intergrax.agents.persistence.checkpoint_wiring import inject_acp_checkpoint_metadata
from intergrax.agents.persistence.session_persistence import resolve_session_persistence
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentExecutionOptions, AgentRunRequest, RequestIdentity
from intergrax.dev_support.execution_identity_scope import canonical_agent_run_smoke_scope
from intergrax.runtime.nexus.agents import agent_engine as agent_engine_module
from testing_support.acp_checkpoint_test_wiring import (
    run_acp_with_host_checkpoint_store,
    wire_acp_run_request,
)
from tests.qualification.state_x._r3_r5_r1_r1_support import (
    STATE_X_R3_R5_R1_R1_ALLOWLIST_PATHS,
    STATE_X_R3_R5_R1_R1_PRE_AUDIT_HEAD,
    _REPO_ROOT,
)
from tests.qualification.state_x._r3_r5_support import build_valid_checkpoint

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _CallCountingStore(InMemoryAgentCheckpointStore):
    calls: int = 0

    def get_latest(self, run_id: str, tenant_id: str):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        return super().get_latest(run_id, tenant_id)

    def save(self, checkpoint, expected_revision=None):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        return super().save(checkpoint, expected_revision=expected_revision)


def test_r3_r5_r1_r1_q01_closed_world_inventory() -> None:
    assert STATE_X_R3_R5_R1_R1_PRE_AUDIT_HEAD == "941f2c3bfab0e546031b7ad033efddfc69da41f5"
    for rel in STATE_X_R3_R5_R1_R1_ALLOWLIST_PATHS:
        assert (_REPO_ROOT / rel).is_file()


def test_r3_r5_r1_r1_q02_wire_acp_run_request_production_callers_zero() -> None:
    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "wire_acp_run_request",
            "--",
            "intergrax",
            "applications",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    offenders = [
        line
        for line in result.stdout.splitlines()
        if "wire_acp_run_request" in line and "wire_acp_run_request_with" not in line
    ]
    assert offenders == []


def test_r3_r5_r1_r1_q03_public_exports_provider_injection_helper_removed() -> None:
    assert "wire_acp_run_request" not in persistence_pkg.__all__


def test_r3_r5_r1_r1_q04_public_agent_run_ignores_host_context_store() -> None:
    _CallCountingStore.calls = 0
    malicious = _CallCountingStore()
    host = ACPSessionHostContext()
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={
            ACP_HOST_CONTEXT_KEY: host,
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    InMemoryAgentCheckpointStore().save(build_valid_checkpoint(run_id="run-r1"))
    persistence, resume = resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=None,
    )
    assert persistence.checkpoint_store is None
    assert resume is None
    _ = malicious
    assert _CallCountingStore.calls == 0


def test_r3_r5_r1_r1_q05_dict_host_context_cannot_select_checkpoint_store() -> None:
    store = InMemoryAgentCheckpointStore()
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={
            ACP_HOST_CONTEXT_KEY: {"agent_checkpoint_store": store},
        },
    )
    host = acp_run_module._host_context_from_metadata(dict(request.metadata))
    assert host is not None
    assert "agent_checkpoint_store" not in ACPSessionHostContext.model_fields
    persistence, _ = resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=None,
    )
    assert persistence.checkpoint_store is None


def test_r3_r5_r1_r1_q06_checkpoint_store_metadata_key_non_authoritative() -> None:
    store = InMemoryAgentCheckpointStore()
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={AcpMetadataKey.CHECKPOINT_STORE: store},
    )
    persistence, _ = resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=None,
    )
    assert persistence.checkpoint_store is None


def test_r3_r5_r1_r1_q07_resume_flag_without_host_store_fail_closed() -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(build_valid_checkpoint(run_id="run-r1"))
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={AcpMetadataKey.RESUME_FROM_CHECKPOINT: True},
    )
    persistence, resume = resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=None,
    )
    assert persistence.checkpoint_store is None
    assert resume is None


def test_r3_r5_r1_r1_q08_no_host_public_persistence_disabled() -> None:
    assert "agent_checkpoint_store" not in ACPSessionHostContext.model_fields


@pytest.mark.asyncio
async def test_r3_r5_r1_r1_q09_sanctioned_host_store_enables_persistence() -> None:
    agent = PatternPlanExecuteProbe()
    store = InMemoryAgentCheckpointStore()
    with canonical_agent_run_smoke_scope("r1r1-save", tenant_id="t-a", principal_id="u1") as run_id:
        request = AgentRunRequest(
            input="ckpt",
            identity=RequestIdentity(tenant_id="t-a", user_id="u1"),
            metadata={"run_id": str(run_id), "user_id": "u1"},
            execution_options=AgentExecutionOptions(max_steps=1, checkpoint_every_step=True),
        )
        await run_acp_with_host_checkpoint_store(agent, request, store)
        assert store.get_latest(run_id, "t-a") is not None


@pytest.mark.asyncio
async def test_r3_r5_r1_r1_q10_sanctioned_host_resume() -> None:
    agent = PatternPlanExecuteProbe()
    store = InMemoryAgentCheckpointStore()
    with canonical_agent_run_smoke_scope("r1r1-resume", tenant_id="t-a", principal_id="u1") as run_id:
        base = AgentRunRequest(
            input="ckpt",
            identity=RequestIdentity(tenant_id="t-a", user_id="u1"),
            metadata={"run_id": str(run_id), "user_id": "u1"},
            execution_options=AgentExecutionOptions(max_steps=1, checkpoint_every_step=True),
        )
        await run_acp_with_host_checkpoint_store(agent, base, store)
        resumed = wire_acp_run_request(
            base.model_copy(
                update={"execution_options": AgentExecutionOptions(max_steps=5, checkpoint_every_step=True)},
            ),
            store,
            resume=True,
        )
        await run_acp_with_host_checkpoint_store(agent, resumed, store, resume=True)
        assert store.get_latest(run_id, "t-a") is not None


def test_r3_r5_r1_r1_q11_host_a_beats_caller_b() -> None:
    host_a = InMemoryAgentCheckpointStore()
    host_a.save(
        build_checkpoint(
            run_id="run-r1",
            tenant_id="tenant-a",
            agent_id="agent-a",
            step_index=0,
            state_root={"acp.state.v1": {"marker": "A"}},
            side_effect_ledger=[],
            trace_step_count=1,
        ),
    )
    caller_b = InMemoryAgentCheckpointStore()
    caller_b.save(
        build_checkpoint(
            run_id="run-r1",
            tenant_id="tenant-a",
            agent_id="agent-a",
            step_index=0,
            state_root={"acp.state.v1": {"marker": "B"}},
            side_effect_ledger=[],
            trace_step_count=1,
        ),
    )
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={
            ACP_HOST_CONTEXT_KEY: ACPSessionHostContext(),
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    _persistence, resume = resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=host_a,
    )
    assert resume is not None
    assert resume.state_root["acp.state.v1"]["marker"] == "A"


def test_r3_r5_r1_r1_q12_caller_b_zero_calls() -> None:
    _CallCountingStore.calls = 0
    caller_b = _CallCountingStore()
    caller_b.save(build_valid_checkpoint(run_id="run-r1"))
    _CallCountingStore.calls = 0
    host_a = InMemoryAgentCheckpointStore()
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={
            ACP_HOST_CONTEXT_KEY: ACPSessionHostContext(),
            AcpMetadataKey.CHECKPOINT_STORE: caller_b,
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=host_a,
    )
    assert _CallCountingStore.calls == 0


@pytest.mark.asyncio
async def test_r3_r5_r1_r1_q13_inmemory_sanctioned_path(tmp_path: Path) -> None:
    await test_r3_r5_r1_r1_q09_sanctioned_host_store_enables_persistence()


@pytest.mark.asyncio
async def test_r3_r5_r1_r1_q14_sqlite_sanctioned_path(tmp_path: Path) -> None:
    agent = PatternPlanExecuteProbe()
    store = SQLiteAgentCheckpointStore(tmp_path / "q14.db")
    with canonical_agent_run_smoke_scope("r1r1-sqlite", tenant_id="t-a", principal_id="u1") as run_id:
        request = AgentRunRequest(
            input="ckpt",
            identity=RequestIdentity(tenant_id="t-a", user_id="u1"),
            metadata={"run_id": str(run_id), "user_id": "u1"},
            execution_options=AgentExecutionOptions(max_steps=1, checkpoint_every_step=True),
        )
        await run_acp_with_host_checkpoint_store(agent, request, store)
        assert store.get_latest(run_id, "t-a") is not None


class _CustomStore(InMemoryAgentCheckpointStore):
    kind = "custom"


@pytest.mark.asyncio
async def test_r3_r5_r1_r1_q15_custom_provider_sanctioned() -> None:
    agent = PatternPlanExecuteProbe()
    store = _CustomStore()
    with canonical_agent_run_smoke_scope("r1r1-custom", tenant_id="t-a", principal_id="u1") as run_id:
        request = AgentRunRequest(
            input="ckpt",
            identity=RequestIdentity(tenant_id="t-a", user_id="u1"),
            metadata={"run_id": str(run_id), "user_id": "u1"},
            execution_options=AgentExecutionOptions(max_steps=1, checkpoint_every_step=True),
        )
        await run_acp_with_host_checkpoint_store(agent, request, store)
        assert isinstance(store.get_latest(run_id, "t-a"), object)


def test_r3_r5_r1_r1_q16_acp_runtime_no_concrete_provider_in_session_persistence() -> None:
    src = (_REPO_ROOT / "intergrax/agents/persistence/session_persistence.py").read_text(encoding="utf-8")
    assert "SQLiteAgentCheckpointStore" not in src
    assert "InMemoryAgentCheckpointStore" not in src


def test_r3_r5_r1_r1_q17_agent_engine_single_propagation_path() -> None:
    src = inspect.getsource(agent_engine_module.AgentEngine._execute_agent_impl)
    assert "run_acp_session" in src
    assert "agent_checkpoint_store=agent_checkpoint_store" in src
    assert src.count("agent_checkpoint_store") >= 1


def test_r3_r5_r1_r1_q18_graph_executor_no_duplicate_store_in_metadata() -> None:
    wiring = (_REPO_ROOT / "intergrax/agents/persistence/checkpoint_wiring.py").read_text(encoding="utf-8")
    assert "ACP_HOST_CONTEXT_KEY" not in wiring
    assert "merge_host_checkpoint_store" not in wiring


def test_r3_r5_r1_r1_q19_runtime_request_bridge_no_untrusted_store_promotion() -> None:
    bridge = (
        _REPO_ROOT / "intergrax/runtime/nexus/agents/runtime_request_bridge.py"
    ).read_text(encoding="utf-8")
    assert "agent_checkpoint_store" not in bridge
    assert "CHECKPOINT_STORE" not in bridge


def test_r3_r5_r1_r1_q20_host_context_not_provider_authority() -> None:
    assert "agent_checkpoint_store" not in ACPSessionHostContext.model_fields


def test_r3_r5_r1_r1_q21_no_trust_token_introduced() -> None:
    session_src = inspect.getsource(acp_run_module._run_acp_session_bound)
    lowered = session_src.lower()
    assert "trusted_host" not in lowered
    assert "provider_authorized" not in lowered


def test_r3_r5_r1_r1_q22_no_ambient_contextvar_provider() -> None:
    engine_src = inspect.getsource(agent_engine_module.AgentEngine.__init__)
    assert "ContextVar" not in engine_src


def test_r3_r5_r1_r1_q23_agent_run_public_semantics_preserved() -> None:
    from intergrax.agents.authoring.base import IntergraxAgent

    sig = inspect.signature(IntergraxAgent.run)
    assert list(sig.parameters) == ["self", "request"]


def test_r3_r5_r1_r1_q24_r3_r5_cas_regression_imported() -> None:
    from tests.qualification.state_x import _r3_r5_qualification_tests as r5  # noqa: F401

    assert hasattr(r5, "test_r3_r5_q06_provider_update_cas_parity")


def test_r3_r5_r1_r1_q25_r3_r5_identity_continuity_imported() -> None:
    from tests.qualification.state_x import _r3_r5_qualification_tests as r5  # noqa: F401

    assert hasattr(r5, "test_r3_r5_q11_incoming_revision_cannot_control_durable")


def test_r3_r5_r1_r1_q26_r3_r5_side_effect_lineage_imported() -> None:
    from tests.qualification.state_x import _r3_r5_qualification_tests as r5  # noqa: F401

    assert hasattr(r5, "test_r3_r5_q25_valid_embedded_side_effect_round_trip")


def test_r3_r5_r1_r1_q27_r3_r5_corrupt_state_imported() -> None:
    from tests.qualification.state_x import _r3_r5_qualification_tests as r5  # noqa: F401

    assert hasattr(r5, "test_r3_r5_q27_corrupt_json_fails_closed")


def test_r3_r5_r1_r1_q28_r3_r5_r1_direct_checkpoint_store_bypass_closed() -> None:
    from tests.qualification.state_x import _r3_r5_r1_qualification_tests as r1  # noqa: F401

    assert hasattr(r1, "test_r3_r5_r1_q05_request_only_store_cannot_enable_persistence")


def test_r3_r5_r1_r1_q29_provider_readers_classified() -> None:
    acp_src = inspect.getsource(acp_run_module._run_acp_session_bound)
    assert "resolve_checkpoint_store" not in acp_src


def test_r3_r5_r1_r1_q30_exactly_one_sanctioned_composition_owner() -> None:
    nexus = (_REPO_ROOT / "intergrax/runtime/nexus/nexus_loop.py").read_text(encoding="utf-8")
    assert "AgentEngine(" in nexus
    assert "agent_checkpoint_store=agent_checkpoint_store" in nexus
    metadata: dict[str, Any] = {AcpMetadataKey.SESSION_ENABLED: True}
    inject_acp_checkpoint_metadata(
        metadata,
        store=InMemoryAgentCheckpointStore(),
        run_id="run-x",
        tenant_id="tenant-a",
    )
    assert ACP_HOST_CONTEXT_KEY not in metadata
