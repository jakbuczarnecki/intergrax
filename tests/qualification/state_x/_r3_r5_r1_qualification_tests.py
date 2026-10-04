# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R5-R1 — checkpoint provider provenance & sanctioned composition."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from intergrax.agents.authoring import acp_run as acp_run_module
from intergrax.agents.authoring.acp_session_host import ACP_HOST_CONTEXT_KEY, ACPSessionHostContext
from intergrax.agents.persistence.checkpoint_store import (
    AgentCheckpointStore,
    InMemoryAgentCheckpointStore,
    SQLiteAgentCheckpointStore,
    build_checkpoint,
)
from intergrax.agents.persistence.checkpoint_wiring import (
    inject_acp_checkpoint_metadata,
    wire_acp_run_request,
)
from intergrax.agents.persistence.session_persistence import resolve_session_persistence
from intergrax.applications._shared.acp_checkpoint_host_wiring import resolve_host_agent_checkpoint_store
from intergrax.applications._shared.acp_checkpoint_task_enricher import make_acp_checkpoint_task_enricher
from intergrax.applications._shared.acp_session_host_wiring import (
    build_acp_session_host_context,
)
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.runtime.task.task import Task
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from tests.qualification.state_x._r3_r5_r1_support import (
    STATE_X_R3_R5_R1_ALLOWLIST_PATHS,
    STATE_X_R3_R5_R1_PRE_AUDIT_HEAD,
    _REPO_ROOT,
)
from tests.qualification.state_x._r3_r5_support import (
    agent_checkpoint_store_implementations,
    build_valid_checkpoint,
    parity_checkpoint_store_factories,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _CallCountingStore(InMemoryAgentCheckpointStore):
    calls: int = 0

    def get_latest(self, run_id: str, tenant_id: str):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        return super().get_latest(run_id, tenant_id)

    def save(self, checkpoint, expected_revision=None):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        return super().save(checkpoint, expected_revision=expected_revision)


def test_r3_r5_r1_q01_closed_world_provider_composition_inventory() -> None:
    assert STATE_X_R3_R5_R1_PRE_AUDIT_HEAD
    for rel in STATE_X_R3_R5_R1_ALLOWLIST_PATHS:
        assert (_REPO_ROOT / rel).is_file()


def test_r3_r5_r1_q02_exactly_one_sanctioned_production_composition_chain() -> None:
    wiring = (_REPO_ROOT / "intergrax/applications/_shared/acp_checkpoint_host_wiring.py").read_text(
        encoding="utf-8",
    )
    assert "resolve_host_agent_checkpoint_store" in wiring
    session_src = inspect.getsource(acp_run_module._run_acp_session_bound)
    assert "host.agent_checkpoint_store" in session_src


def test_r3_r5_r1_q03_acp_session_receives_store_through_host_carrier() -> None:
    fields = ACPSessionHostContext.model_fields
    assert "agent_checkpoint_store" in fields
    assert "AgentCheckpointStore" in str(fields["agent_checkpoint_store"].annotation)


def test_r3_r5_r1_q04_resolve_session_persistence_no_metadata_checkpoint_store() -> None:
    source = inspect.getsource(resolve_session_persistence)
    assert "resolve_checkpoint_store" not in source
    assert "CHECKPOINT_STORE" not in source


def test_r3_r5_r1_q05_request_only_store_cannot_enable_persistence() -> None:
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


def test_r3_r5_r1_q06_request_only_store_cannot_enable_resume() -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(build_valid_checkpoint(run_id="run-r1"))
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={
            AcpMetadataKey.CHECKPOINT_STORE: store,
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    _persistence, resume = resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=None,
    )
    assert resume is None


def test_r3_r5_r1_q07_host_store_wins_over_conflicting_request_store() -> None:
    host_store = InMemoryAgentCheckpointStore()
    host_store.save(
        build_checkpoint(
            run_id="run-r1",
            tenant_id="tenant-a",
            agent_id="agent-a",
            step_index=0,
            state_root={"acp.state.v1": {"marker": "host"}},
            side_effect_ledger=[],
            trace_step_count=1,
        ),
    )
    request_store = InMemoryAgentCheckpointStore()
    request_store.save(
        build_checkpoint(
            run_id="run-r1",
            tenant_id="tenant-a",
            agent_id="agent-a",
            step_index=0,
            state_root={"acp.state.v1": {"marker": "request"}},
            side_effect_ledger=[],
            trace_step_count=1,
        ),
    )
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={
            AcpMetadataKey.CHECKPOINT_STORE: request_store,
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    _persistence, resume = resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=host_store,
    )
    assert resume is not None
    assert resume.state_root["acp.state.v1"]["marker"] == "host"


def test_r3_r5_r1_q08_malicious_request_store_zero_calls() -> None:
    _CallCountingStore.calls = 0
    malicious = _CallCountingStore()
    malicious.save(build_valid_checkpoint(run_id="run-mal"))
    _CallCountingStore.calls = 0
    host_store = InMemoryAgentCheckpointStore()
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={
            AcpMetadataKey.CHECKPOINT_STORE: malicious,
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    resolve_session_persistence(
        request,
        run_id="run-mal",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=host_store,
    )
    assert _CallCountingStore.calls == 0


def test_r3_r5_r1_q09_host_absent_request_store_fail_closed() -> None:
    _CallCountingStore.calls = 0
    malicious = _CallCountingStore()
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={AcpMetadataKey.CHECKPOINT_STORE: malicious},
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
    assert _CallCountingStore.calls == 0


def test_r3_r5_r1_q10_resume_flag_cannot_make_request_store_authoritative() -> None:
    _CallCountingStore.calls = 0
    malicious = _CallCountingStore()
    malicious.save(build_valid_checkpoint(run_id="run-r1"))
    _CallCountingStore.calls = 0
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={
            AcpMetadataKey.CHECKPOINT_STORE: malicious,
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    _persistence, resume = resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=None,
    )
    assert resume is None
    assert _CallCountingStore.calls == 0


def _resume_via_host_store(store: AgentCheckpointStore) -> None:
    host = ACPSessionHostContext(agent_checkpoint_store=store)
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata={ACP_HOST_CONTEXT_KEY: host, AcpMetadataKey.RESUME_FROM_CHECKPOINT: True},
    )
    store.save(build_valid_checkpoint(run_id="run-r1"))
    _persistence, resume = resolve_session_persistence(
        request,
        run_id="run-r1",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=host.agent_checkpoint_store,
    )
    assert resume is not None


def test_r3_r5_r1_q11_q12_q13_q14_pluginability_through_host(tmp_path: Path) -> None:
    _resume_via_host_store(InMemoryAgentCheckpointStore())
    _resume_via_host_store(SQLiteAgentCheckpointStore(tmp_path / "r1-plugin.db"))


def test_r3_r5_r1_q15_graph_executor_uses_host_checkpoint_injection() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/nexus/execution/graph_executor.py").read_text(
        encoding="utf-8",
    )
    assert "inject_acp_checkpoint_metadata" in source
    wiring = (_REPO_ROOT / "intergrax/agents/persistence/checkpoint_wiring.py").read_text(
        encoding="utf-8",
    )
    assert "merge_host_checkpoint_store" in wiring
    assert 'wired[AcpMetadataKey.CHECKPOINT_STORE]' not in wiring


def test_r3_r5_r1_q16_nexus_host_propagation_via_harness() -> None:
    store = resolve_host_agent_checkpoint_store(agent_checkpoint_store=InMemoryAgentCheckpointStore())
    harness_src = (
        _REPO_ROOT / "intergrax/applications/_shared/acp_session_host_wiring.py"
    ).read_text(encoding="utf-8")
    assert "agent_checkpoint_store=runtime.agent_checkpoint_store" in harness_src
    host = build_acp_session_host_context(
        app_profile=ApplicationEnvironmentProfile.lab_defaults(),
        agent_checkpoint_store=store,
    )
    assert host.agent_checkpoint_store is store


def test_r3_r5_r1_q17_q18_checkpoint_store_metadata_writers_readers_inventoried() -> None:
    wiring = (_REPO_ROOT / "intergrax/agents/persistence/checkpoint_wiring.py").read_text(
        encoding="utf-8",
    )
    assert "CHECKPOINT_STORE" not in wiring
    session = inspect.getsource(resolve_session_persistence)
    assert "CHECKPOINT_STORE" not in session


def test_r3_r5_r1_q19_public_shadow_composition_paths_zero() -> None:
    acp_src = inspect.getsource(acp_run_module._run_acp_session_bound)
    assert "CHECKPOINT_STORE" not in acp_src


def test_r3_r5_r1_q20_r3_r5_regression_matrix_imported() -> None:
    from tests.qualification.state_x import _r3_r5_qualification_tests as r5  # noqa: F401

    assert hasattr(r5, "test_r3_r5_q15_resume_rejects_checkpoint_agent_mismatch")


def test_r3_r5_r1_q25_host_context_checkpoint_field_strongly_typed() -> None:
    host = ACPSessionHostContext(agent_checkpoint_store=InMemoryAgentCheckpointStore())
    assert isinstance(host.agent_checkpoint_store, AgentCheckpointStore)


def test_r3_r5_r1_q26_no_caller_controlled_trust_boolean() -> None:
    session_src = inspect.getsource(resolve_session_persistence)
    assert "trusted" not in session_src.lower()
    host_fields = ACPSessionHostContext.model_fields
    assert "trusted" not in host_fields


def test_r3_r5_r1_q27_production_ingress_host_enrichment() -> None:
    host_store = resolve_host_agent_checkpoint_store(agent_checkpoint_store=InMemoryAgentCheckpointStore())
    enricher = make_acp_checkpoint_task_enricher(host_store)
    assert enricher is not None
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="echo",
        message="hello",
        metadata={AcpMetadataKey.SESSION_ENABLED: True},
    )
    enriched = enricher(task)
    metadata: dict[str, Any] = dict(enriched.metadata)
    inject_acp_checkpoint_metadata(
        metadata,
        store=host_store,
        run_id=str(task.task_id),
        tenant_id=task.tenant_id,
    )
    host = metadata[ACP_HOST_CONTEXT_KEY]
    assert isinstance(host, ACPSessionHostContext)
    assert host.agent_checkpoint_store is host_store


def test_r3_r5_r1_q28_no_duplicate_composition_owner() -> None:
    source = inspect.getsource(resolve_session_persistence)
    assert source.count("checkpoint_store") >= 1
    assert "resolve_checkpoint_store" not in source


def test_r3_r5_r1_q29_all_implementations_classified() -> None:
    impls = {name for name, _ in agent_checkpoint_store_implementations()}
    assert impls == {"InMemoryAgentCheckpointStore", "SQLiteAgentCheckpointStore"}


def test_r3_r5_r1_q30_negative_ingress_malicious_metadata_ignored() -> None:
    _CallCountingStore.calls = 0
    malicious = _CallCountingStore()
    host_store = resolve_host_agent_checkpoint_store(agent_checkpoint_store=InMemoryAgentCheckpointStore())
    metadata: dict[str, Any] = {
        AcpMetadataKey.SESSION_ENABLED: True,
        AcpMetadataKey.CHECKPOINT_STORE: malicious,
        AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
    }
    inject_acp_checkpoint_metadata(
        metadata,
        store=host_store,
        run_id="run-ingress",
        tenant_id="tenant-a",
    )
    host = metadata[ACP_HOST_CONTEXT_KEY]
    assert isinstance(host, ACPSessionHostContext)
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        agent_id="agent-a",
        metadata=metadata,
    )
    resolve_session_persistence(
        request,
        run_id="run-ingress",
        tenant_id="tenant-a",
        agent_id="agent-a",
        checkpoint_store=host.agent_checkpoint_store,
    )
    assert _CallCountingStore.calls == 0


def test_r3_r5_r1_wire_acp_run_request_uses_host_context() -> None:
    store = InMemoryAgentCheckpointStore()
    request = wire_acp_run_request(
        AgentRunRequest(
            input="x",
            identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
            agent_id="agent-a",
        ),
        store,
    )
    host = request.metadata[ACP_HOST_CONTEXT_KEY]
    assert isinstance(host, ACPSessionHostContext)
    assert host.agent_checkpoint_store is store
    assert request.metadata.get(AcpMetadataKey.CHECKPOINT_STORE) is None


def test_r3_r5_r1_provider_replaceability_at_composition(tmp_path: Path) -> None:
    for _label, factory in parity_checkpoint_store_factories(tmp_path):
        store = factory()
        host = ACPSessionHostContext(agent_checkpoint_store=store)
        request = AgentRunRequest(
            input="x",
            identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
            agent_id="agent-a",
            metadata={ACP_HOST_CONTEXT_KEY: host},
        )
        persistence, _ = resolve_session_persistence(
            request,
            run_id="run-r5",
            tenant_id="tenant-a",
            agent_id="agent-a",
            checkpoint_store=host.agent_checkpoint_store,
        )
        assert persistence.checkpoint_store is store


def test_r3_r5_r1_acp_runtime_no_sqlite_import_in_session_persistence() -> None:
    source = inspect.getsource(resolve_session_persistence)
    assert "SQLiteAgentCheckpointStore" not in source
    assert "InMemoryAgentCheckpointStore" not in source
