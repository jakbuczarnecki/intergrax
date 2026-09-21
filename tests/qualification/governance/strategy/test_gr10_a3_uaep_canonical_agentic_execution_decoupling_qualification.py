# © Artur Czarnecki. All rights reserved.

"""GR-10-A3 — checkpoint persistence decoupled from acp.session.v1 execution branch."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agents.persistence.checkpoint_store import InMemoryAgentCheckpointStore
from intergrax.applications._shared.acp_checkpoint_task_enricher import make_acp_checkpoint_task_enricher
from intergrax.applications._shared.task_control_wiring import build_reliability_task_enricher
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.nexus.agents.runtime_request_bridge import acp_session_enabled
from intergrax.runtime.task.task import Task
from tests.qualification.governance.strategy.catalog import (
    GR10_A2_CHECKPOINT_SESSION_COUPLING,
    GR10_AGENTIC_LEGAL_PRODUCTION_PATHS,
    Gr10CheckpointSessionCouplingStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENRICHER = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "acp_checkpoint_task_enricher.py"
)


def _agent_engine_branch(metadata: dict[str, object]) -> str:
    """Mirror AgentEngine._execute_agent_impl gate: acp.session.v1 vs UAEP."""
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="legal",
        message="review",
        metadata=dict(metadata),
    )
    request = task.to_runtime_request(run_id=mint_run_id())
    return "ACP" if acp_session_enabled(request) else "UAEP"


@pytest.mark.parametrize(
    ("checkpoint", "session_enabled", "expected"),
    [
        (False, False, "UAEP"),
        (True, False, "UAEP"),
        (False, True, "ACP"),
        (True, True, "ACP"),
    ],
)
def test_gr10_a3_execution_path_matrix(
    checkpoint: bool,
    session_enabled: bool,
    expected: str,
) -> None:
    metadata: dict[str, object] = {}
    if session_enabled:
        metadata[AcpMetadataKey.SESSION_ENABLED] = True
    if checkpoint:
        store = InMemoryAgentCheckpointStore()
        enricher = make_acp_checkpoint_task_enricher(store)
        task = Task(
            task_id=mint_task_id(),
            tenant_id="tenant-a",
            user_id="user-1",
            agent_id="legal",
            message="review",
            metadata=metadata,
        )
        metadata = dict(enricher(task).metadata)
        if session_enabled:
            assert metadata.get(AcpMetadataKey.CHECKPOINT_STORE) is store
        else:
            assert metadata.get(AcpMetadataKey.CHECKPOINT_STORE) is None
    assert _agent_engine_branch(metadata) == expected


def test_gr10_a3_checkpoint_enricher_wires_store_without_session_flag() -> None:
    store = InMemoryAgentCheckpointStore()
    enricher = make_acp_checkpoint_task_enricher(store)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="echo",
        message="hello",
        metadata={},
    )
    enriched = enricher(task)
    assert enriched.metadata.get(AcpMetadataKey.CHECKPOINT_STORE) is None
    assert enriched.metadata.get(AcpMetadataKey.SESSION_ENABLED) is None


def test_gr10_a3_checkpoint_enricher_wires_store_when_session_explicit() -> None:
    store = InMemoryAgentCheckpointStore()
    enricher = make_acp_checkpoint_task_enricher(store)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="echo",
        message="hello",
        metadata={AcpMetadataKey.SESSION_ENABLED: True},
    )
    enriched = enricher(task)
    assert enriched.metadata.get(AcpMetadataKey.CHECKPOINT_STORE) is store


def test_gr10_a3_reliability_enricher_checkpoint_decoupled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    store = InMemoryAgentCheckpointStore()
    enricher = build_reliability_task_enricher(env, agent_checkpoint_store=store)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="legal",
        message="review",
        metadata={},
    )
    enriched = enricher(task)
    assert enriched.metadata.get(AcpMetadataKey.CHECKPOINT_STORE) is None
    assert enriched.metadata.get(AcpMetadataKey.SESSION_ENABLED) is None


def test_gr10_a3_explicit_acp_metadata_still_selects_acp_branch() -> None:
    metadata = {AcpMetadataKey.SESSION_ENABLED: True}
    assert _agent_engine_branch(metadata) == "ACP"


def test_gr10_a3_catalog_acp_path_explicit_non_canonical() -> None:
    acp = next(row for row in GR10_AGENTIC_LEGAL_PRODUCTION_PATHS if row.path_id == "P-ACP-SESSION")
    assert acp.status == "EXPLICIT_NON_CANONICAL"


def test_gr10_a3_ssot_checkpoint_coupling_decoupled() -> None:
    assert GR10_A2_CHECKPOINT_SESSION_COUPLING is (
        Gr10CheckpointSessionCouplingStatus.DEPRECATED_MIGRATION_TARGET
    )
    source = _ENRICHER.read_text(encoding="utf-8-sig")
    assert "metadata[AcpMetadataKey.SESSION_ENABLED] = True" not in source
    assert "SESSION_ENABLED] = True" not in source
