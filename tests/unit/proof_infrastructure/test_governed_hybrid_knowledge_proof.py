# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.project_status.integration import ProjectStatusIntegration
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrationStatus,
)
from local_workspace_application.workspaces.hybrid_ask_models import EvidenceTypeV1
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.fixtures import (
    DEPLOYMENT_POLICY_CONTENT,
    PROOF_BINDING_ID,
    PROOF_CONNECTION_REF,
    PROOF_TENANT_ID,
    PROOF_WORKSPACE_ID,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.harness import build_harness
from proof_infrastructure.governed_hybrid_knowledge_proof.models import SemanticDecisionV1
from proof_infrastructure.governed_hybrid_knowledge_proof.runner import run_flagship_proof

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HARNESS_PATH = (
    _REPO_ROOT / "proof_infrastructure/governed_hybrid_knowledge_proof/harness.py"
)


def _harness_source() -> str:
    return _HARNESS_PATH.read_text(encoding="utf-8")


def test_flagship_harness_has_no_demo_plumbing() -> None:
    source = _harness_source()
    forbidden = (
        "_SearchTaskResult",
        "_SearchTaskExecutor",
        "create_project_status_integration(",
        "task_run_bridge.new_run_id =",
        "hasattr(",
        "getattr(",
    )
    for token in forbidden:
        assert token not in source


@pytest.mark.asyncio
async def test_flagship_harness_real_indexed_and_connection_boundaries(
    project_status_server,
) -> None:
    from proof_infrastructure.controlled_project_status_service.lifecycle import (
        ControlledProjectStatusServer,
    )

    server: ControlledProjectStatusServer = project_status_server
    harness = await build_harness(server=server)

    assert harness.rehydration_status == TenantConnectionRehydrationStatus.REGISTERED.value
    integration = harness.connection_registry.resolve(
        tenant_id=PROOF_TENANT_ID,
        connection_ref=PROOF_CONNECTION_REF,
        provider_id=PROJECT_STATUS_PROVIDER_ID,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
    )
    assert isinstance(integration, ProjectStatusIntegration)
    assert integration.config.base_url == server.base_url

    connection = harness.tenant_connection_repository.get(
        tenant_id=PROOF_TENANT_ID,
        connection_ref=PROOF_CONNECTION_REF,
    )
    assert connection is not None
    assert connection.validated_secret_free_config["base_url"] == server.base_url

    assert harness.indexed_stack.search_executions == 0
    command = harness.build_command(run_id="boundary-indexed-run")
    run = await harness.service.ask(command)
    assert harness.indexed_stack.search_executions >= 1
    indexed_evidence = next(
        item
        for item in run.persisted_evidence
        if item.evidence_type is EvidenceTypeV1.INDEXED
    )
    assert indexed_evidence.document_id == harness.indexed_stack.indexed_document_id
    indexed_citation = next(
        citation
        for citation in run.citations
        if citation.evidence_type is EvidenceTypeV1.INDEXED
    )
    assert DEPLOYMENT_POLICY_CONTENT.splitlines()[0] in indexed_citation.excerpt


@pytest.mark.asyncio
async def test_flagship_revoke_uses_canonical_disable_mutation(project_status_server) -> None:
    server = project_status_server
    harness = await build_harness(server=server, revoke_after_indexed=True)
    configuration_before = harness.configuration_service.get_configuration(
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
    )
    assert configuration_before is not None
    revision_before = configuration_before.configuration_revision

    await harness.service.ask(harness.build_command(run_id="boundary-revoke-run"))

    assert harness.indexed_retriever.configuration_revision_before_disable == revision_before
    assert harness.indexed_retriever.configuration_revision_after_disable == revision_before + 1

    configuration_after = harness.configuration_service.get_configuration(
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
    )
    assert configuration_after is not None
    binding = next(
        item
        for item in configuration_after.live_access_bindings
        if item.live_access_binding_id == PROOF_BINDING_ID
    )
    assert binding.status is LiveAccessBindingStatusV1.DISABLED


def test_flagship_proof_all_scenarios_pass() -> None:
    result = run_flagship_proof(emit_terminal=False)

    assert result.passed_count == 4
    assert result.all_passed is True
    assert result.overall_status == "PASS"
    assert result.scenario_1.observed == SemanticDecisionV1.NO
    assert result.scenario_1.http_read_count == 1
    assert result.scenario_2.observed == SemanticDecisionV1.YES
    assert result.scenario_2.http_read_count == 1
    assert result.scenario_3.http_read_count == 0
    assert result.scenario_3.llm_call_count == 0
    assert result.scenario_3.observed == SemanticDecisionV1.CANNOT_DETERMINE
    assert result.scenario_4.observed == SemanticDecisionV1.NO
    assert result.scenario_1_run_id is not None


def test_flagship_proof_repeatable_twice() -> None:
    first = run_flagship_proof(emit_terminal=False)
    second = run_flagship_proof(emit_terminal=False)
    assert first.all_passed is True
    assert second.all_passed is True


def test_flagship_proof_cli_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "proof_infrastructure.governed_hybrid_knowledge_proof",
            "--json",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    json_start = completed.stdout.find("{")
    assert json_start >= 0
    payload = json.loads(completed.stdout[json_start:])
    assert payload["overall_status"] == "PASS"
    assert payload["scenario_1"]["passed"] is True
    assert payload["scenario_4"]["passed"] is True


@pytest.fixture
def project_status_server():
    from proof_infrastructure.controlled_project_status_service.lifecycle import (
        ControlledProjectStatusServer,
    )

    server = ControlledProjectStatusServer.start()
    yield server
    server.stop()
