# © Artur Czarnecki. All rights reserved.

"""Stage 15 — canonical agent lifecycle E2E proof."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import (
    BindAgentRequest,
    InstallAgentRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.agent_manager_models import AgentManagerDerivedStatus
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.catalog import CatalogProviderKind
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.core.qualification import QualificationStatus
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.canonical_agent_lifecycle_composition import (
    CanonicalAgentLifecycleProofStack,
    default_stage15_proof_config,
)
from testing_support.canonical_lifecycle_ping_agent import CANONICAL_PING_CAPABILITY
from tests.unit.agent_distribution.test_agent_platform_admin_service import admin_test_principal

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


_SECONDARY_LOGICAL_ID = "canonical-shadow-agent"
_SECONDARY_SLOT = "slot-canonical-shadow"
_SECONDARY_BINDING = "bind-canonical-shadow"
_SECONDARY_INSTALLATION = "inst-canonical-shadow"
_SECONDARY_META = "meta://canonical-shadow"
_SECONDARY_FACTORY = AgentBindingFactoryReference(
    factory_path="example_agent.shadow_factory.build_agent",
)


def _install_secondary_desired_state(stack: CanonicalAgentLifecycleProofStack) -> None:
    principal = admin_test_principal()
    stack.admin.install_agent(
        application_id=stack.config.application_id,
        application_environment_id=stack.config.environment_id,
        request=InstallAgentRequest(
            mutation_id="mut-stage15-secondary-install",
            installation_id=_SECONDARY_INSTALLATION,
            installation_slot_id=_SECONDARY_SLOT,
            package_identity=AgentPackageIdentity(
                distribution_package_id="intergrax-canonical-shadow-agent",
                package_version="1.0.0",
                package_digest=stack.config.package_digest,
            ),
            artifact_store_ref=f"store://artifacts/{_SECONDARY_INSTALLATION}",
            trust_record=AgentInstallationTrustRecord(
                qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
                package_digest=stack.config.package_digest,
                publisher_identity_ref="publisher:stage15",
                source_provider_id=stack.config.catalog_source_id,
                trust_evidence_refs=(
                    AgentTrustEvidenceRef(
                        evidence_id="evidence:stage15-shadow",
                        kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                    ),
                ),
            ),
            agent_project_metadata_ref=_SECONDARY_META,
        ),
        principal=principal,
    )
    stack.admin.bind_agent(
        application_id=stack.config.application_id,
        application_environment_id=stack.config.environment_id,
        request=BindAgentRequest(
            mutation_id="mut-stage15-secondary-bind",
            application_binding_id=_SECONDARY_BINDING,
            logical_agent_id=_SECONDARY_LOGICAL_ID,
            installation_slot_id=_SECONDARY_SLOT,
            factory_reference=_SECONDARY_FACTORY,
            enablement=True,
        ),
        principal=principal,
    )
    binding = next(
        item
        for item in stack.admin.list_bindings(
            application_id=stack.config.application_id,
            application_environment_id=stack.config.environment_id,
        ).bindings
        if item.application_binding_id == _SECONDARY_BINDING
    )
    stack.admin.enable_binding(
        application_id=stack.config.application_id,
        application_environment_id=stack.config.environment_id,
        application_binding_id=_SECONDARY_BINDING,
        request=SetAgentEnablementRequest(
            mutation_id="mut-stage15-secondary-enable",
            expected_revision=binding.binding_revision,
        ),
        principal=principal,
    )


@pytest.mark.parametrize(
    "catalog_provider_kind,catalog_source_id",
    [
        (CatalogProviderKind.BUILTIN, "builtin-stage15"),
        (CatalogProviderKind.ENTERPRISE_PRIVATE, "enterprise-private-stage15"),
    ],
)
def test_canonical_agent_lifecycle_reaches_serving_and_execution(
    tmp_path: Path,
    catalog_provider_kind: CatalogProviderKind,
    catalog_source_id: str,
) -> None:
    config = default_stage15_proof_config(
        catalog_provider_kind=catalog_provider_kind,
        catalog_source_id=catalog_source_id,
    )
    stack = CanonicalAgentLifecycleProofStack.build(tmp_path, config)
    catalog_entry = stack.discover_catalog_entry()
    assert catalog_entry.catalog_source.provider_kind is catalog_provider_kind

    proof = stack.run_happy_path()
    assert proof.catalog_source_id == catalog_source_id
    assert proof.distribution_package_id == config.distribution_package_id
    assert proof.package_digest == config.package_digest
    assert proof.traffic_serving_revision_id == config.revision_id
    assert proof.execution_agent_id == config.logical_agent_id
    assert proof.execution_answer == config.expected_output

    manager_entry = next(
        item
        for item in stack.agent_manager_query.list_agents(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
        ).items
        if item.lifecycle.logical_agent_id == config.logical_agent_id
    )
    assert manager_entry.lifecycle.installed is True
    assert manager_entry.lifecycle.bound is True
    assert manager_entry.lifecycle.enabled_in_desired_state is True
    assert manager_entry.runtime.included_in_active_revision is True
    assert manager_entry.runtime.serving is True
    assert manager_entry.runtime.traffic_serving_revision_id == config.revision_id
    assert manager_entry.derived_status is AgentManagerDerivedStatus.SERVING


def test_desired_state_does_not_become_serving_before_activation(tmp_path: Path) -> None:
    stack = CanonicalAgentLifecycleProofStack.build(tmp_path)
    stack.install_from_catalog()
    stack.bind_enabled_agent()
    built = stack.build_revision()

    entry = next(
        item
        for item in stack.agent_manager_query.list_agents(
            application_id=stack.config.application_id,
            application_environment_id=stack.config.environment_id,
        ).items
        if item.lifecycle.logical_agent_id == stack.config.logical_agent_id
    )
    assert entry.lifecycle.enabled_in_desired_state is True
    assert entry.runtime.serving is False
    assert entry.runtime.included_in_candidate_revision is True
    assert entry.derived_status is AgentManagerDerivedStatus.READY_FOR_REVISION

    stack.register_projection_and_activate(built)
    serving_entry = next(
        item
        for item in stack.agent_manager_query.list_agents(
            application_id=stack.config.application_id,
            application_environment_id=stack.config.environment_id,
        ).items
        if item.lifecycle.logical_agent_id == stack.config.logical_agent_id
    )
    assert serving_entry.runtime.serving is True
    assert serving_entry.derived_status is AgentManagerDerivedStatus.SERVING


def test_serving_runtime_cannot_execute_agent_outside_serving_projection(
    tmp_path: Path,
) -> None:
    stack = CanonicalAgentLifecycleProofStack.build(tmp_path)
    stack.install_from_catalog()
    stack.bind_enabled_agent()
    built = stack.build_revision()
    stack.register_projection_and_activate(built)

    _install_secondary_desired_state(stack)
    registry = stack.resolve_registry_read()
    assert isinstance(registry, AgentRegistryRead)
    assert registry.has(stack.config.logical_agent_id)
    assert not registry.has(_SECONDARY_LOGICAL_ID)

    projection = stack.resolve_serving_projection()
    host_runtime = build_harness_host_runtime(
        stack.manifest,
        stack.environment,
        registry_projection=projection,
        trace_db_path=tmp_path / "secondary-trace.db",
        runtime_events_db_path=tmp_path / "secondary-runtime_events.db",
        document_store=InMemoryDocumentStore(),
    )
    task = Task(
        tenant_id="tenant-test",
        user_id="proof-user",
        message="ping",
        agent_id=_SECONDARY_LOGICAL_ID,
        context=TaskContext(capability=CANONICAL_PING_CAPABILITY),
    )
    with pytest.raises(KeyError, match="not registered"):
        asyncio.run(host_runtime.execution.execute(task))


def test_historical_revision_is_not_rewritten_by_later_desired_state(
    tmp_path: Path,
) -> None:
    stack = CanonicalAgentLifecycleProofStack.build(tmp_path)
    stack.install_from_catalog()
    stack.bind_enabled_agent()
    built = stack.build_revision()
    stack.register_projection_and_activate(built)

    historical_projection = stack.resolve_projection_for_revision(built.runtime_revision_id)
    historical_entry = next(iter(historical_projection.agent_registry.list_agent_ids()))
    assert historical_entry == stack.config.logical_agent_id

    stack.admin.disable_binding(
        application_id=stack.config.application_id,
        application_environment_id=stack.config.environment_id,
        application_binding_id=stack.config.application_binding_id,
        request=SetAgentEnablementRequest(
            mutation_id="mut-stage15-disable",
            expected_revision=stack.admin.list_bindings(
                application_id=stack.config.application_id,
                application_environment_id=stack.config.environment_id,
            ).bindings[0].binding_revision,
        ),
        principal=admin_test_principal(),
    )
    current_roster = stack.admin.inspect_effective_roster(
        application_id=stack.config.application_id,
        application_environment_id=stack.config.environment_id,
    )
    current_entry = next(
        item
        for item in current_roster.entries
        if item.logical_agent_id == stack.config.logical_agent_id
    )
    assert current_entry.effective_enablement is False

    frozen_projection = stack.resolve_projection_for_revision(built.runtime_revision_id)
    assert frozen_projection.evidence.runtime_revision_id == built.runtime_revision_id
    assert frozen_projection.agent_registry.has(stack.config.logical_agent_id)
    assert frozen_projection.evidence.effective_roster_revision_id == (
        historical_projection.evidence.effective_roster_revision_id
    )
