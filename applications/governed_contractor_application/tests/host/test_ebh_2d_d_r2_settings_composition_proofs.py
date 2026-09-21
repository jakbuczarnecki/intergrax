# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D-R2 — replaceability and settings/composition independence proofs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.agent_builders import (
    build_external_contractor_adapter_factory,
    build_governed_contractor_agent_builders,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
    compose_governed_contractor_host_runtime,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.execution_identity import mint_task_id
from tests.unit.runtime.governance.gr3_test_support import StaticActiveTaskScope

pytestmark = pytest.mark.unit


def _policy_bundle_from_gr6() -> object:
    from applications.governed_contractor_application.tests.host.test_gr6_wire_production_decision_governance import (
        _test_policy_bundle,
    )

    return _test_policy_bundle()


def _cw_repositories() -> object:
    from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
        gr6_seeded_collaborative_work_repositories,
    )

    return gr6_seeded_collaborative_work_repositories(
        tenant_id="tenant-r2",
        workspace_id="workspace-r2",
        principal_id="principal-r2",
    )


def test_same_settings_two_runtime_compositions_without_settings_mutation() -> None:
    settings = replace(
        GovernedContractorBackendSettings.from_env(),
        runtime_policy_bundle=_policy_bundle_from_gr6(),  # type: ignore[arg-type]
    )
    settings_id = id(settings)
    fake_a = DeterministicExternalWorkFake()
    fake_b = DeterministicExternalWorkFake()
    cw = _cw_repositories()
    task_scope = StaticActiveTaskScope(mint_task_id())
    runtime_a = compose_governed_contractor_host_runtime(
        settings,
        integration=fake_a,
        task_scope=task_scope,
        collaborative_work_repositories=cw,  # type: ignore[arg-type]
    )
    runtime_b = compose_governed_contractor_host_runtime(
        settings,
        integration=fake_b,
        task_scope=task_scope,
        collaborative_work_repositories=cw,  # type: ignore[arg-type]
    )
    assert id(settings) == settings_id
    assert runtime_a.external_work_integration is fake_a
    assert runtime_b.external_work_integration is fake_b
    assert runtime_a is not runtime_b


def test_configured_factory_swaps_external_work_without_changing_settings() -> None:
    manifest = ApplicationManifest.lab(app_id="r2", name="R2", agents=[])
    settings = GovernedContractorBackendSettings.from_env()
    ctx = ApplicationBuildContext.for_manifest(manifest, settings=settings)
    binding = AgentBinding.mount(ExternalContractorAdapterAgent, contract_id="external_contractor_adapter")
    fake_a = DeterministicExternalWorkFake()
    fake_b = DeterministicExternalWorkFake()
    agent_a = build_external_contractor_adapter_factory(
        external_work=fake_a,
        authorization_boundary=None,
    )(ctx, binding)
    agent_b = build_external_contractor_adapter_factory(
        external_work=fake_b,
        authorization_boundary=None,
    )(ctx, binding)
    assert agent_a._external_work is fake_a
    assert agent_b._external_work is fake_b
    assert ctx.settings is settings


def test_agent_builders_derived_from_runtime_composition() -> None:
    fake = DeterministicExternalWorkFake()
    runtime = GovernedContractorHostRuntimeComposition(external_work_integration=fake)
    builders = build_governed_contractor_agent_builders(runtime)
    manifest = ApplicationManifest.lab(app_id="r2b", name="R2B", agents=[])
    ctx = ApplicationBuildContext.for_manifest(manifest, settings=GovernedContractorBackendSettings.from_env())
    binding = AgentBinding.mount(ExternalContractorAdapterAgent, contract_id="external_contractor_adapter")
    agent = builders[ExternalContractorAdapterAgent](ctx, binding)
    assert agent._external_work is fake
