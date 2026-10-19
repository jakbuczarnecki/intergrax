# © Artur Czarnecki. All rights reserved.

"""IMPL-01-R3 — AC3 provenance, provider capabilities, typed durability (behavioral gates)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from echo.echo_agent import EchoAgent
from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from governed_contractor_application.host.governed_external_work_host_stores import (
    GovernedExternalWorkHostStoreBundle,
    validate_production_governed_external_work_host_store_bundle,
)
from governed_contractor_application.host.resolve_strict_external_work_provider_capabilities import (
    StrictExternalWorkProviderResolutionError,
    resolve_strict_external_work_provider_capabilities,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.stores import (
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
)
from governed_contractor_application.host.strict_external_work_authoritative_factory_injection import (
    StrictExternalWorkHostAgentInjection,
    build_strict_external_work_authoritative_factory_wrapper,
)
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.applications._shared.host_overriding_runtime_agent_factory_resolver import (
    HostOverridingRuntimeAgentFactoryResolver,
    HostRuntimeAgentFactoryOverride,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolutionError,
    _factory_key,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.external_work import (
    ExternalContractorIdentity,
    ExternalWorkCapability,
    ExternalWorkProviderDescriptor,
)
from intergrax.contracts.external_work_provider_capability_projection import (
    project_descriptor_mapped_provider_capabilities,
)
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_RESOLVE_CAPS = (
    _REPO
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "resolve_strict_external_work_provider_capabilities.py"
)
_HOST_STORES = (
    _REPO
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "governed_external_work_host_stores.py"
)


_DIGEST = "sha256:" + ("a" * 64)
_ARTIFACT = "sha256:" + ("d" * 64)


def _minimal_revision() -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id="rev-r3",
        application_id="app_r3",
        application_environment_id="env-r3",
        application_release_id="rel-r3",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-r3",
        installed_agent_package_digests=(_DIGEST,),
        materialized_runtime_lock_id="lock-r3",
        materialized_runtime_lock_digest="sha256:" + ("b" * 64),
        runtime_graph_digest="sha256:" + ("c" * 64),
        materialization_artifact_digest=_ARTIFACT,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=RuntimeRevisionState.VALIDATED,
    )


def _artifact_external_contractor_factory(
    *,
    instances: list[ExternalContractorAdapterAgent] | None = None,
):
    def _factory(_ctx: ApplicationBuildContext[Any], _binding: AgentBinding):
        agent = ExternalContractorAdapterAgent()
        if instances is not None:
            instances.append(agent)
        return agent

    return _factory


@dataclass
class _RecordingDelegate:
    factory: Any
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail: bool = False

    def resolve_factory(self, **kwargs: Any):
        self.calls.append(dict(kwargs))
        if self.fail:
            raise RuntimeAgentFactoryResolutionError("delegate resolution failed")
        return self.factory


def test_g_r3_03_wrapped_materialization_invokes_delegate_and_factory_once() -> None:
    authoritative_instances: list[ExternalContractorAdapterAgent] = []
    factory = _artifact_external_contractor_factory(instances=authoritative_instances)
    factory_calls = 0

    def _counting_factory(ctx, binding):
        nonlocal factory_calls
        factory_calls += 1
        return factory(ctx, binding)

    delegate = _RecordingDelegate(factory=_counting_factory)
    ac3_key = _factory_key(_DIGEST, AgentBindingFactoryReference(builder_key="ac3"))
    injection = StrictExternalWorkHostAgentInjection(
        external_work=MagicMock(spec=ExternalWorkIntegration),
        authorization_boundary=None,
        sanctioned_operation_port=MagicMock(),
    )
    resolver = HostOverridingRuntimeAgentFactoryResolver(
        delegate,
        override=HostRuntimeAgentFactoryOverride(
            factory_keys=frozenset({ac3_key}),
            wrap_authoritative_factory=build_strict_external_work_authoritative_factory_wrapper(
                injection,
            ),
        ),
    )
    wrapped = resolver.resolve_factory(
        runtime_revision=_minimal_revision(),
        package_digest=_DIGEST,
        factory_reference=AgentBindingFactoryReference(builder_key="ac3"),
    )
    manifest = ApplicationManifest.lab(app_id="app_r3", name="R3", agents=[])
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = AgentBinding.mount(
        ExternalContractorAdapterAgent,
        contract_id="external_contractor_adapter",
    )
    agent = wrapped(ctx, binding)
    assert len(delegate.calls) == 1
    assert factory_calls == 1
    assert len(authoritative_instances) == 1
    assert agent is authoritative_instances[0]


def test_g_r3_01_ac3_resolver_invokes_delegate_before_wrap() -> None:
    delegate = _RecordingDelegate(factory=_artifact_external_contractor_factory())
    ac3_key = _factory_key(_DIGEST, AgentBindingFactoryReference(builder_key="ac3"))
    injection = StrictExternalWorkHostAgentInjection(
        external_work=MagicMock(spec=ExternalWorkIntegration),
        authorization_boundary=None,
        sanctioned_operation_port=MagicMock(),
        require_sanctioned_external_work_mutations=True,
    )
    resolver = HostOverridingRuntimeAgentFactoryResolver(
        delegate,
        override=HostRuntimeAgentFactoryOverride(
            factory_keys=frozenset({ac3_key}),
            wrap_authoritative_factory=build_strict_external_work_authoritative_factory_wrapper(
                injection,
            ),
        ),
    )
    revision = _minimal_revision()
    ref_ac3 = AgentBindingFactoryReference(builder_key="ac3")
    ref_other = AgentBindingFactoryReference(builder_key="echo")

    wrapped = resolver.resolve_factory(
        runtime_revision=revision,
        package_digest=_DIGEST,
        factory_reference=ref_ac3,
    )
    plain = resolver.resolve_factory(
        runtime_revision=revision,
        package_digest=_DIGEST,
        factory_reference=ref_other,
    )

    assert len(delegate.calls) == 2
    assert wrapped is not delegate.factory
    assert plain is delegate.factory


def test_n_r3_01_delegate_failure_fail_closed() -> None:
    delegate = _RecordingDelegate(
        factory=_artifact_external_contractor_factory(),
        fail=True,
    )
    ac3_key = _factory_key(_DIGEST, AgentBindingFactoryReference(builder_key="ac3"))
    injection = StrictExternalWorkHostAgentInjection(
        external_work=MagicMock(spec=ExternalWorkIntegration),
        authorization_boundary=None,
        sanctioned_operation_port=MagicMock(),
    )
    resolver = HostOverridingRuntimeAgentFactoryResolver(
        delegate,
        override=HostRuntimeAgentFactoryOverride(
            factory_keys=frozenset({ac3_key}),
            wrap_authoritative_factory=build_strict_external_work_authoritative_factory_wrapper(
                injection,
            ),
        ),
    )
    with pytest.raises(RuntimeAgentFactoryResolutionError):
        resolver.resolve_factory(
            runtime_revision=_minimal_revision(),
            package_digest=_DIGEST,
            factory_reference=AgentBindingFactoryReference(builder_key="ac3"),
        )


def test_n_r3_02_wrong_agent_type_rejected_at_materialization() -> None:
    def _echo_factory(_ctx, _binding):
        return EchoAgent()

    delegate = _RecordingDelegate(factory=_echo_factory)
    ac3_key = _factory_key(_DIGEST, AgentBindingFactoryReference(builder_key="ac3"))
    injection = StrictExternalWorkHostAgentInjection(
        external_work=MagicMock(spec=ExternalWorkIntegration),
        authorization_boundary=None,
        sanctioned_operation_port=MagicMock(),
    )
    resolver = HostOverridingRuntimeAgentFactoryResolver(
        delegate,
        override=HostRuntimeAgentFactoryOverride(
            factory_keys=frozenset({ac3_key}),
            wrap_authoritative_factory=build_strict_external_work_authoritative_factory_wrapper(
                injection,
            ),
        ),
    )
    wrapped = resolver.resolve_factory(
        runtime_revision=_minimal_revision(),
        package_digest=_DIGEST,
        factory_reference=AgentBindingFactoryReference(builder_key="ac3"),
    )
    manifest = ApplicationManifest.lab(app_id="app_r3", name="R3", agents=[])
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = AgentBinding.mount(
        ExternalContractorAdapterAgent,
        contract_id="external_contractor_adapter",
    )
    with pytest.raises(RuntimeAgentFactoryResolutionError):
        wrapped(ctx, binding)


def test_g_r3_02_injection_applies_sanctioned_port_and_require_flag() -> None:
    port = MagicMock()
    integration = MagicMock(spec=ExternalWorkIntegration)
    authoritative_instances: list[ExternalContractorAdapterAgent] = []
    delegate = _RecordingDelegate(
        factory=_artifact_external_contractor_factory(instances=authoritative_instances),
    )
    ac3_key = _factory_key(_DIGEST, AgentBindingFactoryReference(builder_key="ac3"))
    injection = StrictExternalWorkHostAgentInjection(
        external_work=integration,
        authorization_boundary=None,
        sanctioned_operation_port=port,
    )
    resolver = HostOverridingRuntimeAgentFactoryResolver(
        delegate,
        override=HostRuntimeAgentFactoryOverride(
            factory_keys=frozenset({ac3_key}),
            wrap_authoritative_factory=build_strict_external_work_authoritative_factory_wrapper(
                injection,
            ),
        ),
    )
    wrapped = resolver.resolve_factory(
        runtime_revision=_minimal_revision(),
        package_digest=_DIGEST,
        factory_reference=AgentBindingFactoryReference(builder_key="ac3"),
    )
    manifest = ApplicationManifest.lab(app_id="app_r3", name="R3", agents=[])
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = AgentBinding.mount(
        ExternalContractorAdapterAgent,
        contract_id="external_contractor_adapter",
    )
    agent = wrapped(ctx, binding)
    assert len(authoritative_instances) == 1
    assert agent is authoritative_instances[0]
    assert agent._sanctioned_operation_port is port
    assert agent._require_sanctioned_external_work_mutations is True
    assert agent._external_work is integration


def test_missing_sanctioned_port_rejected_at_injection_build() -> None:
    runtime = GovernedContractorHostRuntimeComposition(
        external_work_integration=MagicMock(spec=ExternalWorkIntegration),
        require_sanctioned_external_work_mutations=True,
        sanctioned_external_work_operation_port=None,
    )
    from governed_contractor_application.host.strict_external_work_authoritative_factory_injection import (
        strict_external_work_host_agent_injection_from_runtime,
    )

    with pytest.raises(ValueError, match="sanctioned_external_work_operation_port"):
        strict_external_work_host_agent_injection_from_runtime(runtime)


def test_descriptor_mapped_capabilities_exclude_integration_invariants() -> None:
    descriptor = ExternalWorkProviderDescriptor(
        identity=ExternalContractorIdentity(
            provider_id="p1",
            contractor_id="c1",
            protocol_id="intergrax.external_work.v1",
        ),
        capabilities=(ExternalWorkCapability.QUOTE_FIRST,),
        protocol_id="intergrax.external_work.v1",
        descriptor_digest="sha256:" + ("e" * 64),
        schema_id="external_work_provider_descriptor.v1",
    )
    mapped = project_descriptor_mapped_provider_capabilities(descriptor)
    assert mapped.supports_quote is True
    assert mapped.supports_create is False
    assert mapped.supports_idempotency is False


def test_n_r3_03_pin_mismatch_denied() -> None:
    descriptor = ExternalWorkProviderDescriptor(
        identity=ExternalContractorIdentity(
            provider_id="expected",
            contractor_id="c1",
            protocol_id="intergrax.external_work.v1",
        ),
        capabilities=(ExternalWorkCapability.QUOTE_FIRST,),
        protocol_id="intergrax.external_work.v1",
        descriptor_digest="sha256:" + ("f" * 64),
        schema_id="external_work_provider_descriptor.v1",
    )
    integration = MagicMock(spec=ExternalWorkIntegration)
    integration.discover.return_value = descriptor
    settings = GovernedContractorBackendSettings(
        strict_external_work_provider_id="other",
        strict_external_work_provider_descriptor_digest="sha256:" + ("f" * 64),
    )
    with pytest.raises(StrictExternalWorkProviderResolutionError, match="provider_id pin"):
        resolve_strict_external_work_provider_capabilities(integration, settings=settings)


def test_n_r3_06_no_unconditional_capability_literals_in_strict_resolver() -> None:
    text = _RESOLVE_CAPS.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "supports_create":
            if isinstance(node.value, ast.Constant) and node.value.value is True:
                pytest.fail("unconditional supports_create=True in strict resolver")
        if isinstance(node, ast.keyword) and node.arg == "supports_idempotency":
            if isinstance(node.value, ast.Constant) and node.value.value is True:
                pytest.fail("unconditional supports_idempotency=True in strict resolver")


def test_n_r3_05_non_durable_bundle_rejected() -> None:
    bundle = GovernedExternalWorkHostStoreBundle(
        execution_store=InMemoryGovernedExecutionStore(),
        receipt_store=InMemoryProofReceiptStore(),
        bundle_store=InMemoryPolicyBundleArtifactStore(),
        continuation_store=InMemoryContinuationStateStore(),
    )
    assert bundle.is_durable is False
    with pytest.raises(ValueError, match="durable governed external work host stores"):
        validate_production_governed_external_work_host_store_bundle(bundle)


def test_n_r3_07_no_getattr_durability_probing_in_bundle() -> None:
    text = _HOST_STORES.read_text(encoding="utf-8")
    assert "getattr" not in text


class _DurableMember:
    @property
    def is_durable(self) -> bool:
        return True


def test_g_r3_05_all_durable_members_pass_validate() -> None:
    bundle = GovernedExternalWorkHostStoreBundle(
        execution_store=_DurableMember(),
        receipt_store=_DurableMember(),
        bundle_store=_DurableMember(),
        continuation_store=_DurableMember(),
    )
    assert bundle.is_durable is True
    validate_production_governed_external_work_host_store_bundle(bundle)
