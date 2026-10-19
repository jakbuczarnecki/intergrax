# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q — materialization-boundary gates for strict external-work roster."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from echo.echo_agent import EchoAgent
from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from governed_contractor_application.host.strict_sanctioned_external_work_registry_projection import (
    reproject_registry_with_strict_sanctioned_external_work,
)
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.applications._shared.registry_projection import (
    RegistryProjectionError,
    RegistryProjectionInputBundle,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    InMemoryRuntimeAgentFactoryResolver,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _APP,
    _DIGEST_A,
    _EXTERNAL_CONTRACTOR_FACTORY_REF,
    _FACTORY_REF,
    _ac3_revision_bound_strict_external_work_host_manifest,
    _build_echo_artifact,
    _revision,
    _roster,
    _roster_with_strict_external_work,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_ROSTER_ERROR = "revision-bound 'external_contractor_adapter' roster entry"


def _host_runtime() -> GovernedContractorHostRuntimeComposition:
    return GovernedContractorHostRuntimeComposition(
        external_work_integration=MagicMock(spec=ExternalWorkIntegration),
        sanctioned_external_work_operation_port=MagicMock(),
        require_sanctioned_external_work_mutations=True,
    )


def _register_factories(resolver: InMemoryRuntimeAgentFactoryResolver) -> None:
    def _echo(
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> EchoAgent:
        return EchoAgent()

    def _external(
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> ExternalContractorAdapterAgent:
        return ExternalContractorAdapterAgent()

    _echo_factory: AgentFactory = _echo
    _external_factory: AgentFactory = _external

    resolver.register(
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
        factory=_echo_factory,
    )
    resolver.register(
        package_digest=_DIGEST_A,
        factory_reference=_EXTERNAL_CONTRACTOR_FACTORY_REF,
        factory=_external_factory,
    )


def _bundle(
    *,
    roster,
    manifest: ApplicationManifest,
    tmp_path: Path,
    marker: str,
    resolver: InMemoryRuntimeAgentFactoryResolver | None = None,
) -> RegistryProjectionInputBundle:
    _artifact_root, digest, lock = _build_echo_artifact(
        tmp_path,
        marker=marker,
        revision_bound_strict_external_work=True,
    )
    revision = _revision(
        "rev-roster-materialization",
        roster=roster,
        artifact_digest=digest,
        lock=lock,
    )
    delegate = resolver or InMemoryRuntimeAgentFactoryResolver()
    if resolver is None:
        _register_factories(delegate)
    return RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        factory_resolver=delegate,
        builders=None,
        materialization_artifact_digest=digest,
        roster_contract_authority=None,
    )


def _project(bundle: RegistryProjectionInputBundle):
    host_runtime = _host_runtime()
    return reproject_registry_with_strict_sanctioned_external_work(
        bundle,
        bundle.factory_resolver,  # type: ignore[arg-type]
        host_runtime=host_runtime,
        manifest=bundle.manifest,
    )


def test_e2e_positive_revision_bound_external_contractor_materialized(tmp_path: Path) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    roster = _roster_with_strict_external_work()
    projection = _project(_bundle(roster=roster, manifest=manifest, tmp_path=tmp_path, marker="e2e-pos"))
    assert "external_contractor_adapter" in projection.agent_registry.list_agent_ids()


def test_e2e_missing_external_contractor_roster_entry_fail_closed(tmp_path: Path) -> None:
    manifest = ApplicationManifest.lab(
        app_id=_APP,
        name="search-only",
        agents=[AgentBinding(contract_id="search")],
    )
    with pytest.raises(ValueError, match=_ROSTER_ERROR):
        _project(
            _bundle(
                roster=_roster(),
                manifest=manifest,
                tmp_path=tmp_path,
                marker="e2e-missing",
            ),
        )


def test_e2e_disabled_external_contractor_entry_fail_closed(tmp_path: Path) -> None:
    roster = _roster_with_strict_external_work()
    entries = list(roster.entries)
    entries[1] = entries[1].model_copy(update={"effective_enablement": False})
    roster = roster.model_copy(update={"entries": tuple(entries)})
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    with pytest.raises(ValueError, match=_ROSTER_ERROR):
        _project(
            _bundle(
                roster=roster,
                manifest=manifest,
                tmp_path=tmp_path,
                marker="e2e-disabled",
            ),
        )


def test_e2e_unknown_factory_reference_fail_closed_at_materialization(tmp_path: Path) -> None:
    roster = _roster_with_strict_external_work()
    entries = list(roster.entries)
    entries[1] = entries[1].model_copy(
        update={
            "factory_reference": AgentBindingFactoryReference(
                factory_path="missing.factory.build_agent",
            ),
        },
    )
    roster = roster.model_copy(update={"entries": tuple(entries)})
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    resolver = InMemoryRuntimeAgentFactoryResolver()
    _register_factories(resolver)
    with pytest.raises(RegistryProjectionError):
        _project(
            _bundle(
                roster=roster,
                manifest=manifest,
                tmp_path=tmp_path,
                marker="e2e-unknown-factory",
                resolver=resolver,
            ),
        )


def test_e2e_untrusted_package_digest_fail_closed(tmp_path: Path) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    roster = _roster_with_strict_external_work()
    entries = list(roster.entries)
    entries[1] = entries[1].model_copy(update={"package_digest": "sha256:" + ("9" * 64)})
    roster = roster.model_copy(update={"entries": tuple(entries)})
    with pytest.raises(RegistryProjectionError, match="not trusted"):
        _project(
            _bundle(
                roster=roster,
                manifest=manifest,
                tmp_path=tmp_path,
                marker="e2e-bad-digest",
            ),
        )


def test_e2e_runtime_revision_roster_revision_mismatch_fail_closed(tmp_path: Path) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    roster = _roster_with_strict_external_work()
    bundle = _bundle(
        roster=roster,
        manifest=manifest,
        tmp_path=tmp_path,
        marker="e2e-roster-rev",
    )
    stale = RegistryProjectionInputBundle(
        runtime_revision=bundle.runtime_revision.model_copy(
            update={"effective_roster_revision_id": "rev-stale-roster"},
        ),
        effective_roster=bundle.effective_roster,
        manifest=bundle.manifest,
        build_context=bundle.build_context,
        factory_resolver=bundle.factory_resolver,
        builders=None,
        materialization_artifact_digest=bundle.materialization_artifact_digest,
        roster_contract_authority=bundle.roster_contract_authority,
    )
    with pytest.raises(RegistryProjectionError, match="roster revision mismatch"):
        _project(stale)


def test_e2e_cross_environment_roster_fail_closed(tmp_path: Path) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    roster = _roster_with_strict_external_work().model_copy(
        update={"application_environment_id": "env-other"},
    )
    with pytest.raises(RegistryProjectionError, match="environment mismatch"):
        _project(
            _bundle(
                roster=roster,
                manifest=manifest,
                tmp_path=tmp_path,
                marker="e2e-cross-env",
            ),
        )


def test_e2e_unauthorized_factory_substitution_fail_closed(tmp_path: Path) -> None:
    manifest = _ac3_revision_bound_strict_external_work_host_manifest()
    roster = _roster_with_strict_external_work()
    resolver = InMemoryRuntimeAgentFactoryResolver()

    def _echo_substitute(
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> EchoAgent:
        return EchoAgent()

    resolver.register(
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
        factory=_echo_substitute,
    )
    resolver.register(
        package_digest=_DIGEST_A,
        factory_reference=_EXTERNAL_CONTRACTOR_FACTORY_REF,
        factory=_echo_substitute,
    )
    with pytest.raises(RegistryProjectionError, match="ExternalContractorAdapterAgent"):
        _project(
            _bundle(
                roster=roster,
                manifest=manifest,
                tmp_path=tmp_path,
                marker="e2e-substitution",
                resolver=resolver,
            ),
        )
