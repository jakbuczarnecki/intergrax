# © Artur Czarnecki. All rights reserved.

"""AP-10 registry projection from frozen RuntimeRevision + EffectiveRoster."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agent_distribution.activation import ActivationService
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.applications._shared.registry_projection import (
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
    InMemoryRuntimeRegistryProjectionStore,
    MaterializedRegistryProjection,
    RegistryProjectionError,
    RegistryProjectionInputBundle,
    build_registry_projection,
    projection_audit_snapshot,
)
from intergrax.applications._shared.wiring import (
    build_application_registry,
    binding_from_roster_entry,
    _index_manifest_bindings,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app_a"
_ENV = "env-prod"
_RELEASE = "rel-1"
_DIGEST = "sha256:" + ("a" * 64)
_LOCK_ID = "lock-1"
_LOCK_DIGEST = "sha256:" + ("b" * 64)
_GRAPH_DIGEST = "sha256:" + ("c" * 64)
_ARTIFACT = "sha256:" + ("d" * 64)
_ROSTER_A = "sha256:" + ("e" * 64)
_ROSTER_B = "sha256:" + ("f" * 64)

_RELEASE_OTHER = "rel-2"
_ARTIFACT_OTHER = "sha256:" + ("9" * 64)


def _echo_factory(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> EchoAgent:
    return EchoAgent()


ECHO_BUILDERS = {EchoAgent: _echo_factory}


def _manifest(
    *,
    agents: list[AgentBinding] | None = None,
    app_id: str = _APP,
) -> ApplicationManifest:
    roster_agents = agents or [
        AgentBinding.mount(EchoAgent, contract_id="search", factory=_echo_factory),
        AgentBinding.mount(EchoAgent, contract_id="indexer", factory=_echo_factory),
        AgentBinding.mount(EchoAgent, contract_id="synthesizer", factory=_echo_factory),
    ]
    return ApplicationManifest.lab(app_id=app_id, name="App A", agents=roster_agents)


def _entry(
    logical_agent_id: str,
    *,
    enabled: bool = True,
    default: bool = False,
    factory_reference: AgentBindingFactoryReference | None = None,
    manifest_origin_ref: str | None = None,
) -> EffectiveRosterEntry:
    return EffectiveRosterEntry(
        logical_agent_id=logical_agent_id,
        installation_slot_id=f"slot-{logical_agent_id}",
        package_digest=_DIGEST,
        distribution_package_id=f"pkg-{logical_agent_id}",
        effective_enablement=enabled,
        effective_default_agent=default,
        factory_reference=factory_reference,
        manifest_origin_ref=manifest_origin_ref or f"manifest:agents/{logical_agent_id}",
    )


def _roster(
    entries: tuple[EffectiveRosterEntry, ...],
    *,
    revision_id: str | None = None,
    manifest_release_id: str = _RELEASE,
) -> EffectiveRoster:
    roster = EffectiveRoster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id=manifest_release_id,
        entries=entries,
    ).with_revision_id()
    if revision_id is not None:
        roster = roster.model_copy(update={"effective_roster_revision_id": revision_id})
    return roster


def _revision(
    revision_id: str,
    *,
    roster_revision_id: str = _ROSTER_A,
    environment: str = _ENV,
    state: RuntimeRevisionState = RuntimeRevisionState.VALIDATED,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_environment_id=environment,
        application_release_id=_RELEASE,
        platform_version="0.1.0",
        effective_roster_revision_id=roster_revision_id,
        materialized_runtime_lock_id=_LOCK_ID,
        materialized_runtime_lock_digest=_LOCK_DIGEST,
        runtime_graph_digest=_GRAPH_DIGEST,
        materialization_artifact_digest=_ARTIFACT,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
        activated_at=datetime.now(UTC) if state is RuntimeRevisionState.ACTIVE else None,
    )


def _bundle(
    revision_id: str,
    roster: EffectiveRoster,
    manifest: ApplicationManifest | None = None,
    *,
    release_id: str = _RELEASE,
    artifact_digest: str | None = _ARTIFACT,
) -> RegistryProjectionInputBundle:
    manifest = manifest or _manifest()
    revision = _revision(
        revision_id,
        roster_revision_id=roster.effective_roster_revision_id or _ROSTER_A,
    )
    if release_id != _RELEASE:
        revision = revision.model_copy(update={"application_release_id": release_id})
    if artifact_digest is not None:
        revision = revision.model_copy(update={"materialization_artifact_digest": artifact_digest})
    ctx = ApplicationBuildContext.for_manifest(manifest)
    return RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=ctx,
        builders=ECHO_BUILDERS,
        materialization_artifact_digest=artifact_digest,
    )


def _bundle_parts(
    revision: RuntimeRevision,
    roster: EffectiveRoster,
    manifest: ApplicationManifest | None = None,
) -> RegistryProjectionInputBundle:
    manifest = manifest or _manifest()
    return RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        builders=ECHO_BUILDERS,
        materialization_artifact_digest=revision.materialization_artifact_digest,
    )


def _coordinator(
    state: AgentDistributionStoreState | None = None,
) -> tuple[
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
    InMemoryRuntimeRegistryProjectionStore,
    InMemoryRuntimeRevisionStore,
]:
    state = state or AgentDistributionStoreState()
    revision_store = InMemoryRuntimeRevisionStore(state)
    input_store = InMemoryRegistryProjectionInputStore()
    projection_store = InMemoryRuntimeRegistryProjectionStore()
    coordinator = ApplicationRegistryProjectionCoordinator(
        revision_store=revision_store,
        input_store=input_store,
        projection_store=projection_store,
    )
    return coordinator, input_store, projection_store, revision_store


def test_legacy_build_without_effective_roster_unchanged() -> None:
    manifest = _manifest(
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", factory=_echo_factory)]
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    registry = build_application_registry(manifest, ctx, builders=ECHO_BUILDERS)
    assert registry.list_agent_ids() == ["echo"]


def test_enabled_roster_entry_appears_in_registry() -> None:
    roster = _roster((_entry("search"), _entry("indexer")))
    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    registry = build_application_registry(
        manifest,
        ctx,
        builders=ECHO_BUILDERS,
        effective_roster=roster,
    )
    assert set(registry.list_agent_ids()) == {"search", "indexer"}


def test_disabled_roster_entry_not_in_registry() -> None:
    roster = _roster((_entry("search"), _entry("synthesizer", enabled=False)))
    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    registry = build_application_registry(
        manifest,
        ctx,
        builders=ECHO_BUILDERS,
        effective_roster=roster,
    )
    assert registry.list_agent_ids() == ["search"]


def test_operator_added_roster_agent_via_builder_key() -> None:
    roster = _roster(
        (
            EffectiveRosterEntry(
                logical_agent_id="custom",
                installation_slot_id="slot-custom",
                package_digest=_DIGEST,
                distribution_package_id="pkg-custom",
                effective_enablement=True,
                factory_reference=AgentBindingFactoryReference(builder_key="custom"),
            ),
        )
    )
    manifest = ApplicationManifest.lab(app_id=_APP, name="App A", agents=[])
    ctx = ApplicationBuildContext.for_manifest(manifest)
    custom_builders = {"custom": _echo_factory}
    registry = build_application_registry(
        manifest,
        ctx,
        builders=custom_builders,
        effective_roster=roster,
    )
    assert registry.has("custom")


def test_manifest_agent_excluded_by_roster_does_not_reappear() -> None:
    roster = _roster((_entry("search"),))
    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    registry = build_application_registry(
        manifest,
        ctx,
        builders=ECHO_BUILDERS,
        effective_roster=roster,
    )
    assert registry.list_agent_ids() == ["search"]


def test_roster_default_override_preserved() -> None:
    manifest = _manifest()
    binding = binding_from_roster_entry(
        _entry("search", default=True),
        _index_manifest_bindings(manifest),
    )
    assert binding.default is True


def test_frozen_roster_ignores_later_binding_mutation_inputs() -> None:
    roster_frozen = _roster((_entry("search"),))
    roster_mutated = _roster((_entry("search"), _entry("indexer")))
    manifest = _manifest()
    projection_a = build_registry_projection(_bundle("rev-1", roster_frozen, manifest))
    projection_b = build_registry_projection(
        _bundle_parts(
            _revision(
                "rev-2",
                roster_revision_id=roster_mutated.effective_roster_revision_id or _ROSTER_B,
            ),
            roster_mutated,
            manifest,
        )
    )
    assert projection_a.agent_registry.list_agent_ids() == ["search"]
    assert set(projection_b.agent_registry.list_agent_ids()) == {"search", "indexer"}


def test_projection_bound_to_runtime_revision_id() -> None:
    roster = _roster((_entry("search"),))
    projection = build_registry_projection(_bundle("rev-18", roster))
    assert projection.evidence.runtime_revision_id == "rev-18"


def test_projection_rejects_mismatched_environment() -> None:
    roster = _roster((_entry("search"),))
    bundle = _bundle("rev-1", roster)
    state = AgentDistributionStoreState()
    coordinator, input_store, _, revision_store = _coordinator(state)
    revision_store.persist_candidate_revision(
        _revision("rev-1", environment="env-other")
    )
    input_store.register(bundle)
    with pytest.raises(RegistryProjectionError, match="environment"):
        coordinator.prepare_projection("rev-1")


def test_projection_rejects_mismatched_roster_revision_id() -> None:
    roster = _roster((_entry("search"),), revision_id=_ROSTER_A)
    bundle = _bundle("rev-1", roster)
    state = AgentDistributionStoreState()
    coordinator, input_store, _, revision_store = _coordinator(state)
    revision_store.persist_candidate_revision(
        _revision("rev-1", roster_revision_id=_ROSTER_B)
    )
    input_store.register(bundle)
    with pytest.raises(RegistryProjectionError, match="roster revision"):
        coordinator.prepare_projection("rev-1")


def test_projection_rejects_lock_identity_mismatch() -> None:
    roster = _roster((_entry("search"),))
    bundle = _bundle("rev-1", roster)
    state = AgentDistributionStoreState()
    coordinator, input_store, _, revision_store = _coordinator(state)
    revision_store.persist_candidate_revision(
        bundle.runtime_revision.model_copy(
            update={"materialized_runtime_lock_digest": "sha256:" + ("9" * 64)}
        )
    )
    input_store.register(bundle)
    with pytest.raises(RegistryProjectionError, match="lock digest"):
        coordinator.prepare_projection("rev-1")


def test_registry_n_and_n_plus_one_coexist() -> None:
    roster_n = _roster((_entry("search"),))
    roster_n1 = _roster((_entry("search"), _entry("indexer")), revision_id=_ROSTER_B)
    projection_n = build_registry_projection(_bundle("rev-n", roster_n))
    projection_n1 = build_registry_projection(
        _bundle_parts(
            _revision(
                "rev-n1",
                roster_revision_id=roster_n1.effective_roster_revision_id or _ROSTER_B,
            ),
            roster_n1,
        )
    )
    assert projection_n.agent_registry is not projection_n1.agent_registry
    assert projection_n.agent_registry.list_agent_ids() == ["search"]
    assert set(projection_n1.agent_registry.list_agent_ids()) == {"search", "indexer"}


def test_preparing_n_plus_one_does_not_mutate_registry_n() -> None:
    coordinator, input_store, projection_store, revision_store = _coordinator()
    roster_n = _roster((_entry("search"),))
    roster_n1 = _roster((_entry("search"), _entry("indexer")), revision_id=_ROSTER_B)
    bundle_n = _bundle("rev-n", roster_n)
    bundle_n1 = _bundle_parts(
        _revision(
            "rev-n1",
            roster_revision_id=roster_n1.effective_roster_revision_id or _ROSTER_B,
        ),
        roster_n1,
    )
    revision_store.persist_candidate_revision(bundle_n.runtime_revision)
    revision_store.persist_candidate_revision(bundle_n1.runtime_revision)
    input_store.register(bundle_n)
    input_store.register(bundle_n1)
    coordinator.prepare_projection("rev-n")
    registry_n_before = projection_store.get("rev-n").agent_registry.list_agent_ids()
    coordinator.prepare_projection("rev-n1")
    registry_n_after = projection_store.get("rev-n").agent_registry.list_agent_ids()
    assert registry_n_before == registry_n_after == ["search"]


def test_activation_uses_prepared_projection() -> None:
    state = AgentDistributionStoreState()
    coordinator, input_store, projection_store, revision_store = _coordinator(state)
    roster = _roster((_entry("search"),))
    bundle = _bundle("rev-1", roster)
    revision_store.persist_candidate_revision(bundle.runtime_revision)
    input_store.register(bundle)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    activation_store = InMemoryApplicationEnvironmentActivationStore(state)
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_store,
        serving_store=serving_store,
        activation_store=activation_store,
        deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        projection_coordinator=coordinator,
    )
    activation.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-1",
        artifact_locator=f"artifact://{_ARTIFACT}",
    )
    activation.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-1",
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=_ARTIFACT,
    )
    prepared = projection_store.get("rev-1")
    assert prepared is not None
    assert prepared.agent_registry.has("search")


def test_projection_prepare_failure_blocks_activation_commit() -> None:
    state = AgentDistributionStoreState()
    coordinator, _, _, revision_store = _coordinator(state)
    roster = _roster((_entry("search"),))
    bundle = _bundle("rev-1", roster)
    revision_store.persist_candidate_revision(bundle.runtime_revision)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    activation_store = InMemoryApplicationEnvironmentActivationStore(state)
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_store,
        serving_store=serving_store,
        activation_store=activation_store,
        deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        projection_coordinator=coordinator,
    )
    activation.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-1",
        artifact_locator=f"artifact://{_ARTIFACT}",
    )
    with pytest.raises(RegistryProjectionError):
        activation.commit_activation(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id="rev-1",
            expected_prior_traffic_revision_id=None,
            expected_serving_pointer_revision=0,
            expected_artifact_digest=_ARTIFACT,
        )


def test_rollback_restores_registry_n_projection() -> None:
    coordinator, input_store, projection_store, revision_store = _coordinator()
    roster_n = _roster((_entry("search"),))
    roster_n1 = _roster((_entry("search"), _entry("indexer")), revision_id=_ROSTER_B)
    bundle_n = _bundle("rev-n", roster_n)
    bundle_n1 = _bundle_parts(
        _revision(
            "rev-n1",
            roster_revision_id=roster_n1.effective_roster_revision_id or _ROSTER_B,
        ),
        roster_n1,
    )
    revision_store.persist_candidate_revision(bundle_n.runtime_revision)
    revision_store.persist_candidate_revision(bundle_n1.runtime_revision)
    input_store.register(bundle_n)
    input_store.register(bundle_n1)
    coordinator.prepare_projection("rev-n")
    coordinator.prepare_projection("rev-n1")
    original_n = projection_store.get("rev-n")
    coordinator.rollback_projection("rev-n")
    restored = projection_store.get("rev-n")
    assert restored is original_n
    assert restored.agent_registry.list_agent_ids() == ["search"]


def test_rollback_reuses_cached_projection_not_desired_state() -> None:
    coordinator, input_store, projection_store, revision_store = _coordinator()
    roster_n = _roster((_entry("search"),))
    bundle_n = _bundle("rev-n", roster_n)
    revision_store.persist_candidate_revision(bundle_n.runtime_revision)
    input_store.register(bundle_n)
    token_first = coordinator.prepare_projection("rev-n")
    roster_mutated = _roster((_entry("search"), _entry("indexer")), revision_id=_ROSTER_B)
    with pytest.raises(RegistryProjectionError, match="conflicting frozen projection inputs"):
        input_store.register(
            _bundle_parts(bundle_n.runtime_revision, roster_mutated),
        )
    token_second = coordinator.prepare_projection("rev-n")
    assert token_first == token_second
    assert projection_store.get("rev-n").agent_registry.list_agent_ids() == ["search"]


def test_capability_lookup_only_enabled_revision_agents() -> None:
    roster = _roster((_entry("search"),))
    projection = build_registry_projection(_bundle("rev-1", roster))
    registry = projection.agent_registry
    matches = registry.find_by_capability("echo.basic")
    assert len(matches) == 1
    assert registry.find_by_capability("nonexistent.cap") == []


def test_disabled_agent_not_routable_via_registry() -> None:
    roster = _roster((_entry("search"), _entry("synthesizer", enabled=False)))
    projection = build_registry_projection(_bundle("rev-1", roster))
    assert projection.agent_registry.list_agent_ids() == ["search"]


def test_audit_snapshot_contains_revision_identity_fields() -> None:
    roster = _roster((_entry("search"),))
    projection = build_registry_projection(_bundle("rev-42", roster))
    snapshot = projection_audit_snapshot(projection)
    assert snapshot.evidence.runtime_revision_id == "rev-42"
    assert snapshot.evidence.effective_roster_revision_id == roster.effective_roster_revision_id
    assert snapshot.agent_contract_ids == ("search",)


def test_snapshot_n_distinct_from_n_plus_one() -> None:
    roster_n = _roster((_entry("search"),))
    roster_n1 = _roster((_entry("search"), _entry("indexer")), revision_id=_ROSTER_B)
    snap_n = projection_audit_snapshot(build_registry_projection(_bundle("rev-n", roster_n)))
    snap_n1 = projection_audit_snapshot(
        build_registry_projection(
            _bundle_parts(
                _revision(
                    "rev-n1",
                    roster_revision_id=roster_n1.effective_roster_revision_id or _ROSTER_B,
                ),
                roster_n1,
            )
        )
    )
    assert snap_n.evidence.runtime_revision_id != snap_n1.evidence.runtime_revision_id
    assert snap_n.agent_contract_ids != snap_n1.agent_contract_ids


def test_projection_accepts_correct_application_release_identity() -> None:
    roster = _roster((_entry("search"),))
    projection = build_registry_projection(_bundle("rev-1", roster))
    assert projection.evidence.application_release_id == _RELEASE
    assert projection.evidence.materialization_artifact_digest == _ARTIFACT


def test_projection_rejects_mismatched_application_release_id() -> None:
    roster = _roster((_entry("search"),), manifest_release_id=_RELEASE)
    bundle = _bundle("rev-1", roster, release_id=_RELEASE_OTHER)
    with pytest.raises(RegistryProjectionError, match="release mismatch"):
        build_registry_projection(bundle)


def test_projection_rejects_mismatched_materialization_artifact_digest() -> None:
    roster = _roster((_entry("search"),))
    revision = _revision("rev-1", roster_revision_id=roster.effective_roster_revision_id or _ROSTER_A)
    bundle = RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=_manifest(),
        build_context=ApplicationBuildContext.for_manifest(_manifest()),
        builders=ECHO_BUILDERS,
        materialization_artifact_digest=_ARTIFACT_OTHER,
    )
    with pytest.raises(RegistryProjectionError, match="materialization artifact"):
        build_registry_projection(bundle)


def test_other_release_build_context_cannot_build_registry_n() -> None:
    roster_n = _roster((_entry("search"),), manifest_release_id=_RELEASE)
    roster_other = _roster((_entry("search"),), manifest_release_id=_RELEASE_OTHER)
    manifest_other = _manifest(
        agents=[AgentBinding.mount(EchoAgent, contract_id="other", factory=_echo_factory)]
    )
    with pytest.raises(RegistryProjectionError, match="release mismatch"):
        build_registry_projection(
            RegistryProjectionInputBundle(
                runtime_revision=_revision(
                    "rev-n",
                    roster_revision_id=roster_n.effective_roster_revision_id or _ROSTER_A,
                ),
                effective_roster=roster_other,
                manifest=manifest_other,
                build_context=ApplicationBuildContext.for_manifest(manifest_other),
                builders=ECHO_BUILDERS,
                materialization_artifact_digest=_ARTIFACT,
            )
        )


def test_future_revision_wiring_cannot_replace_revision_n_inputs() -> None:
    coordinator, input_store, projection_store, revision_store = _coordinator()
    roster_n = _roster((_entry("search"),))
    roster_n1 = _roster((_entry("search"), _entry("indexer")), revision_id=_ROSTER_B)
    bundle_n = _bundle("rev-n", roster_n)
    bundle_n1 = _bundle_parts(
        _revision(
            "rev-n1",
            roster_revision_id=roster_n1.effective_roster_revision_id or _ROSTER_B,
        ),
        roster_n1,
    )
    revision_store.persist_candidate_revision(bundle_n.runtime_revision)
    revision_store.persist_candidate_revision(bundle_n1.runtime_revision)
    input_store.register(bundle_n)
    input_store.register(bundle_n1)
    coordinator.prepare_projection("rev-n")
    with pytest.raises(RegistryProjectionError, match="conflicting frozen projection inputs"):
        input_store.register(
            _bundle_parts(bundle_n.runtime_revision, roster_n1),
        )
    assert projection_store.get("rev-n").agent_registry.list_agent_ids() == ["search"]


def test_register_same_frozen_input_is_idempotent() -> None:
    _, input_store, _, _ = _coordinator()
    bundle = _bundle("rev-1", _roster((_entry("search"),)))
    input_store.register(bundle)
    input_store.register(bundle)
    assert input_store.get("rev-1") is bundle


def test_register_conflicting_input_for_same_revision_rejected() -> None:
    _, input_store, _, _ = _coordinator()
    bundle = _bundle("rev-1", _roster((_entry("search"),)))
    input_store.register(bundle)
    conflicting = _bundle_parts(
        bundle.runtime_revision,
        _roster((_entry("search"), _entry("indexer")), revision_id=_ROSTER_B),
    )
    with pytest.raises(RegistryProjectionError, match="conflicting frozen projection inputs"):
        input_store.register(conflicting)


def test_concurrent_prepare_same_revision_returns_one_semantic_projection() -> None:
    coordinator, input_store, projection_store, revision_store = _coordinator()
    roster = _roster((_entry("search"),))
    bundle = _bundle("rev-concurrent", roster)
    revision_store.persist_candidate_revision(bundle.runtime_revision)
    input_store.register(bundle)

    def _prepare() -> str:
        return coordinator.prepare_projection("rev-concurrent")

    with ThreadPoolExecutor(max_workers=4) as pool:
        tokens = list(pool.map(lambda _: _prepare(), range(8)))

    assert len(set(tokens)) == 1
    projection = projection_store.get("rev-concurrent")
    assert projection is not None
    assert projection.agent_registry.list_agent_ids() == ["search"]


def test_concurrent_conflicting_projection_fails_closed() -> None:
    store = InMemoryRuntimeRegistryProjectionStore()
    roster = _roster((_entry("search"),))
    projection_a = build_registry_projection(_bundle("rev-conflict", roster))
    conflicting_evidence = projection_a.evidence.model_copy(
        update={"registry_factory_wiring_digest": "sha256:" + ("1" * 64)}
    )
    projection_b = MaterializedRegistryProjection(
        evidence=conflicting_evidence,
        agent_registry=projection_a.agent_registry,
        harness_snapshot=projection_a.harness_snapshot,
    )
    store.put(projection_a)
    with pytest.raises(RegistryProjectionError, match="conflicting registry projection"):
        store.put(projection_b)


def test_operator_added_agent_rejects_missing_frozen_builder_reference() -> None:
    roster = _roster(
        (
            EffectiveRosterEntry(
                logical_agent_id="custom",
                installation_slot_id="slot-custom",
                package_digest=_DIGEST,
                distribution_package_id="pkg-custom",
                effective_enablement=True,
                factory_reference=AgentBindingFactoryReference(builder_key="custom"),
            ),
        )
    )
    manifest = ApplicationManifest.lab(app_id=_APP, name="App A", agents=[])
    bundle = RegistryProjectionInputBundle(
        runtime_revision=_revision(
            "rev-1",
            roster_revision_id=roster.effective_roster_revision_id or _ROSTER_A,
        ),
        effective_roster=roster,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        builders={},
        materialization_artifact_digest=_ARTIFACT,
    )
    with pytest.raises(RegistryProjectionError, match="missing from frozen release builders"):
        build_registry_projection(bundle)


def test_audit_snapshot_contains_release_binding_fields() -> None:
    roster = _roster((_entry("search"),))
    projection = build_registry_projection(_bundle("rev-42", roster))
    snapshot = projection_audit_snapshot(projection)
    assert snapshot.evidence.application_release_id == _RELEASE
    assert snapshot.evidence.registry_factory_wiring_digest
    assert snapshot.evidence.materialization_artifact_digest == _ARTIFACT
