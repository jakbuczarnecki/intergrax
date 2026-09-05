# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only Agent Manager query facade — composes catalog + lifecycle projections."""

from __future__ import annotations

from intergrax.agent_distribution.agent_manager_models import (
    AgentManagerAvailabilityView,
    AgentManagerDerivedStatus,
    AgentManagerDiscoveryView,
    AgentManagerEntry,
    AgentManagerIdentityView,
    AgentManagerLifecycleView,
    AgentManagerListFilters,
    AgentManagerListResult,
    AgentManagerListScope,
    AgentManagerRuntimeView,
    LifecycleMatchResolution,
)
from intergrax.agent_distribution.binding import ApplicationAgentBinding
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogEntryFilters,
    CatalogSourceProvider,
)
from intergrax.agent_distribution.effective_roster import EffectiveRosterBuilder
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    installation_state_is_installed,
)
from intergrax.agent_distribution.runtime_revision import (
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.roster import ManifestDefaultAgentDeclaration
from intergrax.agent_distribution.stores import (
    AgentInstallationStore,
    ApplicationAgentBindingStore,
    ApplicationEnvironmentServingStore,
    RuntimeRevisionStore,
)
from intergrax.runtime.architecture.capability_graph_query import CapabilityGraphQuery


def _manager_entry_id_for_catalog(entry: AgentCatalogEntry) -> str:
    return (
        f"catalog:{entry.catalog_source.catalog_source_id}:{entry.catalog_entry_id}"
    )


def _manager_entry_id_for_lifecycle(
  *,
    logical_agent_id: str | None,
    installation_id: str | None,
) -> str:
    if logical_agent_id is not None:
        return f"lifecycle:logical:{logical_agent_id}"
    if installation_id is not None:
        return f"lifecycle:installation:{installation_id}"
    return "lifecycle:unknown"


def _entry_sort_key(entry: AgentManagerEntry) -> tuple[str, ...]:
    identity = entry.identity
    source = identity.catalog_source
    source_id = source.catalog_source_id if source is not None else ""
    origin_rank = "0" if source is not None else "1"
    return (
        origin_rank,
        source_id,
        identity.catalog_entry_id or "",
        identity.package_id_line or "",
        identity.display_name,
        entry.identity.manager_entry_id,
    )


class _LifecycleIndexes:
    def __init__(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        installation_store: AgentInstallationStore,
        binding_store: ApplicationAgentBindingStore,
        roster_builder: EffectiveRosterBuilder,
        revision_store: RuntimeRevisionStore,
        serving_store: ApplicationEnvironmentServingStore,
        manifest_defaults: tuple[object, ...],
    ) -> None:
        self.application_id = application_id
        self.application_environment_id = application_environment_id
        self.installations = installation_store.list_installations_for_environment(
            application_environment_id,
        )
        self.bindings = [
            binding
            for binding in binding_store.list_bindings_for_environment(
                application_id,
                application_environment_id,
            )
            if not binding.tombstone
        ]
        self.roster = roster_builder.build(
            application_id=application_id,
            application_environment_id=application_environment_id,
            manifest_release_id="unspecified",
            manifest_defaults=manifest_defaults,
            durable_bindings=self.bindings,
        )
        self.serving = serving_store.get_serving_record(
            application_id,
            application_environment_id,
        )
        self.revisions = revision_store.list_revisions_for_environment(
            application_id,
            application_environment_id,
        )
        self._slot_active: dict[str, AgentInstallationRecord] = {}
        for record in self.installations:
            if record.active_for_slot and installation_state_is_installed(
                record.installation_state,
            ):
                self._slot_active[record.installation_slot_id] = record
        self._bindings_by_slot: dict[str, ApplicationAgentBinding] = {
            binding.installation_slot_id: binding for binding in self.bindings
        }
        self._bindings_by_logical: dict[str, ApplicationAgentBinding] = {
            binding.logical_agent_id: binding for binding in self.bindings
        }
        self._roster_by_logical = {
            entry.logical_agent_id: entry for entry in self.roster.entries
        }

    def active_installation_for_slot(
        self,
        installation_slot_id: str,
    ) -> AgentInstallationRecord | None:
        return self._slot_active.get(installation_slot_id)

    def binding_for_slot(
        self,
        installation_slot_id: str,
    ) -> ApplicationAgentBinding | None:
        return self._bindings_by_slot.get(installation_slot_id)

    def binding_for_logical(
        self,
        logical_agent_id: str,
    ) -> ApplicationAgentBinding | None:
        return self._bindings_by_logical.get(logical_agent_id)

    def roster_enabled(self, logical_agent_id: str) -> bool:
        entry = self._roster_by_logical.get(logical_agent_id)
        return entry.effective_enablement if entry is not None else False

    def traffic_serving_revision_id(self) -> str | None:
        if self.serving is None:
            return None
        return self.serving.traffic_serving_revision_id

    def pending_candidate(self) -> RuntimeRevision | None:
        traffic_id = self.traffic_serving_revision_id()
        pending: list[RuntimeRevision] = []
        for revision in self.revisions:
            if revision.runtime_revision_id == traffic_id:
                continue
            if revision.revision_state in {
                RuntimeRevisionState.CANDIDATE,
                RuntimeRevisionState.VALIDATED,
            }:
                pending.append(revision)
        if not pending:
            return None
        pending.sort(key=lambda item: item.runtime_revision_id)
        return pending[-1]

    def active_revision(self) -> RuntimeRevision | None:
        traffic_id = self.traffic_serving_revision_id()
        if traffic_id is None:
            return None
        for revision in self.revisions:
            if revision.runtime_revision_id == traffic_id:
                return revision
        return None

    def installations_for_package_line(
        self,
        package_id_line: str,
    ) -> tuple[AgentInstallationRecord, ...]:
        matched = tuple(
            record
            for record in self.installations
            if record.package_identity.distribution_package_id == package_id_line
            and installation_state_is_installed(record.installation_state)
            and record.active_for_slot
        )
        return matched


class AgentManagerQueryService:
    """Compose Agent Manager read model from catalog and lifecycle read APIs."""

    def __init__(
        self,
        *,
        catalog_provider: CatalogSourceProvider,
        installation_store: AgentInstallationStore,
        binding_store: ApplicationAgentBindingStore,
        revision_store: RuntimeRevisionStore,
        serving_store: ApplicationEnvironmentServingStore,
        roster_builder: EffectiveRosterBuilder,
        capability_graph_query: CapabilityGraphQuery | None = None,
        manifest_defaults: tuple[ManifestDefaultAgentDeclaration, ...] = (),
    ) -> None:
        self._catalog_provider = catalog_provider
        self._installation_store = installation_store
        self._binding_store = binding_store
        self._revision_store = revision_store
        self._serving_store = serving_store
        self._roster_builder = roster_builder
        self._capability_graph_query = capability_graph_query
        self._manifest_defaults = manifest_defaults

    def list_agents(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        filters: AgentManagerListFilters | None = None,
    ) -> AgentManagerListResult:
        scope = AgentManagerListScope(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        catalog_filters = _to_catalog_filters(filters)
        catalog_entries = tuple(
            self._catalog_provider.list_entries(catalog_filters),
        )
        indexes = _LifecycleIndexes(
            application_id=application_id,
            application_environment_id=application_environment_id,
            installation_store=self._installation_store,
            binding_store=self._binding_store,
            roster_builder=self._roster_builder,
            revision_store=self._revision_store,
            serving_store=self._serving_store,
            manifest_defaults=self._manifest_defaults,
        )
        items: list[AgentManagerEntry] = []
        consumed_slots: set[str] = set()
        consumed_logical: set[str] = set()
        for entry in catalog_entries:
            composed = self._compose_catalog_entry(
                entry,
                indexes=indexes,
                application_id=application_id,
            )
            items.append(composed)
            if composed.lifecycle.installation_slot_id is not None:
                consumed_slots.add(composed.lifecycle.installation_slot_id)
            if composed.lifecycle.logical_agent_id is not None:
                consumed_logical.add(composed.lifecycle.logical_agent_id)
        for binding in indexes.bindings:
            if binding.logical_agent_id in consumed_logical:
                continue
            items.append(
                self._compose_lifecycle_only_binding(
                    binding,
                    indexes=indexes,
                    application_id=application_id,
                ),
            )
            consumed_logical.add(binding.logical_agent_id)
            consumed_slots.add(binding.installation_slot_id)
        for record in indexes.installations:
            if (
                not installation_state_is_installed(record.installation_state)
                or not record.active_for_slot
            ):
                continue
            if record.installation_slot_id in consumed_slots:
                continue
            items.append(
                self._compose_lifecycle_only_installation(
                    record,
                    indexes=indexes,
                ),
            )
            consumed_slots.add(record.installation_slot_id)
        filtered = tuple(
            item for item in items if _matches_filters(item, filters, application_id)
        )
        ordered = tuple(sorted(filtered, key=_entry_sort_key))
        return AgentManagerListResult(
            items=ordered,
            total=len(ordered),
            scope=scope,
            filters=filters,
        )

    def inspect_agent(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        manager_entry_id: str,
    ) -> AgentManagerEntry | None:
        result = self.list_agents(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
        for item in result.items:
            if item.identity.manager_entry_id == manager_entry_id:
                return item
        return None

    def list_agents_for_application(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        filters: AgentManagerListFilters | None = None,
    ) -> AgentManagerListResult:
        if self._capability_graph_query is None:
            return self.list_agents(
                application_id=application_id,
                application_environment_id=application_environment_id,
                filters=filters,
            )
        graph_agents = self._capability_graph_query.agents_for_application(
            application_id,
        )
        base = self.list_agents(
            application_id=application_id,
            application_environment_id=application_environment_id,
            filters=filters,
        )
        allowed = set(graph_agents)
        items = tuple(
            item
            for item in base.items
            if item.lifecycle.logical_agent_id in allowed
        )
        return AgentManagerListResult(
            items=items,
            total=len(items),
            scope=base.scope,
            filters=filters,
        )

    def _compose_catalog_entry(
        self,
        entry: AgentCatalogEntry,
        *,
        indexes: _LifecycleIndexes,
        application_id: str,
    ) -> AgentManagerEntry:
        capabilities = self._capabilities_for_catalog(entry, application_id)
        discovery = AgentManagerDiscoveryView(
            categories=entry.categories,
            trust_labels=entry.trust_labels,
            capabilities=capabilities,
            compatibility_summary=entry.compatibility_summary,
            provider_kind=entry.catalog_source.provider_kind,
        )
        identity = AgentManagerIdentityView(
            manager_entry_id=_manager_entry_id_for_catalog(entry),
            catalog_entry_id=entry.catalog_entry_id,
            package_id_line=entry.package_id_line,
            display_name=entry.display_name,
            catalog_source=entry.catalog_source,
            publisher=entry.publisher,
        )
        matched = indexes.installations_for_package_line(entry.package_id_line)
        lifecycle, runtime = self._lifecycle_runtime_from_match(
            matched_installations=matched,
            indexes=indexes,
        )
        if lifecycle.logical_agent_id is not None:
            logical_caps = self._capabilities_for_logical(lifecycle.logical_agent_id)
            if logical_caps:
                discovery = discovery.model_copy(update={"capabilities": logical_caps})
        derived = _derive_status(lifecycle=lifecycle, runtime=runtime)
        availability = _derive_availability(lifecycle=lifecycle, runtime=runtime)
        return AgentManagerEntry(
            derived_status=derived,
            identity=identity,
            discovery=discovery,
            lifecycle=lifecycle,
            runtime=runtime,
            availability=availability,
        )

    def _compose_lifecycle_only_binding(
        self,
        binding: ApplicationAgentBinding,
        *,
        indexes: _LifecycleIndexes,
        application_id: str,
    ) -> AgentManagerEntry:
        active = indexes.active_installation_for_slot(binding.installation_slot_id)
        matched = (active,) if active is not None else ()
        lifecycle, runtime = self._lifecycle_runtime_from_match(
            matched_installations=matched,
            indexes=indexes,
            binding=binding,
        )
        lifecycle = lifecycle.model_copy(
            update={
                "match_resolution": LifecycleMatchResolution.UNRESOLVED,
                "logical_agent_id": binding.logical_agent_id,
                "application_binding_id": binding.application_binding_id,
                "installation_slot_id": binding.installation_slot_id,
                "bound": True,
                "enabled_in_desired_state": indexes.roster_enabled(
                    binding.logical_agent_id,
                ),
            },
        )
        display = binding.logical_agent_id
        if active is not None:
            display = active.package_identity.distribution_package_id
        derived = _derive_status(lifecycle=lifecycle, runtime=runtime)
        return AgentManagerEntry(
            derived_status=derived,
            identity=AgentManagerIdentityView(
                manager_entry_id=_manager_entry_id_for_lifecycle(
                    logical_agent_id=binding.logical_agent_id,
                    installation_id=None,
                ),
                display_name=display,
                package_id_line=(
                    active.package_identity.distribution_package_id
                    if active is not None
                    else None
                ),
            ),
            discovery=AgentManagerDiscoveryView(
                capabilities=self._capabilities_for_logical(binding.logical_agent_id),
            ),
            lifecycle=lifecycle,
            runtime=runtime,
            availability=_derive_availability(lifecycle=lifecycle, runtime=runtime),
        )

    def _compose_lifecycle_only_installation(
        self,
        record: AgentInstallationRecord,
        *,
        indexes: _LifecycleIndexes,
    ) -> AgentManagerEntry:
        binding = indexes.binding_for_slot(record.installation_slot_id)
        matched = (record,)
        lifecycle, runtime = self._lifecycle_runtime_from_match(
            matched_installations=matched,
            indexes=indexes,
            binding=binding,
        )
        lifecycle = lifecycle.model_copy(
            update={
                "match_resolution": LifecycleMatchResolution.UNRESOLVED,
                "installed": True,
                "installation_state": record.installation_state,
                "installation_id": record.installation_id,
                "installation_slot_id": record.installation_slot_id,
                "distribution_package_id": record.package_identity.distribution_package_id,
                "package_version": record.package_identity.package_version,
                "package_digest": record.package_identity.package_digest,
            },
        )
        derived = _derive_status(lifecycle=lifecycle, runtime=runtime)
        return AgentManagerEntry(
            derived_status=derived,
            identity=AgentManagerIdentityView(
                manager_entry_id=_manager_entry_id_for_lifecycle(
                    logical_agent_id=None,
                    installation_id=record.installation_id,
                ),
                display_name=record.package_identity.distribution_package_id,
                package_id_line=record.package_identity.distribution_package_id,
            ),
            discovery=AgentManagerDiscoveryView(),
            lifecycle=lifecycle,
            runtime=runtime,
            availability=_derive_availability(lifecycle=lifecycle, runtime=runtime),
        )

    def _lifecycle_runtime_from_match(
        self,
        *,
        matched_installations: tuple[AgentInstallationRecord, ...],
        indexes: _LifecycleIndexes,
        binding: ApplicationAgentBinding | None = None,
    ) -> tuple[AgentManagerLifecycleView, AgentManagerRuntimeView]:
        if len(matched_installations) > 1:
            lifecycle = AgentManagerLifecycleView(
                match_resolution=LifecycleMatchResolution.AMBIGUOUS,
            )
            return lifecycle, AgentManagerRuntimeView()
        record = matched_installations[0] if matched_installations else None
        if binding is None and record is not None:
            binding = indexes.binding_for_slot(record.installation_slot_id)
        logical_agent_id = binding.logical_agent_id if binding is not None else None
        enabled = (
            indexes.roster_enabled(logical_agent_id)
            if logical_agent_id is not None
            else False
        )
        lifecycle = AgentManagerLifecycleView(
            match_resolution=(
                LifecycleMatchResolution.RESOLVED
                if record is not None
                else LifecycleMatchResolution.NOT_APPLICABLE
            ),
            installation_state=record.installation_state if record else None,
            installed=record is not None
            and installation_state_is_installed(record.installation_state),
            bound=binding is not None,
            enabled_in_desired_state=enabled,
            logical_agent_id=logical_agent_id,
            installation_id=record.installation_id if record else None,
            installation_slot_id=(
                record.installation_slot_id if record else None
            ),
            application_binding_id=(
                binding.application_binding_id if binding is not None else None
            ),
            distribution_package_id=(
                record.package_identity.distribution_package_id if record else None
            ),
            package_version=(
                record.package_identity.package_version if record else None
            ),
            package_digest=(
                record.package_identity.package_digest if record else None
            ),
        )
        runtime = _runtime_projection(
            package_digest=lifecycle.package_digest,
            indexes=indexes,
            enabled=enabled,
        )
        return lifecycle, runtime

    def _capabilities_for_catalog(
        self,
        entry: AgentCatalogEntry,
        application_id: str,
    ) -> tuple[str, ...]:
        if self._capability_graph_query is None:
            return ()
        for version_channel in entry.version_channel_refs:
            contract_id = version_channel.package_version
            if contract_id:
                caps = self._capability_graph_query.capabilities_for_agent(
                    contract_id,
                )
                if caps:
                    return caps
        agents = self._capability_graph_query.agents_for_application(application_id)
        if not agents:
            return ()
        package_line = entry.package_id_line
        matched: list[str] = []
        for agent_id in agents:
            if package_line in agent_id or agent_id in package_line:
                matched.extend(
                    self._capability_graph_query.capabilities_for_agent(agent_id),
                )
        return tuple(sorted(set(matched)))

    def _capabilities_for_logical(self, logical_agent_id: str) -> tuple[str, ...]:
        if self._capability_graph_query is None:
            return ()
        return self._capability_graph_query.capabilities_for_agent(logical_agent_id)


def _runtime_projection(
    *,
    package_digest: str | None,
    indexes: _LifecycleIndexes,
    enabled: bool,
) -> AgentManagerRuntimeView:
    active = indexes.active_revision()
    pending = indexes.pending_candidate()
    traffic_id = indexes.traffic_serving_revision_id()
    included_active = False
    included_candidate = False
    if package_digest is not None and active is not None:
        included_active = package_digest in active.installed_agent_package_digests
    if package_digest is not None and pending is not None:
        included_candidate = (
            package_digest in pending.installed_agent_package_digests
        )
    serving = included_active and enabled
    return AgentManagerRuntimeView(
        included_in_active_revision=included_active,
        included_in_candidate_revision=included_candidate,
        traffic_serving_revision_id=traffic_id,
        pending_candidate_revision_id=(
            pending.runtime_revision_id if pending is not None else None
        ),
        pending_candidate_revision_state=(
            pending.revision_state if pending is not None else None
        ),
        serving=serving,
    )


def _derive_status(
    *,
    lifecycle: AgentManagerLifecycleView,
    runtime: AgentManagerRuntimeView,
) -> AgentManagerDerivedStatus:
    if lifecycle.match_resolution is LifecycleMatchResolution.AMBIGUOUS:
        return AgentManagerDerivedStatus.DEGRADED
    if runtime.serving:
        return AgentManagerDerivedStatus.SERVING
    if lifecycle.enabled_in_desired_state and runtime.included_in_candidate_revision:
        return AgentManagerDerivedStatus.READY_FOR_REVISION
    if lifecycle.enabled_in_desired_state:
        return AgentManagerDerivedStatus.ENABLED
    if lifecycle.bound:
        return AgentManagerDerivedStatus.BOUND
    if lifecycle.installed:
        return AgentManagerDerivedStatus.INSTALLED
    if lifecycle.match_resolution is not LifecycleMatchResolution.UNRESOLVED:
        return AgentManagerDerivedStatus.DISCOVERABLE
    return AgentManagerDerivedStatus.UNAVAILABLE


def _derive_availability(
    *,
    lifecycle: AgentManagerLifecycleView,
    runtime: AgentManagerRuntimeView,
) -> AgentManagerAvailabilityView:
    installable = not lifecycle.installed
    bindable = lifecycle.installed and not lifecycle.bound
    activatable = (
        lifecycle.enabled_in_desired_state
        and runtime.included_in_candidate_revision
        and not runtime.serving
    )
    return AgentManagerAvailabilityView(
        installable=installable,
        bindable=bindable,
        activatable=activatable,
    )


def _to_catalog_filters(
    filters: AgentManagerListFilters | None,
) -> CatalogEntryFilters | None:
    if filters is None:
        return None
    if (
        filters.category is None
        and filters.publisher is None
        and filters.catalog_source_id is None
        and filters.provider_kind is None
    ):
        return None
    return CatalogEntryFilters(
        category=filters.category,
        publisher=filters.publisher,
    )


def _matches_filters(
    item: AgentManagerEntry,
    filters: AgentManagerListFilters | None,
    application_id: str,
) -> bool:
    del application_id
    if filters is None:
        return True
    if filters.catalog_source_id is not None:
        source = item.identity.catalog_source
        if source is None or source.catalog_source_id != filters.catalog_source_id:
            return False
    if filters.provider_kind is not None:
        kind = item.discovery.provider_kind
        if kind is None or kind != filters.provider_kind:
            return False
    if filters.category is not None:
        if filters.category not in item.discovery.categories:
            return False
    if filters.publisher is not None:
        if item.identity.publisher != filters.publisher:
            return False
    if filters.installed is not None:
        if item.lifecycle.installed != filters.installed:
            return False
    if filters.bound is not None:
        if item.lifecycle.bound != filters.bound:
            return False
    if filters.enabled is not None:
        if item.lifecycle.enabled_in_desired_state != filters.enabled:
            return False
    if filters.capability is not None:
        if filters.capability not in item.discovery.capabilities:
            return False
    return True
