# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Application binding lifecycle domain service (AGENT_DISTRIBUTION §12, AP-4)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intergrax.agent_distribution.binding import (
    AgentBindingFactoryReference,
    AgentBindingPolicyOverrides,
    ApplicationAgentBinding,
)
from intergrax.agent_distribution.errors import (
    AgentDistributionNotFoundError,
    BindingLifecycleError,
    BindingRevisionConflict,
)
from intergrax.agent_distribution.events import TransitionResult, distribution_event
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.stores import ApplicationAgentBindingStore


class BindingService:
    """Transactional binding lifecycle operations — fail closed."""

    def __init__(
        self,
        binding_store: ApplicationAgentBindingStore,
        installation_service: InstallationService,
    ) -> None:
        self._binding_store = binding_store
        self._installation_service = installation_service

    def create_binding(
        self,
        *,
        application_binding_id: str,
        application_id: str,
        application_environment_id: str,
        logical_agent_id: str,
        installation_slot_id: str,
        config: Mapping[str, Any] | None = None,
        secret_refs: tuple[str, ...] = (),
        policy_overrides: AgentBindingPolicyOverrides | None = None,
        factory_reference: AgentBindingFactoryReference | None = None,
        manifest_origin_ref: str | None = None,
        builtin_package_ref: str | None = None,
        enablement: bool = False,
        default_agent: bool | None = None,
    ) -> TransitionResult[ApplicationAgentBinding]:
        active_installation_id = self._resolve_active_installation_id(
            installation_slot_id=installation_slot_id,
            builtin_package_ref=builtin_package_ref,
            require_for_enablement=enablement,
        )
        binding = ApplicationAgentBinding(
            application_binding_id=application_binding_id,
            application_id=application_id,
            application_environment_id=application_environment_id,
            logical_agent_id=logical_agent_id,
            installation_slot_id=installation_slot_id,
            active_installation_id=active_installation_id,
            builtin_package_ref=builtin_package_ref,
            enablement=enablement,
            default_agent=default_agent,
            config=config or {},
            secret_refs=secret_refs,
            policy_overrides=policy_overrides,
            factory_reference=factory_reference,
            manifest_origin_ref=manifest_origin_ref,
            binding_revision=0,
        )
        persisted = self._binding_store.persist_binding(binding, expected_revision=None)
        return TransitionResult(
            value=persisted,
            events=(
                distribution_event(
                    "binding.created",
                    application_binding_id,
                    installation_slot_id=installation_slot_id,
                ),
            ),
        )

    def update_config(
        self,
        application_binding_id: str,
        config: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> TransitionResult[ApplicationAgentBinding]:
        return self._update_binding(
            application_binding_id,
            expected_revision=expected_revision,
            updates={"config": config},
            event_type="binding.updated",
        )

    def update_secret_refs(
        self,
        application_binding_id: str,
        secret_refs: tuple[str, ...],
        *,
        expected_revision: int,
    ) -> TransitionResult[ApplicationAgentBinding]:
        return self._update_binding(
            application_binding_id,
            expected_revision=expected_revision,
            updates={"secret_refs": secret_refs},
            event_type="binding.updated",
        )

    def update_policy_overrides(
        self,
        application_binding_id: str,
        policy_overrides: AgentBindingPolicyOverrides | None,
        *,
        expected_revision: int,
    ) -> TransitionResult[ApplicationAgentBinding]:
        return self._update_binding(
            application_binding_id,
            expected_revision=expected_revision,
            updates={"policy_overrides": policy_overrides},
            event_type="binding.updated",
        )

    def update_default_agent(
        self,
        application_binding_id: str,
        default_agent: bool | None,
        *,
        expected_revision: int,
    ) -> TransitionResult[ApplicationAgentBinding]:
        return self._update_binding(
            application_binding_id,
            expected_revision=expected_revision,
            updates={"default_agent": default_agent},
            event_type="binding.updated",
        )

    def enable(
        self,
        application_binding_id: str,
        *,
        expected_revision: int,
    ) -> TransitionResult[ApplicationAgentBinding]:
        binding = self._require_binding(application_binding_id)
        if binding.tombstone:
            raise BindingLifecycleError("tombstoned bindings cannot be enabled")
        active_installation_id = self._resolve_active_installation_id(
            installation_slot_id=binding.installation_slot_id,
            builtin_package_ref=binding.builtin_package_ref,
            require_for_enablement=True,
        )
        return self._update_binding(
            application_binding_id,
            expected_revision=expected_revision,
            updates={
                "enablement": True,
                "active_installation_id": active_installation_id,
            },
            event_type="binding.enablement_changed",
        )

    def disable(
        self,
        application_binding_id: str,
        *,
        expected_revision: int,
    ) -> TransitionResult[ApplicationAgentBinding]:
        return self._update_binding(
            application_binding_id,
            expected_revision=expected_revision,
            updates={"enablement": False},
            event_type="binding.enablement_changed",
        )

    def tombstone(
        self,
        application_binding_id: str,
        *,
        expected_revision: int,
    ) -> TransitionResult[ApplicationAgentBinding]:
        return self._update_binding(
            application_binding_id,
            expected_revision=expected_revision,
            updates={"tombstone": True, "enablement": False},
            event_type="binding.updated",
        )

    def list_bindings_for_environment(
        self,
        application_environment_id: str,
    ) -> list[ApplicationAgentBinding]:
        return self._binding_store.list_bindings_for_environment(application_environment_id)

    def refresh_active_installation_for_slot(
        self,
        installation_slot_id: str,
        *,
        prior_active_installation_id: str,
        next_active_installation_id: str,
    ) -> list[TransitionResult[ApplicationAgentBinding]]:
        results: list[TransitionResult[ApplicationAgentBinding]] = []
        bindings = [
            binding
            for binding in self._binding_store.list_bindings_for_slot(installation_slot_id)
            if not binding.tombstone
        ]
        for binding in bindings:
            if binding.active_installation_id != prior_active_installation_id:
                continue
            updated = binding.model_copy(
                update={
                    "active_installation_id": next_active_installation_id,
                    "binding_revision": binding.binding_revision + 1,
                }
            )
            persisted = self._binding_store.persist_binding(
                updated,
                expected_revision=binding.binding_revision,
            )
            results.append(
                TransitionResult(
                    value=persisted,
                    events=(
                        distribution_event(
                            "binding.updated",
                            binding.application_binding_id,
                            installation_slot_id=installation_slot_id,
                            active_installation_id=next_active_installation_id,
                        ),
                    ),
                )
            )
        return results

    def _update_binding(
        self,
        application_binding_id: str,
        *,
        expected_revision: int,
        updates: dict[str, Any],
        event_type: str,
    ) -> TransitionResult[ApplicationAgentBinding]:
        binding = self._require_binding(application_binding_id)
        if binding.tombstone and updates.get("enablement") is True:
            raise BindingLifecycleError("tombstoned bindings cannot be enabled")
        next_revision = expected_revision + 1
        updated = binding.model_copy(update={**updates, "binding_revision": next_revision})
        try:
            persisted = self._binding_store.persist_binding(
                updated,
                expected_revision=expected_revision,
            )
        except BindingRevisionConflict:
            raise
        return TransitionResult(
            value=persisted,
            events=(distribution_event(event_type, application_binding_id),),
        )

    def _require_binding(self, application_binding_id: str) -> ApplicationAgentBinding:
        binding = self._binding_store.get_binding(application_binding_id)
        if binding is None:
            raise AgentDistributionNotFoundError(f"binding {application_binding_id} was not found")
        return binding

    def _resolve_active_installation_id(
        self,
        *,
        installation_slot_id: str,
        builtin_package_ref: str | None,
        require_for_enablement: bool,
    ) -> str | None:
        active = self._installation_service.resolve_active_for_slot(installation_slot_id)
        if active is not None:
            return active.installation_id
        if builtin_package_ref is not None:
            return None
        if require_for_enablement:
            raise BindingLifecycleError(
                "enablement requires resolvable active installation or builtin_package_ref"
            )
        return None
