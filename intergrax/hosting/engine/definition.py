# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable hosted application definition resolution (APP-HOST-2B)."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from intergrax.hosting.contracts.components import HostedApplicationComponentRegistration
from intergrax.hosting.contracts.events import HostedApplicationEventSubscription
from intergrax.hosting.contracts.hooks import (
    HOSTED_APPLICATION_HOOK_POINT_ORDER,
    HostedApplicationHook,
    HostedApplicationHookPoint,
)
from intergrax.hosting.contracts.policies import (
    ComponentFailureAction,
    ComponentFailurePolicy,
    HookFailurePolicy,
    InstancePolicy,
    LifecyclePolicy,
    RestartPolicy,
    ShutdownPolicy,
)
from intergrax.hosting.contracts.profile import (
    HostedApplicationProfile,
    HostedApplicationProfilePublicView,
)
from intergrax.hosting.contracts.public_data import public_json_digest
from intergrax.hosting.engine.ports import HostedApplicationRuntime
from intergrax.hosting.errors import HostedApplicationConfigurationError, HostedApplicationDefinitionError

ApplicationFactory = Callable[..., HostedApplicationRuntime | object]

_UNSUPPORTED_COMPONENT_ACTIONS = frozenset(
    {
        ComponentFailureAction.RESTART_COMPONENT,
        ComponentFailureAction.REQUEST_PROCESS_RESTART,
    }
)


class HostedApplicationDefinitionPublicView(BaseModel):
    """Safe public projection of a resolved hosted application definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: HostedApplicationProfilePublicView
    profile_digest: str
    definition_digest: str
    hook_ids_by_point: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    event_subscription_ids: tuple[str, ...] = ()
    enabled_component_ids: tuple[str, ...] = ()
    disabled_component_ids: tuple[str, ...] = ()
    component_dependency_levels: tuple[tuple[str, ...], ...] = ()
    component_start_order: tuple[str, ...] = ()
    component_stop_order: tuple[str, ...] = ()
    pre_runtime_component_ids: tuple[str, ...] = ()
    post_runtime_component_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedComponentRegistration:
    """Internal resolved component registration with graph metadata."""

    registration: HostedApplicationComponentRegistration
    declaration_index: int
    dependency_level: int


@dataclass(frozen=True, slots=True)
class ResolvedEventSubscription:
    """Internal resolved event subscription with declaration ordering metadata."""

    subscription: HostedApplicationEventSubscription
    declaration_index: int


@dataclass(frozen=True, slots=True)
class HostedApplicationDefinition:
    """Immutable runtime composition derived from a hosted application profile."""

    application_id: str
    profile_public_snapshot: HostedApplicationProfilePublicView
    profile_digest: str
    definition_digest: str
    application_factory: ApplicationFactory
    lifecycle_policy: LifecyclePolicy
    shutdown_policy: ShutdownPolicy
    restart_policy: RestartPolicy
    component_failure_policy: ComponentFailurePolicy
    hook_failure_policy: HookFailurePolicy
    instance_policy: InstancePolicy
    hook_registrations: Mapping[HostedApplicationHookPoint, tuple[HostedApplicationHook, ...]]
    event_subscriptions: tuple[ResolvedEventSubscription, ...]
    enabled_components: Mapping[str, ResolvedComponentRegistration]
    disabled_components: tuple[HostedApplicationComponentRegistration, ...]
    component_dependency_levels: tuple[tuple[str, ...], ...]
    component_start_order: tuple[str, ...]
    component_stop_order: tuple[str, ...]
    pre_runtime_component_ids: tuple[str, ...]
    post_runtime_component_ids: tuple[str, ...]

    def public_view(self) -> HostedApplicationDefinitionPublicView:
        hook_ids_by_point = {
            point.value: tuple(hook.hook_id for hook in self.hook_registrations.get(point, ()))
            for point in HOSTED_APPLICATION_HOOK_POINT_ORDER
        }
        return HostedApplicationDefinitionPublicView(
            profile=self.profile_public_snapshot,
            profile_digest=self.profile_digest,
            definition_digest=self.definition_digest,
            hook_ids_by_point=hook_ids_by_point,
            event_subscription_ids=tuple(
                resolved.subscription.subscription_id for resolved in self.event_subscriptions
            ),
            enabled_component_ids=self.component_start_order,
            disabled_component_ids=tuple(
                registration.component_id or ""
                for registration in self.disabled_components
            ),
            component_dependency_levels=self.component_dependency_levels,
            component_start_order=self.component_start_order,
            component_stop_order=self.component_stop_order,
            pre_runtime_component_ids=self.pre_runtime_component_ids,
            post_runtime_component_ids=self.post_runtime_component_ids,
        )


def _declaration_index_sort_key(
    component_id: str,
    enabled: Mapping[str, ResolvedComponentRegistration],
) -> tuple[int, str]:
    return (enabled[component_id].declaration_index, component_id)


def _reject_unsupported_component_actions(profile: HostedApplicationProfile) -> None:
    for registration in profile.components:
        action = registration.failure_action
        if action is not None and action in _UNSUPPORTED_COMPONENT_ACTIONS:
            raise HostedApplicationConfigurationError(
                f"component failure action not supported in W2: {action.value}"
            )


def _validate_component_graph(
    components: tuple[HostedApplicationComponentRegistration, ...],
) -> dict[str, ResolvedComponentRegistration]:
    enabled: dict[str, ResolvedComponentRegistration] = {}
    disabled: list[HostedApplicationComponentRegistration] = []
    all_ids: dict[str, HostedApplicationComponentRegistration] = {}

    for index, registration in enumerate(components):
        component_id = registration.component_id or ""
        if component_id in all_ids:
            raise HostedApplicationDefinitionError(f"duplicate component_id: {component_id}")
        all_ids[component_id] = registration
        if registration.enabled:
            enabled[component_id] = ResolvedComponentRegistration(
                registration=registration,
                declaration_index=index,
                dependency_level=0,
            )
        else:
            disabled.append(registration)

    for component_id, resolved in enabled.items():
        registration = resolved.registration
        for dependency in registration.dependencies:
            if dependency not in all_ids:
                raise HostedApplicationDefinitionError(
                    f"missing component dependency: {dependency}"
                )
            dep_registration = all_ids[dependency]
            if not dep_registration.enabled:
                raise HostedApplicationDefinitionError(
                    f"enabled component {component_id} depends on disabled component {dependency}"
                )
            if dependency == component_id:
                raise HostedApplicationDefinitionError("component cannot depend on itself")

    indegree: dict[str, int] = {component_id: 0 for component_id in enabled}
    dependents: dict[str, list[str]] = {component_id: [] for component_id in enabled}
    for component_id, resolved in enabled.items():
        for dependency in resolved.registration.dependencies:
            indegree[component_id] += 1
            dependents[dependency].append(component_id)

    levels: list[list[str]] = []
    current_level = sorted(
        (component_id for component_id, degree in indegree.items() if degree == 0),
        key=lambda cid: _declaration_index_sort_key(cid, enabled),
    )
    visited = 0
    while current_level:
        levels.append(current_level)
        visited += len(current_level)
        next_level: list[str] = []
        for component_id in current_level:
            for dependent in sorted(
                dependents[component_id],
                key=lambda cid: _declaration_index_sort_key(cid, enabled),
            ):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    next_level.append(dependent)
        current_level = sorted(
            next_level,
            key=lambda cid: _declaration_index_sort_key(cid, enabled),
        )

    if visited != len(enabled):
        cycle_nodes = _find_cycle_path(enabled)
        raise HostedApplicationDefinitionError(
            f"component dependency cycle detected: {' -> '.join(cycle_nodes)}"
        )

    rebuilt_enabled: dict[str, ResolvedComponentRegistration] = {}
    for level_index, level in enumerate(levels):
        for component_id in level:
            resolved = enabled[component_id]
            rebuilt_enabled[component_id] = ResolvedComponentRegistration(
                registration=resolved.registration,
                declaration_index=resolved.declaration_index,
                dependency_level=level_index,
            )

    return rebuilt_enabled


def _find_cycle_path(enabled: Mapping[str, ResolvedComponentRegistration]) -> list[str]:
    visited: set[str] = set()
    stack: set[str] = set()
    parent: dict[str, str | None] = {component_id: None for component_id in enabled}

    def dfs(node: str) -> list[str] | None:
        visited.add(node)
        stack.add(node)
        registration = enabled[node].registration
        for dependency in sorted(registration.dependencies):
            if dependency not in enabled:
                continue
            if dependency not in visited:
                parent[dependency] = node
                cycle = dfs(dependency)
                if cycle is not None:
                    return cycle
            elif dependency in stack:
                path = [dependency, node]
                current = node
                while parent.get(current) not in {None, dependency}:
                    current = parent[current] or current
                    if current in path:
                        break
                    path.append(current)
                path.reverse()
                return path
        stack.remove(node)
        return None

    for component_id in sorted(enabled):
        if component_id not in visited:
            cycle = dfs(component_id)
            if cycle is not None:
                return cycle
    return sorted(enabled.keys())


def _compute_pre_runtime_closure(
    enabled: Mapping[str, ResolvedComponentRegistration],
) -> tuple[str, ...]:
    required_roots = sorted(
        (component_id for component_id, resolved in enabled.items() if resolved.registration.required),
        key=lambda cid: _declaration_index_sort_key(cid, enabled),
    )
    closure: set[str] = set()
    queue: deque[str] = deque(required_roots)
    while queue:
        component_id = queue.popleft()
        if component_id in closure:
            continue
        if component_id not in enabled:
            continue
        closure.add(component_id)
        for dependency in enabled[component_id].registration.dependencies:
            queue.append(dependency)
    return tuple(
        sorted(closure, key=lambda cid: _declaration_index_sort_key(cid, enabled))
    )


def _build_orders(
    enabled: Mapping[str, ResolvedComponentRegistration],
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...], tuple[str, ...]]:
    max_level = max((resolved.dependency_level for resolved in enabled.values()), default=-1)
    levels: list[list[str]] = []
    for level in range(max_level + 1):
        level_ids = sorted(
            (
                component_id
                for component_id, resolved in enabled.items()
                if resolved.dependency_level == level
            ),
            key=lambda cid: _declaration_index_sort_key(cid, enabled),
        )
        if level_ids:
            levels.append(level_ids)
    start_order = tuple(component_id for level in levels for component_id in level)
    stop_order = tuple(reversed(start_order))
    return tuple(tuple(level) for level in levels), start_order, stop_order


def resolve_hosted_application_definition(
    profile: HostedApplicationProfile,
) -> HostedApplicationDefinition:
    """Resolve an immutable hosted application definition from a profile."""
    _reject_unsupported_component_actions(profile)
    profile_digest = profile.profile_digest()
    enabled = _validate_component_graph(profile.components)
    disabled = tuple(
        registration for registration in profile.components if not registration.enabled
    )
    levels, start_order, stop_order = _build_orders(enabled)
    pre_runtime = _compute_pre_runtime_closure(enabled)
    pre_runtime_set = set(pre_runtime)
    post_runtime = tuple(
        component_id for component_id in start_order if component_id not in pre_runtime_set
    )

    hook_registrations = MappingProxyType(
        {
            point: profile.hooks.hooks_for_point(point)
            for point in HOSTED_APPLICATION_HOOK_POINT_ORDER
        }
    )

    event_subscriptions = tuple(
        ResolvedEventSubscription(subscription=subscription, declaration_index=index)
        for index, subscription in enumerate(profile.event_subscriptions)
    )

    public_payload = {
        "profile_digest": profile_digest,
        "hook_ids": [
            hook.hook_id
            for point in HOSTED_APPLICATION_HOOK_POINT_ORDER
            for hook in hook_registrations[point]
        ],
        "subscription_ids": [
            resolved.subscription.subscription_id for resolved in event_subscriptions
        ],
        "enabled_component_ids": list(start_order),
        "disabled_component_ids": [
            registration.component_id or "" for registration in disabled
        ],
        "component_dependency_levels": [list(level) for level in levels],
        "pre_runtime_component_ids": list(pre_runtime),
        "post_runtime_component_ids": list(post_runtime),
    }
    definition_digest = public_json_digest(public_payload)
    profile_public_snapshot = HostedApplicationProfilePublicView.model_validate(
        profile.public_view().model_dump(mode="json"),
    )

    return HostedApplicationDefinition(
        application_id=profile.application_id,
        profile_public_snapshot=profile_public_snapshot,
        profile_digest=profile_digest,
        definition_digest=definition_digest,
        application_factory=profile.application_factory,
        lifecycle_policy=profile.lifecycle,
        shutdown_policy=profile.shutdown,
        restart_policy=profile.restart,
        component_failure_policy=profile.component_failure,
        hook_failure_policy=profile.hook_failure,
        instance_policy=profile.instance,
        hook_registrations=hook_registrations,
        event_subscriptions=event_subscriptions,
        enabled_components=MappingProxyType(enabled),
        disabled_components=disabled,
        component_dependency_levels=levels,
        component_start_order=start_order,
        component_stop_order=stop_order,
        pre_runtime_component_ids=pre_runtime,
        post_runtime_component_ids=post_runtime,
    )
