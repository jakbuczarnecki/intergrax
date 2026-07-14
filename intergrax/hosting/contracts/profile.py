# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application profile composition root (APP-HOST-1A.1 / APP-HOST-1A.2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from intergrax.hosting.contracts.components import (
    HostedApplicationComponentPublicDescriptor,
    HostedApplicationComponentRegistration,
)
from intergrax.hosting.contracts.events import (
    HostedApplicationEventSubscription,
    HostedApplicationEventSubscriptionPublicDescriptor,
)
from intergrax.hosting.contracts.hooks import (
    HOSTED_APPLICATION_HOOK_POINT_ORDER,
    HostedApplicationHookPublicDescriptor,
    HostedApplicationHooks,
)
from intergrax.hosting.contracts.identity import (
    HostedApplicationIdentity,
    normalize_application_id,
    validate_application_factory_id,
)
from intergrax.hosting.contracts.policies import (
    ComponentFailurePolicy,
    HookFailurePolicy,
    InstancePolicy,
    LifecyclePolicy,
    RestartPolicy,
    ShutdownPolicy,
)
from intergrax.hosting.contracts.public_data import (
    derive_stable_callable_id,
    normalize_public_json_mapping,
    public_json_digest,
)

HOSTED_APPLICATION_PROFILE_SPEC_VERSION = "1.0"


def derive_stable_application_factory_id(factory: Callable[..., object]) -> str:
    """Derive a stable factory identifier from a module-level callable."""
    return derive_stable_callable_id(factory, field_name="application_factory_id")


class HostedApplicationProfilePublicView(BaseModel):
    """Explicit public projection of a hosted application profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: Literal["1.0"]
    identity: HostedApplicationIdentity
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    hooks: tuple[HostedApplicationHookPublicDescriptor, ...] = ()
    components: tuple[HostedApplicationComponentPublicDescriptor, ...] = ()
    lifecycle: LifecyclePolicy = Field(default_factory=LifecyclePolicy.standard)
    shutdown: ShutdownPolicy = Field(default_factory=ShutdownPolicy.standard)
    restart: RestartPolicy = Field(default_factory=RestartPolicy.on_failure)
    component_failure: ComponentFailurePolicy = Field(default_factory=ComponentFailurePolicy.standard)
    hook_failure: HookFailurePolicy = Field(default_factory=HookFailurePolicy.standard)
    instance: InstancePolicy = Field(default_factory=InstancePolicy.standard)
    event_subscriptions: tuple[HostedApplicationEventSubscriptionPublicDescriptor, ...] = ()

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return normalize_public_json_mapping(value)


class HostedApplicationProfile(BaseModel):
    """Hosted application profile composition root."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    application_id: str
    application_factory: SkipJsonSchema[Callable[..., object]] = Field(exclude=True, repr=False)
    application_factory_id: str | None = None
    spec_version: Literal["1.0"] = HOSTED_APPLICATION_PROFILE_SPEC_VERSION
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    hooks: HostedApplicationHooks = Field(default_factory=HostedApplicationHooks)
    components: tuple[HostedApplicationComponentRegistration, ...] = ()
    lifecycle: LifecyclePolicy = Field(default_factory=LifecyclePolicy.standard)
    shutdown: ShutdownPolicy = Field(default_factory=ShutdownPolicy.standard)
    restart: RestartPolicy = Field(default_factory=RestartPolicy.on_failure)
    component_failure: ComponentFailurePolicy = Field(default_factory=ComponentFailurePolicy.standard)
    hook_failure: HookFailurePolicy = Field(default_factory=HookFailurePolicy.standard)
    instance: InstancePolicy = Field(default_factory=InstancePolicy.standard)
    event_subscriptions: tuple[HostedApplicationEventSubscription, ...] = ()

    @field_validator("application_id")
    @classmethod
    def _validate_application_id(cls, value: str) -> str:
        return normalize_application_id(value)

    @field_validator("application_factory_id")
    @classmethod
    def _validate_explicit_application_factory_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_application_factory_id(value)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return normalize_public_json_mapping(value)

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> HostedApplicationProfile:
        if self.application_factory_id is None:
            derived_factory_id = derive_stable_application_factory_id(self.application_factory)
            object.__setattr__(self, "application_factory_id", derived_factory_id)

        seen_component_ids: set[str] = set()
        for registration in self.components:
            component_id = registration.component_id or ""
            if component_id in seen_component_ids:
                raise ValueError(f"duplicate component_id: {component_id}")
            seen_component_ids.add(component_id)

        seen_subscription_ids: set[str] = set()
        for subscription in self.event_subscriptions:
            if subscription.subscription_id in seen_subscription_ids:
                raise ValueError(f"duplicate subscription_id: {subscription.subscription_id}")
            seen_subscription_ids.add(subscription.subscription_id)

        return self

    @property
    def identity(self) -> HostedApplicationIdentity:
        factory_id = self.application_factory_id
        if factory_id is None:
            raise RuntimeError("hosted application profile is missing application_factory_id")
        return HostedApplicationIdentity(
            application_id=self.application_id,
            application_factory_id=factory_id,
        )

    def _hook_public_descriptors_declaration_order(
        self,
    ) -> tuple[HostedApplicationHookPublicDescriptor, ...]:
        descriptors: list[HostedApplicationHookPublicDescriptor] = []
        for point in HOSTED_APPLICATION_HOOK_POINT_ORDER:
            for index, hook in enumerate(self.hooks.hooks_for_point(point)):
                descriptors.append(
                    hook.public_descriptor(hook_point=point, declaration_index=index),
                )
        return tuple(descriptors)

    def public_view(self) -> HostedApplicationProfilePublicView:
        return HostedApplicationProfilePublicView(
            spec_version=self.spec_version,
            identity=self.identity,
            metadata=self.metadata,
            hooks=self.hooks.flattened_public_descriptors(),
            components=tuple(
                registration.public_descriptor(declaration_index=index)
                for index, registration in enumerate(self.components)
            ),
            lifecycle=self.lifecycle,
            shutdown=self.shutdown,
            restart=self.restart.to_public_policy(),
            component_failure=self.component_failure,
            hook_failure=self.hook_failure,
            instance=self.instance,
            event_subscriptions=tuple(
                subscription.public_descriptor(declaration_index=index)
                for index, subscription in enumerate(self.event_subscriptions)
            ),
        )

    def profile_digest(self) -> str:
        public_view = self.public_view()
        payload = public_view.model_dump(mode="json")
        payload["hooks"] = [
            item.model_dump(mode="json")
            for item in self._hook_public_descriptors_declaration_order()
        ]
        return public_json_digest(payload)
