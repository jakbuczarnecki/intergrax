# © Artur Czarnecki. All rights reserved.

"""LKW HostedApplicationProfile builder (APP-HOST-8A/8C)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.hosting.contracts.components import (
    HostedApplicationComponentRegistration,
)
from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.contracts.hooks import (
    HostedApplicationHook,
    HostedApplicationHooks,
)
from intergrax.hosting.contracts.policies import (
    InstancePolicy,
    LifecyclePolicy,
    RestartPolicy,
    ShutdownPolicy,
)
from intergrax.hosting.contracts.profile import HostedApplicationProfile
from intergrax.hosting.engine.ports import HostedApplicationRuntime
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting.boundary import (
    LOCAL_WORKSPACE_BEFORE_READY_HANDLER_ID,
    LOCAL_WORKSPACE_BEFORE_READY_HOOK_ID,
    LOCAL_WORKSPACE_HOSTING_BOUNDARY_COMPONENT_ID,
    LOCAL_WORKSPACE_HOSTING_BOUNDARY_COMPONENT_TYPE_ID,
    LOCAL_WORKSPACE_HOSTING_SOURCE_ID,
    _LocalWorkspaceHostingBoundary,
)
from local_workspace_application.hosting.runtime import _LocalWorkspaceHostedRuntime

LOCAL_WORKSPACE_HOSTED_FACTORY_ID = (
    "local_workspace_application.hosting.runtime_factory.v1"
)


@dataclass(frozen=True, slots=True)
class _LocalWorkspaceRuntimeFactory:
    """Immutable runtime factory capturing one settings/bind snapshot."""

    settings: LocalWorkspaceBackendSettings
    bind_host: str
    bind_port: int
    process_composition: ProductionProcessComposition

    def __call__(self, context: HostedApplicationContext) -> HostedApplicationRuntime:
        return _LocalWorkspaceHostedRuntime(
            hosted_context=context,
            settings=self.settings,
            bind_host=self.bind_host,
            bind_port=self.bind_port,
            process_composition=self.process_composition,
        )


def build_local_workspace_hosted_profile(
    *,
    process_composition: ProductionProcessComposition,
    settings: LocalWorkspaceBackendSettings | None = None,
) -> HostedApplicationProfile:
    """Build the LKW-owned hosted application profile."""
    resolved_settings = cast(
        LocalWorkspaceBackendSettings,
        settings if settings is not None else LocalWorkspaceBackendSettings.from_env(),
    )
    raw_host = os.environ.get("LOCAL_WORKSPACE_BACKEND_HOST", "127.0.0.1")
    bind_host = raw_host.strip() or "127.0.0.1"
    bind_port = resolved_settings.backend_port
    runtime_factory = _LocalWorkspaceRuntimeFactory(
        settings=resolved_settings,
        bind_host=bind_host,
        bind_port=bind_port,
        process_composition=process_composition,
    )
    boundary = _LocalWorkspaceHostingBoundary()
    before_ready_hook = HostedApplicationHook(
        hook_id=LOCAL_WORKSPACE_BEFORE_READY_HOOK_ID,
        handler=boundary.mark_before_ready,
        handler_id=LOCAL_WORKSPACE_BEFORE_READY_HANDLER_ID,
        priority=0,
        source_id=LOCAL_WORKSPACE_HOSTING_SOURCE_ID,
    )
    return HostedApplicationProfile(
        application_id="local_workspace",
        application_factory=runtime_factory,
        application_factory_id=LOCAL_WORKSPACE_HOSTED_FACTORY_ID,
        metadata={
            "product_id": "local_workspace",
            "product_tier": "tier3",
            "runtime_kind": "fastapi_uvicorn",
        },
        hooks=HostedApplicationHooks(
            before_ready=(before_ready_hook,),
        ),
        components=(
            HostedApplicationComponentRegistration(
                component=boundary,
                component_id=LOCAL_WORKSPACE_HOSTING_BOUNDARY_COMPONENT_ID,
                component_type_id=LOCAL_WORKSPACE_HOSTING_BOUNDARY_COMPONENT_TYPE_ID,
                enabled=True,
                required=True,
                dependencies=(),
            ),
        ),
        lifecycle=LifecyclePolicy.standard(),
        shutdown=ShutdownPolicy.standard(),
        restart=RestartPolicy.on_failure(max_attempts=3),
        instance=InstancePolicy.standard(),
    )
