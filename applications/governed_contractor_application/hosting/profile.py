# © Artur Czarnecki. All rights reserved.

"""GCA HostedApplicationProfile builder (Package B strict production)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingSession,
)
from intergrax.hosting.contracts.components import HostedApplicationComponentRegistration
from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.contracts.hooks import HostedApplicationHook, HostedApplicationHooks
from intergrax.hosting.contracts.policies import (
    InstancePolicy,
    LifecyclePolicy,
    RestartPolicy,
    ShutdownPolicy,
)
from intergrax.hosting.contracts.profile import HostedApplicationProfile
from intergrax.hosting.engine.ports import HostedApplicationRuntime

from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
    compose_governed_contractor_host_runtime,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.hosting.boundary import (
    GOVERNED_CONTRACTOR_BEFORE_READY_HANDLER_ID,
    GOVERNED_CONTRACTOR_BEFORE_READY_HOOK_ID,
    GOVERNED_CONTRACTOR_HOSTING_BOUNDARY_COMPONENT_ID,
    GOVERNED_CONTRACTOR_HOSTING_BOUNDARY_COMPONENT_TYPE_ID,
    GOVERNED_CONTRACTOR_HOSTING_SOURCE_ID,
    _GovernedContractorHostingBoundary,
)
from governed_contractor_application.hosting.runtime import _GovernedContractorHostedRuntime

GOVERNED_CONTRACTOR_HOSTED_FACTORY_ID = "governed_contractor_application.hosting.runtime_factory.v1"


@dataclass(frozen=True, slots=True)
class _GovernedContractorRuntimeFactory:
    """Immutable runtime factory capturing one session/settings/bind snapshot."""

    session: StrictProductionServingSession
    settings: GovernedContractorBackendSettings
    bind_host: str
    bind_port: int
    host_runtime: GovernedContractorHostRuntimeComposition | None

    def __call__(self, context: HostedApplicationContext) -> HostedApplicationRuntime:
        return _GovernedContractorHostedRuntime(
            hosted_context=context,
            session=self.session,
            settings=self.settings,
            bind_host=self.bind_host,
            bind_port=self.bind_port,
            host_runtime=self.host_runtime,
        )


def build_governed_contractor_hosted_profile(
    *,
    session: StrictProductionServingSession,
    settings: GovernedContractorBackendSettings | None = None,
    host_runtime: GovernedContractorHostRuntimeComposition | None = None,
) -> HostedApplicationProfile:
    """Build the GCA-owned hosted profile for an opened strict production session."""
    resolved_settings = settings or GovernedContractorBackendSettings.from_env()
    resolved_host_runtime = host_runtime or compose_governed_contractor_host_runtime(
        resolved_settings,
    )
    raw_host = os.environ.get("GOVERNED_CONTRACTOR_BACKEND_BIND_HOST", "127.0.0.1")
    bind_host = raw_host.strip() or "127.0.0.1"
    bind_port = resolved_settings.backend_port
    runtime_factory = _GovernedContractorRuntimeFactory(
        session=session,
        settings=resolved_settings,
        bind_host=bind_host,
        bind_port=bind_port,
        host_runtime=resolved_host_runtime,
    )
    boundary = _GovernedContractorHostingBoundary()
    before_ready_hook = HostedApplicationHook(
        hook_id=GOVERNED_CONTRACTOR_BEFORE_READY_HOOK_ID,
        handler=boundary.mark_before_ready,
        handler_id=GOVERNED_CONTRACTOR_BEFORE_READY_HANDLER_ID,
        priority=0,
        source_id=GOVERNED_CONTRACTOR_HOSTING_SOURCE_ID,
    )
    return HostedApplicationProfile(
        application_id="governed_contractor",
        application_factory=runtime_factory,
        application_factory_id=GOVERNED_CONTRACTOR_HOSTED_FACTORY_ID,
        metadata={
            "product_id": "governed_contractor",
            "product_tier": "tier3",
            "runtime_kind": "fastapi_uvicorn",
            "serving_mode": "package_b_strict_production",
        },
        hooks=HostedApplicationHooks(
            before_ready=(before_ready_hook,),
        ),
        components=(
            HostedApplicationComponentRegistration(
                component=boundary,
                component_id=GOVERNED_CONTRACTOR_HOSTING_BOUNDARY_COMPONENT_ID,
                component_type_id=GOVERNED_CONTRACTOR_HOSTING_BOUNDARY_COMPONENT_TYPE_ID,
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
