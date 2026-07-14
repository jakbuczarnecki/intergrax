# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared helpers for hosting unit tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from intergrax.hosting import (
    ComponentFailurePolicy,
    HookFailurePolicy,
    HostedApplicationComponentHealth,
    HostedApplicationComponentRegistration,
    HostedApplicationComponentState,
    HostedApplicationContext,
    HostedApplicationEvent,
    HostedApplicationEventSubscription,
    HostedApplicationEventType,
    HostedApplicationHook,
    HostedApplicationHooks,
    HostedApplicationLifecycleSnapshot,
    HostedApplicationLifecycleState,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
    HostedApplicationProfile,
    InstancePolicy,
    LifecyclePolicy,
    RestartPolicy,
    ShutdownPolicy,
)
from intergrax.hosting.contracts.context import (
    HostedApplicationClock,
    HostedApplicationEventPublisher,
    HostedApplicationLogger,
)
from intergrax.hosting.contracts.lifecycle import (
    HostedApplicationEffectiveControlRequest,
    HostedApplicationLifecycleSnapshotProvider,
    HostedApplicationShutdownCoordinator,
    HostedApplicationShutdownRequestSnapshot,
)
from intergrax.hosting.services import HostedApplicationServiceRegistry
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory


class _FixedClock(HostedApplicationClock):
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


class _NoopLogger(HostedApplicationLogger):
    def debug(self, message: str, **fields: object) -> None:
        return None

    def info(self, message: str, **fields: object) -> None:
        return None

    def warning(self, message: str, **fields: object) -> None:
        return None

    def error(self, message: str, **fields: object) -> None:
        return None


class _NoopPublisher(HostedApplicationEventPublisher):
    async def publish(self, event: HostedApplicationEvent) -> None:
        return None


class _FixedLifecycle(HostedApplicationLifecycleSnapshotProvider):
    def __init__(self, snapshot: HostedApplicationLifecycleSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> HostedApplicationLifecycleSnapshot:
        return self._snapshot


class _NoopShutdown(HostedApplicationShutdownCoordinator):
    def is_shutdown_requested(self) -> bool:
        return False

    def current_request(self) -> HostedApplicationShutdownRequestSnapshot | None:
        return None

    def request_shutdown(
        self,
        reason_code: str,
        *,
        deadline_at: datetime | None = None,
    ) -> HostedApplicationShutdownRequestSnapshot:
        return HostedApplicationShutdownRequestSnapshot(
            reason_code=reason_code,
            requested_at=datetime.now(UTC),
            deadline_at=deadline_at,
        )

    async def wait_until_requested(self) -> HostedApplicationEffectiveControlRequest:
        return HostedApplicationEffectiveControlRequest(
            intent="stop",
            reason_code="test",
            requested_at=datetime.now(UTC),
        )


class SampleComponent:
    component_id = "background_worker"

    async def start(self, context: HostedApplicationContext) -> None:
        return None

    async def stop(self, context: HostedApplicationContext) -> None:
        return None

    async def health(self, context: HostedApplicationContext) -> HostedApplicationComponentHealth:
        return HostedApplicationComponentHealth(
            component_id=self.component_id,
            enabled=True,
            required=True,
            state=HostedApplicationComponentState.READY,
            healthy=True,
            ready=True,
        )


async def warm_cache_handler(context: HostedApplicationContext) -> None:
    return None


async def flush_state_handler(context: HostedApplicationContext) -> None:
    return None


async def record_hosting_diagnostic_handler(event: HostedApplicationEvent) -> None:
    return None


def build_minimal_profile() -> HostedApplicationProfile:
    return HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
    )


def build_complete_profile() -> HostedApplicationProfile:
    return HostedApplicationProfile(
        application_id="my_application",
        application_factory=sample_application_factory,
        hooks=HostedApplicationHooks(
            before_ready=(
                HostedApplicationHook(
                    hook_id="warm_cache",
                    handler=warm_cache_handler,
                ),
            ),
            after_stop=(
                HostedApplicationHook(
                    hook_id="flush_state",
                    handler=flush_state_handler,
                ),
            ),
        ),
        components=(
            HostedApplicationComponentRegistration(
                component=SampleComponent(),
                required=True,
            ),
        ),
        lifecycle=LifecyclePolicy.standard(),
        shutdown=ShutdownPolicy.drain_then_cancel(drain_timeout_seconds=30),
        restart=RestartPolicy.on_failure(max_attempts=3),
        component_failure=ComponentFailurePolicy.standard(),
        hook_failure=HookFailurePolicy.standard(),
        instance=InstancePolicy.standard(),
        event_subscriptions=(
            HostedApplicationEventSubscription(
                subscription_id="hosting_diagnostics",
                event_types=(
                    HostedApplicationEventType.APPLICATION_FAILED,
                    HostedApplicationEventType.COMPONENT_FAILED,
                ),
                handler=record_hosting_diagnostic_handler,
            ),
        ),
    )


def build_sample_context(
    *,
    profile: HostedApplicationProfile | None = None,
    instance_id: str = "instance-001",
) -> HostedApplicationContext:
    resolved_profile = profile or build_minimal_profile()
    moment = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    lifecycle = HostedApplicationLifecycleSnapshot(
        state=HostedApplicationLifecycleState.READY,
        accepting_new_work=True,
        shutdown_requested=False,
        last_transition_at=moment,
        reason_code="ready",
    )
    return HostedApplicationContext(
        application_id=resolved_profile.application_id,
        instance_id=instance_id,
        profile=resolved_profile.public_view(),
        profile_digest=resolved_profile.profile_digest(),
        paths=HostedApplicationPaths(
            data_home=Path("data") / resolved_profile.application_id,
            run_directory=Path("data") / resolved_profile.application_id / "run",
        ),
        process_identity=HostedApplicationProcessIdentity(
            process_id=4242,
            host_id="host-1",
            user_scope_id="user-1",
            started_at=moment,
        ),
        services=HostedApplicationServiceRegistry(),
        clock=_FixedClock(moment),
        logger=_NoopLogger(),
        event_publisher=_NoopPublisher(),
        shutdown=_NoopShutdown(),
        lifecycle=_FixedLifecycle(lifecycle),
    )
