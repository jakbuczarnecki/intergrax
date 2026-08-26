# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Foreground hosted application runner facade (APP-HOST-9A)."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from intergrax.hosting.contracts.context import (
    HostedApplicationClock,
    HostedApplicationEventPublisher,
    HostedApplicationLogger,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
)
from intergrax.hosting.contracts.policies import InstanceExclusivityMode
from intergrax.hosting.contracts.profile import HostedApplicationProfile
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.engine.definition import (
    HostedApplicationDefinition,
    resolve_hosted_application_definition,
)
from intergrax.hosting.engine.engine import HostedApplicationEngine
from intergrax.hosting.engine.ports import HostedApplicationInstanceGuardPort
from intergrax.hosting.errors import HostedApplicationConfigurationError, HostedApplicationInstanceOwnershipError
from intergrax.hosting.eventing import ObservabilityHostedApplicationEventPublisher
from intergrax.hosting.instance.contracts import (
    HostedApplicationInstanceAcquisitionResult,
    HostedApplicationInstanceIdentity,
    HostedApplicationInstanceLeasePublicView,
    InstanceAcquisitionClassification,
)
from intergrax.hosting.instance.file_guard import FileHostedApplicationInstanceGuard
from intergrax.hosting.shutdown import MonotonicClock, SystemMonotonicClock
from intergrax.hosting.signals import HostedApplicationSignalBridge, PortableForegroundSignalAdapter
from intergrax.hosting.supervisor.supervisor import (
    HostedApplicationEngineFactory,
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
    HostedApplicationSupervisorResult,
    InstanceIdGenerator,
)

_HOSTING_ROOT_NAME = ".intergrax"
_HOSTING_SUBDIR = "hosting"


class _SystemWallClock:
    """Reference wall-clock adapter using timezone-aware UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class _StandardHostedApplicationLogger:
    """Private logging adapter for hosted application runners."""

    def __init__(self, application_id: str) -> None:
        self._logger = logging.getLogger(f"intergrax.hosting.{application_id}")

    def debug(self, message: str, **fields: object) -> None:
        self._logger.debug(message, extra=_logging_extra(fields))

    def info(self, message: str, **fields: object) -> None:
        self._logger.info(message, extra=_logging_extra(fields))

    def warning(self, message: str, **fields: object) -> None:
        self._logger.warning(message, extra=_logging_extra(fields))

    def error(self, message: str, **fields: object) -> None:
        self._logger.error(message, extra=_logging_extra(fields))


def _logging_extra(fields: dict[str, object]) -> dict[str, object]:
    if not fields:
        return {}
    return {"hosted_fields": fields}


def _best_effort_restrict_permissions(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        path.chmod(0o700)
    except OSError:
        return


def _resolve_reference_paths(application_id: str) -> HostedApplicationPaths:
    root = Path.home() / _HOSTING_ROOT_NAME / _HOSTING_SUBDIR
    data_home = root / "data" / application_id
    run_directory = root / "run"
    try:
        for directory in (data_home, run_directory):
            directory.mkdir(parents=True, exist_ok=True)
            _best_effort_restrict_permissions(directory)
    except OSError as exc:
        raise HostedApplicationConfigurationError("hosted application paths could not be prepared") from exc
    return HostedApplicationPaths(
        data_home=data_home.resolve(),
        run_directory=run_directory.resolve(),
    )


@dataclass
class _NonExclusiveLease:
    """Private non-exclusive lease for multi-instance hosting mode."""

    identity: HostedApplicationInstanceIdentity
    acquired_at: datetime
    _released: bool = field(default=False, init=False)

    def is_valid(self) -> bool:
        return not self._released

    def verify_ownership(self) -> None:
        if self._released:
            raise HostedApplicationInstanceOwnershipError("instance lease is no longer valid")

    def public_view(self) -> HostedApplicationInstanceLeasePublicView:
        return HostedApplicationInstanceLeasePublicView(
            application_id=self.identity.application_id,
            instance_id=self.identity.instance_id,
            process_id=self.identity.process_identity.process_id,
            process_started_at=self.identity.process_identity.started_at,
            host_id=self.identity.process_identity.host_id,
            user_scope_id=self.identity.process_identity.user_scope_id,
            profile_digest=self.identity.profile_digest,
            acquired_at=self.acquired_at,
        )

    async def release(self) -> None:
        self._released = True


@dataclass
class _NonExclusiveInstanceGuard:
    """Private non-exclusive instance guard for multi-instance hosting mode."""

    clock: HostedApplicationClock

    async def acquire(
        self,
        identity: HostedApplicationInstanceIdentity,
    ) -> HostedApplicationInstanceAcquisitionResult:
        lease = _NonExclusiveLease(identity=identity, acquired_at=self.clock.now())
        return HostedApplicationInstanceAcquisitionResult(
            lease=lease,
            classification=InstanceAcquisitionClassification.FRESH,
        )


def _create_instance_guard(
    definition: HostedApplicationDefinition,
    paths: HostedApplicationPaths,
    process_identity: HostedApplicationProcessIdentity,
    clock: HostedApplicationClock,
) -> HostedApplicationInstanceGuardPort:
    if definition.instance_policy.exclusivity_mode is InstanceExclusivityMode.MULTI_INSTANCE:
        return _NonExclusiveInstanceGuard(clock=clock)
    return FileHostedApplicationInstanceGuard(
        run_directory=paths.run_directory,
        instance_policy=definition.instance_policy,
        process_identity=process_identity,
        clock=clock,
    )


@dataclass(frozen=True, slots=True)
class _RunnerFactories:
    """Private dependency seam for deterministic runner tests."""

    create_paths: Callable[[HostedApplicationDefinition], HostedApplicationPaths]
    create_clock: Callable[[], HostedApplicationClock]
    create_monotonic_clock: Callable[[], MonotonicClock]
    create_logger: Callable[[str], HostedApplicationLogger]
    create_event_publisher: Callable[[], Any]
    create_process_identity: Callable[[HostedApplicationClock], HostedApplicationProcessIdentity]
    create_instance_guard: Callable[
        [HostedApplicationDefinition, HostedApplicationPaths, HostedApplicationProcessIdentity, HostedApplicationClock],
        HostedApplicationInstanceGuardPort,
    ]
    create_signal_adapter: Callable[[HostedApplicationControlCoordinator], HostedApplicationSignalBridge]
    instance_id_generator: InstanceIdGenerator | None = None


def _default_runner_factories() -> _RunnerFactories:
    return _RunnerFactories(
        create_paths=lambda definition: _resolve_reference_paths(definition.application_id),
        create_clock=_SystemWallClock,
        create_monotonic_clock=SystemMonotonicClock,
        create_logger=lambda application_id: _StandardHostedApplicationLogger(application_id),
        create_event_publisher=ObservabilityHostedApplicationEventPublisher,
        create_process_identity=lambda clock: HostedApplicationProcessIdentity(
            process_id=os.getpid(),
            started_at=clock.now(),
            host_id=None,
            user_scope_id=None,
        ),
        create_instance_guard=_create_instance_guard,
        create_signal_adapter=lambda control: PortableForegroundSignalAdapter(
            coordinator=control,
            enable_sighup_restart=False,
        ),
    )


def _build_engine_factory(
    *,
    paths: HostedApplicationPaths,
    process_identity: HostedApplicationProcessIdentity,
    clock: HostedApplicationClock,
    monotonic_clock: MonotonicClock,
    logger: HostedApplicationLogger,
    event_publisher: Any,
    create_instance_guard: Callable[
        [HostedApplicationDefinition, HostedApplicationPaths, HostedApplicationProcessIdentity, HostedApplicationClock],
        HostedApplicationInstanceGuardPort,
    ],
) -> HostedApplicationEngineFactory:
    def factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        guard = create_instance_guard(
            launch.definition,
            paths,
            process_identity,
            clock,
        )
        return HostedApplicationEngine(
            definition=launch.definition,
            instance_id=launch.instance_id,
            paths=paths,
            process_identity=process_identity,
            clock=clock,
            logger=logger,
            shutdown=launch.control,
            event_publisher=event_publisher,
            instance_guard=guard,
            monotonic_clock=monotonic_clock,
        )

    return factory


async def _run_resolved_hosted_application(
    definition: HostedApplicationDefinition,
    factories: _RunnerFactories,
) -> HostedApplicationSupervisorResult:
    clock = factories.create_clock()
    monotonic_clock = factories.create_monotonic_clock()
    control = HostedApplicationControlCoordinator(clock=clock)
    signal_adapter = factories.create_signal_adapter(control)

    paths = factories.create_paths(definition)
    logger = factories.create_logger(definition.application_id)
    event_publisher = factories.create_event_publisher()
    process_identity = factories.create_process_identity(clock)

    engine_factory = _build_engine_factory(
        paths=paths,
        process_identity=process_identity,
        clock=clock,
        monotonic_clock=monotonic_clock,
        logger=logger,
        event_publisher=event_publisher,
        create_instance_guard=factories.create_instance_guard,
    )

    supervisor_kwargs: dict[str, Any] = {}
    if factories.instance_id_generator is not None:
        supervisor_kwargs["instance_id_generator"] = factories.instance_id_generator

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=event_publisher,
        clock=clock,
        monotonic_clock=monotonic_clock,
        **supervisor_kwargs,
    )

    signal_adapter.install()
    try:
        return await supervisor.run()
    finally:
        signal_adapter.restore()


def run_hosted_application(
    profile: HostedApplicationProfile,
    *,
    event_publisher_factory: Callable[[], HostedApplicationEventPublisher] | None = None,
) -> HostedApplicationSupervisorResult:
    """Run one hosted application in the foreground until terminal exit."""
    definition = resolve_hosted_application_definition(profile)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise HostedApplicationConfigurationError(
            "run_hosted_application() cannot be called from an active event loop"
        )

    factories = _default_runner_factories()
    if event_publisher_factory is not None:
        factories = replace(factories, create_event_publisher=event_publisher_factory)
    return asyncio.run(_run_resolved_hosted_application(definition, factories))
