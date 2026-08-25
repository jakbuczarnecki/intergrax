# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Application-composition adapter for Vendor Knowledge sync."""

# Domain validation intentionally uses stable ValueError codes.
# ruff: noqa: TRY004

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import cast

from pydantic import BaseModel

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskQueue
from intergrax.queueing.worker.registry import BackgroundTaskHandler, TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge.indexed_source_eligibility import (
    IndexedSourceSyncHandlerRegistrationView,
)
from intergrax.runtime.vendor_knowledge.sync_coordinator import (
    VendorKnowledgeSyncCoordinator,
)
from intergrax.runtime.vendor_knowledge.sync_jobs import (
    VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA,
    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
    VendorKnowledgeSyncJob,
    VendorKnowledgeSyncScheduler,
    decode_vendor_knowledge_sync_job,
    encode_vendor_knowledge_sync_job,
    vendor_knowledge_sync_idempotency_key,
)
from intergrax.runtime.vendor_knowledge.sync_worker import (
    MainLoopProvider,
    Sleeper,
    VendorKnowledgeSyncWorkerOutput,
    make_vendor_knowledge_sync_worker_handler,
)
from intergrax.tools.execution_models import ToolExecutionResult

CoordinatorFactory = Callable[[str, str], VendorKnowledgeSyncCoordinator]
VendorKnowledgeSyncHandlerKey = tuple[str, IntegrationCategory, str, str]


@dataclass(frozen=True)
class VendorKnowledgeSyncHandlerRegistration:
    provider_id: str
    integration_kind: IntegrationCategory
    source_kind: str
    handler_ref: str
    handler: object
    registration_version: str | None = None
    active: bool = True


@dataclass(frozen=True)
class VendorKnowledgeSyncExecutableRegistration:
    """Published identity of the canonical Vendor Knowledge sync executable."""

    handler: BackgroundTaskHandler
    registration_token: str


class VendorKnowledgeSyncHandlerRegistry:
    """Atomic owner of canonical task and Indexed Source registration."""

    def __init__(self, task_registry: TaskExecutionRegistry | None = None) -> None:
        self._task_registry = task_registry or TaskExecutionRegistry()
        self._registrations: dict[
            VendorKnowledgeSyncHandlerKey,
            VendorKnowledgeSyncHandlerRegistration,
        ] = {}
        self._owns_executable = False
        self._executable_registration_token: str | None = None
        self._lock = RLock()

    @property
    def task_registry(self) -> TaskExecutionRegistry:
        """Return the task registry owned by this registration boundary."""
        return self._task_registry

    def register_executable(
        self,
        handler: object,
        *,
        registration_token: str | None = None,
    ) -> None:
        """Publish the generic sync task without Indexed Source dimensions."""
        if not callable(handler):
            raise ValueError("sync_handler_registration_handler_not_executable")
        normalized_token = _normalize_registration_token(registration_token)
        with self._lock:
            task_handler = self._registered_task_handler()
            if task_handler is None:
                self._task_registry.register(
                    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
                    cast(BackgroundTaskHandler, handler),
                )
                self._owns_executable = True
            elif task_handler is not handler:
                raise ValueError("sync_handler_registration_executable_mismatch")
            if self._executable_registration_token is not None and (
                self._executable_registration_token != normalized_token
            ):
                raise ValueError("sync_handler_registration_configuration_mismatch")
            self._executable_registration_token = normalized_token

    def register_dimension(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
        handler_ref: str,
        registration_version: str | None = None,
        active: bool = True,
        executable_registration: VendorKnowledgeSyncExecutableRegistration | None = None,
    ) -> None:
        """Bind one Indexed Source dimension to the published canonical task."""
        key, normalized_version = self._validate_dimension(
            provider_id=provider_id,
            integration_kind=integration_kind,
            source_kind=source_kind,
            handler_ref=handler_ref,
            registration_version=registration_version,
            active=active,
        )
        with self._lock:
            task_handler = self._registered_task_handler()
            if task_handler is None:
                raise ValueError("sync_handler_registration_canonical_task_missing")
            if not callable(task_handler):
                raise ValueError("sync_handler_registration_handler_not_executable")
            next_registration_token = self._executable_registration_token
            if executable_registration is not None:
                if task_handler is not executable_registration.handler:
                    raise ValueError("sync_handler_registration_executable_mismatch")
                if next_registration_token is not None and (
                    next_registration_token != executable_registration.registration_token
                ):
                    raise ValueError("sync_handler_registration_configuration_mismatch")
                next_registration_token = executable_registration.registration_token
            if key in self._registrations:
                raise ValueError("sync_handler_registration_already_registered")
            self._publish_metadata(
                key,
                VendorKnowledgeSyncHandlerRegistration(
                    provider_id=key[0],
                    integration_kind=key[1],
                    source_kind=key[2],
                    handler_ref=key[3],
                    handler=task_handler,
                    registration_version=normalized_version,
                    active=active,
                ),
            )
            self._executable_registration_token = next_registration_token

    def _claim_executable(
        self,
        registration: VendorKnowledgeSyncExecutableRegistration,
        *,
        owned: bool,
    ) -> None:
        with self._lock:
            if self._registered_task_handler() is not registration.handler:
                raise ValueError("sync_handler_registration_executable_mismatch")
            if self._executable_registration_token is not None and (
                self._executable_registration_token
                != registration.registration_token
            ):
                raise ValueError("sync_handler_registration_configuration_mismatch")
            self._executable_registration_token = registration.registration_token
            self._owns_executable = self._owns_executable or owned

    def _rollback_executable(self, handler: object) -> None:
        with self._lock:
            if self._registrations or not self._owns_executable:
                return
            if self._registered_task_handler() is handler:
                self._task_registry.unregister(VENDOR_KNOWLEDGE_SYNC_TASK_NAME)
                self._owns_executable = False
                self._executable_registration_token = None

    def register(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
        handler_ref: str,
        handler: object,
        registration_version: str | None = None,
        active: bool = True,
    ) -> None:
        values = {
            "provider_id": provider_id,
            "source_kind": source_kind,
            "handler_ref": handler_ref,
        }
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("sync_handler_registration_identifier_required")
        if not isinstance(integration_kind, IntegrationCategory):
            raise ValueError("sync_handler_registration_integration_kind_invalid")
        if registration_version is not None and (
            not isinstance(registration_version, str) or not registration_version.strip()
        ):
            raise ValueError("sync_handler_registration_version_invalid")
        if not isinstance(active, bool):
            raise ValueError("sync_handler_registration_active_invalid")
        if not callable(handler):
            raise ValueError("sync_handler_registration_handler_not_executable")
        normalized_version = (
            "legacy"
            if registration_version is None
            else registration_version.strip()
        )
        key = (provider_id.strip(), integration_kind, source_kind.strip(), handler_ref.strip())
        registration = VendorKnowledgeSyncHandlerRegistration(
            provider_id=key[0],
            integration_kind=key[1],
            source_kind=key[2],
            handler_ref=key[3],
            handler=handler,
            registration_version=normalized_version,
            active=active,
        )
        with self._lock:
            if key in self._registrations:
                raise ValueError("sync_handler_registration_already_registered")
            task_handler = self._registered_task_handler()
            task_added = task_handler is None
            if task_handler is not None and task_handler is not handler:
                raise ValueError("sync_handler_registration_executable_mismatch")
            if task_added:
                self._task_registry.register(
                    VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
                    cast(BackgroundTaskHandler, handler),
                )
                self._owns_executable = True
            try:
                self._publish_metadata(key, registration)
            except Exception:
                if task_added:
                    self._task_registry.unregister(VENDOR_KNOWLEDGE_SYNC_TASK_NAME)
                    self._owns_executable = False
                raise

    def resolve_registration(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
        handler_ref: str,
    ) -> IndexedSourceSyncHandlerRegistrationView | None:
        with self._lock:
            registration = self._registrations.get(
                (provider_id, integration_kind, source_kind, handler_ref)
            )
            if registration is None:
                return None
            task_handler = self._registered_task_handler()
            executable = callable(task_handler) and task_handler is registration.handler
            return IndexedSourceSyncHandlerRegistrationView(
                task_name=VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
                handler_ref=registration.handler_ref,
                registration_version=registration.registration_version,
                active=registration.active,
                executable=executable,
                handler=task_handler,
            )

    def resolve_handler(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
        handler_ref: str,
    ) -> object | None:
        view = self.resolve_registration(
            provider_id=provider_id,
            integration_kind=integration_kind,
            source_kind=source_kind,
            handler_ref=handler_ref,
        )
        if view is None or not view.active or not view.executable:
            return None
        return view.handler

    def handler_registration_version(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
        handler_ref: str,
    ) -> str | None:
        view = self.resolve_registration(
            provider_id=provider_id,
            integration_kind=integration_kind,
            source_kind=source_kind,
            handler_ref=handler_ref,
        )
        if view is None or not view.active or not view.executable:
            return None
        return view.registration_version

    def disable(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
        handler_ref: str,
    ) -> bool:
        with self._lock:
            key = (provider_id, integration_kind, source_kind, handler_ref)
            registration = self._registrations.get(key)
            if registration is None:
                return False
            self._registrations[key] = VendorKnowledgeSyncHandlerRegistration(
                provider_id=registration.provider_id,
                integration_kind=registration.integration_kind,
                source_kind=registration.source_kind,
                handler_ref=registration.handler_ref,
                handler=registration.handler,
                registration_version=registration.registration_version,
                active=False,
            )
            return True

    def unregister(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
        handler_ref: str,
    ) -> bool:
        with self._lock:
            removed = self._registrations.pop(
                (provider_id, integration_kind, source_kind, handler_ref),
                None,
            )
            if removed is None:
                return False
            if not self._registrations:
                task_handler = self._registered_task_handler()
                if self._owns_executable and task_handler is removed.handler:
                    self._task_registry.unregister(VENDOR_KNOWLEDGE_SYNC_TASK_NAME)
                    self._owns_executable = False
                    self._executable_registration_token = None
            return True

    def _validate_dimension(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
        handler_ref: str,
        registration_version: str | None,
        active: bool,
    ) -> tuple[VendorKnowledgeSyncHandlerKey, str]:
        values = {
            "provider_id": provider_id,
            "source_kind": source_kind,
            "handler_ref": handler_ref,
        }
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("sync_handler_registration_identifier_required")
        if not isinstance(integration_kind, IntegrationCategory):
            raise ValueError("sync_handler_registration_integration_kind_invalid")
        if registration_version is not None and (
            not isinstance(registration_version, str) or not registration_version.strip()
        ):
            raise ValueError("sync_handler_registration_version_invalid")
        if not isinstance(active, bool):
            raise ValueError("sync_handler_registration_active_invalid")
        return (
            (provider_id.strip(), integration_kind, source_kind.strip(), handler_ref.strip()),
            "legacy" if registration_version is None else registration_version.strip(),
        )

    def _registered_task_handler(self) -> object | None:
        try:
            return self._task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME)
        except ValueError:
            return None

    def _publish_metadata(
        self,
        key: VendorKnowledgeSyncHandlerKey,
        registration: VendorKnowledgeSyncHandlerRegistration,
    ) -> None:
        self._registrations[key] = registration

__all__ = [
    "VENDOR_KNOWLEDGE_SYNC_JOB_SCHEMA",
    "VENDOR_KNOWLEDGE_SYNC_TASK_NAME",
    "VendorKnowledgeSyncDispatcher",
    "VendorKnowledgeSyncExecutableRegistration",
    "VendorKnowledgeSyncHandlerRegistration",
    "VendorKnowledgeSyncHandlerRegistry",
    "VendorKnowledgeSyncJob",
    "VendorKnowledgeSyncWorkerOutput",
    "decode_vendor_knowledge_sync_job",
    "encode_vendor_knowledge_sync_job",
    "make_vendor_knowledge_sync_handler",
    "owner_id_for_sync_run",
    "register_vendor_knowledge_indexed_sync_dimension",
    "register_vendor_knowledge_sync_executable",
    "register_vendor_knowledge_sync_handler",
    "unregister_vendor_knowledge_sync_handler",
    "vendor_knowledge_sync_idempotency_key",
]


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _normalize_registration_token(value: str | None) -> str:
    if value is None:
        return "default"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sync_handler_registration_token_invalid")
    return value.strip()


def owner_id_for_sync_run(run_id: str) -> str:
    """Stable lease owner identity derived from queue run identity (hashed)."""
    cleaned = _require_non_empty(run_id, field_name="run_id")
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return f"vendor_knowledge.sync:{digest}"


class VendorKnowledgeSyncDispatcher:
    """Thin application wrapper over VendorKnowledgeSyncScheduler."""

    def __init__(self, task_queue: TaskQueue) -> None:
        self._scheduler = VendorKnowledgeSyncScheduler(task_queue=task_queue)

    @property
    def scheduler(self) -> VendorKnowledgeSyncScheduler:
        return self._scheduler

    def enqueue(self, *, job: VendorKnowledgeSyncJob, run_id: str) -> TaskHandle:
        return self._scheduler.enqueue_job(job=job, run_id=run_id)

    def enqueue_incremental(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        operation_id: str,
        run_id: str,
        page_size: int = 100,
    ) -> TaskHandle:
        return self._scheduler.enqueue_incremental(
            tenant_id=tenant_id,
            binding_id=binding_id,
            operation_id=operation_id,
            run_id=run_id,
            page_size=page_size,
        )

    def enqueue_reconciliation(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        operation_id: str,
        run_id: str,
        page_size: int = 100,
    ) -> TaskHandle:
        return self._scheduler.enqueue_reconciliation(
            tenant_id=tenant_id,
            binding_id=binding_id,
            operation_id=operation_id,
            run_id=run_id,
            page_size=page_size,
        )


def make_vendor_knowledge_sync_handler(
    coordinator_factory: CoordinatorFactory,
    dispatcher: VendorKnowledgeSyncDispatcher,
    *,
    main_loop_provider: MainLoopProvider | None = None,
    retry_delays_seconds: tuple[float, ...] = (0.25, 1.0, 4.0),
    sleeper: Sleeper | None = None,
):
    """Build a TaskExecutionRegistry-compatible sync handler for applications."""

    def _resolver(tenant_id: str, run_id: str) -> VendorKnowledgeSyncCoordinator:
        return coordinator_factory(tenant_id, owner_id_for_sync_run(run_id))

    return make_vendor_knowledge_sync_worker_handler(
        coordinator_resolver=_resolver,
        scheduler=dispatcher.scheduler,
        main_loop_provider=main_loop_provider,
        retry_delays_seconds=retry_delays_seconds,
        sleeper=sleeper,
    )


def register_vendor_knowledge_sync_executable(
    *,
    task_registry: TaskExecutionRegistry,
    coordinator_factory: CoordinatorFactory | None = None,
    dispatcher: VendorKnowledgeSyncDispatcher | None = None,
    registration_token: str | None = None,
    published_registration: VendorKnowledgeSyncExecutableRegistration | None = None,
    main_loop_provider: MainLoopProvider | None = None,
    retry_delays_seconds: tuple[float, ...] = (0.25, 1.0, 4.0),
    sleeper: Sleeper | None = None,
) -> VendorKnowledgeSyncExecutableRegistration:
    """Build and publish the canonical sync executable exactly once."""
    normalized_token = _normalize_registration_token(registration_token)
    if published_registration is not None:
        if not isinstance(published_registration, VendorKnowledgeSyncExecutableRegistration):
            raise ValueError("sync_handler_registration_token_invalid")
        if registration_token is not None and (
            published_registration.registration_token != normalized_token
        ):
            raise ValueError("sync_handler_registration_configuration_mismatch")
        registration = published_registration
    else:
        if coordinator_factory is None or dispatcher is None:
            raise TypeError("coordinator_factory and dispatcher are required")
        registration = VendorKnowledgeSyncExecutableRegistration(
            handler=cast(
                BackgroundTaskHandler,
                make_vendor_knowledge_sync_handler(
                    coordinator_factory,
                    dispatcher,
                    main_loop_provider=main_loop_provider,
                    retry_delays_seconds=retry_delays_seconds,
                    sleeper=sleeper,
                ),
            ),
            registration_token=normalized_token,
        )
    try:
        existing_handler = task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME)
    except ValueError:
        existing_handler = None
    if existing_handler is not None:
        if existing_handler is not registration.handler:
            raise ValueError("sync_handler_registration_executable_mismatch")
        return registration
    task_registry.register(VENDOR_KNOWLEDGE_SYNC_TASK_NAME, registration.handler)
    return registration


def register_vendor_knowledge_indexed_sync_dimension(
    *,
    handler_registry: VendorKnowledgeSyncHandlerRegistry,
    provider_id: str,
    integration_kind: IntegrationCategory,
    source_kind: str,
    handler_ref: str,
    registration_version: str | None = None,
    active: bool = True,
    executable_registration: VendorKnowledgeSyncExecutableRegistration | None = None,
) -> None:
    """Publish Indexed Source metadata against the existing canonical task."""
    handler_registry.register_dimension(
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        handler_ref=handler_ref,
        registration_version=registration_version,
        active=active,
        executable_registration=executable_registration,
    )


def register_vendor_knowledge_sync_handler(
    registry: TaskExecutionRegistry | None = None,
    coordinator_factory: CoordinatorFactory | None = None,
    dispatcher: VendorKnowledgeSyncDispatcher | None = None,
    *,
    task_registry: TaskExecutionRegistry | None = None,
    handler_registry: VendorKnowledgeSyncHandlerRegistry | None = None,
    provider_id: str | None = None,
    integration_kind: IntegrationCategory | None = None,
    source_kind: str | None = None,
    handler_ref: str | None = None,
    registration_version: str | None = None,
    active: bool = True,
    executable_registration: VendorKnowledgeSyncExecutableRegistration | None = None,
    registration_token: str | None = None,
    main_loop_provider: MainLoopProvider | None = None,
    retry_delays_seconds: tuple[float, ...] = (0.25, 1.0, 4.0),
    sleeper: Sleeper | None = None,
) -> VendorKnowledgeSyncExecutableRegistration:
    """Preserve generic registration and support published dimension composition."""
    effective_task_registry = task_registry or registry
    if effective_task_registry is None:
        raise TypeError("task_registry is required")
    dimensions = (provider_id, integration_kind, source_kind, handler_ref)
    if any(value is not None for value in dimensions) and handler_registry is None:
        raise ValueError("sync_handler_registration_metadata_boundary_required")
    if handler_registry is not None and (
        handler_registry.task_registry is not effective_task_registry
    ):
        raise ValueError("sync_handler_registration_registry_mismatch")
    if not any(value is not None for value in dimensions) and executable_registration is None:
        if coordinator_factory is None or dispatcher is None:
            raise TypeError("coordinator_factory and dispatcher are required")
        if handler_registry is None:
            handler = make_vendor_knowledge_sync_handler(
                coordinator_factory,
                dispatcher,
                main_loop_provider=main_loop_provider,
                retry_delays_seconds=retry_delays_seconds,
                sleeper=sleeper,
            )
            effective_task_registry.register(
                VENDOR_KNOWLEDGE_SYNC_TASK_NAME,
                cast(BackgroundTaskHandler, handler),
            )
            return VendorKnowledgeSyncExecutableRegistration(
                handler=cast(BackgroundTaskHandler, handler),
                registration_token=_normalize_registration_token(registration_token),
            )
    dimension_requested = any(value is not None for value in dimensions)
    if dimension_requested:
        if (
            handler_registry is None
            or provider_id is None
            or integration_kind is None
            or source_kind is None
            or handler_ref is None
        ):
            raise ValueError("sync_handler_registration_metadata_incomplete")
        handler_registry._validate_dimension(
            provider_id=provider_id,
            integration_kind=integration_kind,
            source_kind=source_kind,
            handler_ref=handler_ref,
            registration_version=registration_version,
            active=active,
        )
        if handler_registry.resolve_registration(
            provider_id=provider_id,
            integration_kind=integration_kind,
            source_kind=source_kind,
            handler_ref=handler_ref,
        ) is not None:
            raise ValueError("sync_handler_registration_already_registered")
    try:
        effective_task_registry.get_handler(VENDOR_KNOWLEDGE_SYNC_TASK_NAME)
    except ValueError:
        had_task = False
    else:
        had_task = True
    registration = register_vendor_knowledge_sync_executable(
        task_registry=effective_task_registry,
        coordinator_factory=coordinator_factory,
        dispatcher=dispatcher,
        registration_token=registration_token,
        published_registration=executable_registration,
        main_loop_provider=main_loop_provider,
        retry_delays_seconds=retry_delays_seconds,
        sleeper=sleeper,
    )
    if dimension_requested:
        if (
            handler_registry is None
            or provider_id is None
            or integration_kind is None
            or source_kind is None
            or handler_ref is None
        ):
            raise ValueError("sync_handler_registration_metadata_incomplete")
        handler_registry._claim_executable(registration, owned=not had_task)
        try:
            register_vendor_knowledge_indexed_sync_dimension(
                handler_registry=handler_registry,
                provider_id=provider_id,
                integration_kind=integration_kind,
                source_kind=source_kind,
                handler_ref=handler_ref,
                registration_version=registration_version,
                active=active,
                executable_registration=registration,
            )
        except Exception:
            if not had_task:
                handler_registry._rollback_executable(registration.handler)
            raise
    elif handler_registry is not None:
        handler_registry.register_executable(
            registration.handler,
            registration_token=registration.registration_token,
        )
    return registration


def unregister_vendor_knowledge_sync_handler(
    handler_registry: VendorKnowledgeSyncHandlerRegistry,
    *,
    provider_id: str,
    integration_kind: IntegrationCategory,
    source_kind: str,
    handler_ref: str,
) -> bool:
    """Remove the indexed registration and its canonical executable task."""
    return handler_registry.unregister(
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        handler_ref=handler_ref,
    )
