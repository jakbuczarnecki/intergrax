# © Artur Czarnecki. All rights reserved.

"""Sanctioned production process lifetime — composition + durable store ownership (PROD-Q-R1-R7-D2)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from intergrax.agent_distribution.deployment import RuntimeDeploymentAdapter
from intergrax.applications._shared.durable_agent_platform_runtime import (
    build_durable_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.production_execution_continuation_store import (
    MissingContinuationFilePolicy,
    SerializedSealExecutionContinuationStateStore,
    open_production_execution_continuation_state_store,
)
from intergrax.applications._shared.production_lifecycle import (
    ProductionLifecycleConfigurationError,
    ProductionLifecycleLauncher,
    ProductionLifecycleServices,
    require_production_control_plane_governance,
    wire_production_lifecycle_services,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    assemble_durable_production_process_composition,
)
from intergrax.applications._shared.production_process_serving_reconciliation import (
    ProductionServingReconciliationError,
    ProductionServingReconciliationResult,
    reconcile_production_serving_environment_on_process_open,
    resolve_mandatory_production_serving_reconciliation_targets,
)
from intergrax.applications._shared.registry_projection_rehydrator import (
    RuntimeRegistryProjectionRehydrator,
)
from intergrax.applications._shared.process_lifecycle_resource_cleanup import (
    cleanup_errors_from_close_failure,
    raise_primary_with_cleanup_errors,
    raise_startup_failure_with_cleanup,
    release_lifecycle_owned_stores,
)
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)
from intergrax.contracts.governed_external_work_host_store_lifecycle import (
    OwnedGovernedExternalWorkHostStoreBundle,
)
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    SQLiteProviderInvocationStore,
    open_provider_invocation_store_at,
)

if TYPE_CHECKING:
    from intergrax.applications._shared.production_agent_capability_runtime import (
        AgentCapabilityApplicationComposition,
    )


class _ProductionSlotPhase(Enum):
    AVAILABLE = "available"
    OPENING = "opening"
    ACTIVE = "active"


_production_slot_guard = threading.Lock()
_production_slot_phase = _ProductionSlotPhase.AVAILABLE
_active_production_process_lifecycle: ProductionProcessLifecycle | None = None
_production_opening_thread_id: int | None = None


class ProductionProcessLifecycleError(ValueError):
    """Invalid production process lifecycle wiring."""


@dataclass
class ProductionProcessLifecycle:
    """One sanctioned production composition root, launcher, governance, and owned durable stores."""

    composition: ProductionProcessComposition
    launcher: ProductionLifecycleLauncher
    governance: ProductionControlPlaneGovernance
    lifecycle_services: ProductionLifecycleServices
    durable_agent_distribution_db: Path
    deployment_adapter: RuntimeDeploymentAdapter
    registry_projection_rehydrator: RuntimeRegistryProjectionRehydrator = field(
        repr=False,
    )
    _owned_continuation_store: SerializedSealExecutionContinuationStateStore = field(
        repr=False,
    )
    _owned_invocation_store: SQLiteProviderInvocationStore | None = field(
        default=None,
        repr=False,
    )
    _owned_governed_external_work_host_store_bundle: (
        OwnedGovernedExternalWorkHostStoreBundle | None
    ) = field(default=None, repr=False)
    _continuation_closed: bool = field(default=False, repr=False)
    _invocation_closed: bool = field(default=False, repr=False)
    _governed_external_work_host_store_closed: bool = field(default=False, repr=False)
    _process_closed: bool = field(default=False, repr=False)

    @property
    def execution_continuation_state_store(self) -> ExecutionContinuationStateStore:
        return self._owned_continuation_store

    @property
    def provider_invocation_store(self) -> ProviderInvocationStore | None:
        return self.composition.provider_invocation_store

    def reconcile_serving_environment(
        self,
        application_id: str,
        application_environment_id: str,
        *,
        expected_serving_pointer_revision: int | None = None,
    ) -> ProductionServingReconciliationResult:
        """Reconcile one environment after process restart; does not open execution admission."""
        serving_store = (
            self.composition.agent_platform_runtime.stores.serving_store
        )
        return reconcile_production_serving_environment_on_process_open(
            lifecycle_services=self.lifecycle_services,
            serving_store=serving_store,
            registry_projection_rehydrator=self.registry_projection_rehydrator,
            application_id=application_id,
            application_environment_id=application_environment_id,
            expected_serving_pointer_revision=expected_serving_pointer_revision,
        )

    def close_process_resources(self) -> None:
        """Release lifecycle-owned resources exactly once."""
        if self._process_closed:
            raise ProductionProcessLifecycleError(
                "production process lifecycle resources already closed",
            )
        cleanup_errors: list[BaseException] = []
        if not self._continuation_closed:
            try:
                self._owned_continuation_store.close()
                self._continuation_closed = True
            except BaseException as exc:
                cleanup_errors.append(exc)
        owned_invocation = self._owned_invocation_store
        if owned_invocation is not None and not self._invocation_closed:
            try:
                owned_invocation.close()
                self._invocation_closed = True
            except BaseException as exc:
                cleanup_errors.append(exc)
        owned_gca = self._owned_governed_external_work_host_store_bundle
        if owned_gca is not None and not self._governed_external_work_host_store_closed:
            try:
                owned_gca.close()
                self._governed_external_work_host_store_closed = True
            except BaseException as exc:
                cleanup_errors.append(exc)
        self._process_closed = True
        _release_active_production_slot(self)
        if cleanup_errors:
            if len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            if all(isinstance(exc, Exception) for exc in cleanup_errors):
                exception_only = [exc for exc in cleanup_errors if isinstance(exc, Exception)]
                raise ExceptionGroup(
                    "errors while closing production process lifecycle resources",
                    exception_only,
                )
            raise BaseExceptionGroup(
                "errors while closing production process lifecycle resources",
                cleanup_errors,
            )

    def __enter__(self) -> ProductionProcessLifecycle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        body_exc = exc
        try:
            self.close_process_resources()
        except BaseException as cleanup_exc:
            if body_exc is None:
                raise
            cleanup_errors = cleanup_errors_from_close_failure(cleanup_exc)
            raise_primary_with_cleanup_errors(
                body_exc,
                cleanup_errors,
                message=(
                    "production process lifecycle context exited with an error; "
                    "errors while closing resources"
                ),
            )
        return False


def _reserve_production_opening_slot() -> None:
    global _production_slot_phase, _production_opening_thread_id
    thread_id = threading.get_ident()
    with _production_slot_guard:
        if _production_slot_phase is _ProductionSlotPhase.OPENING:
            raise ProductionProcessLifecycleError(
                "duplicate production process lifecycle is not permitted in one process",
            )
        active = _active_production_process_lifecycle
        if (
            _production_slot_phase is _ProductionSlotPhase.ACTIVE
            and active is not None
            and not active._process_closed
        ):
            raise ProductionProcessLifecycleError(
                "duplicate production process lifecycle is not permitted in one process",
            )
        _production_slot_phase = _ProductionSlotPhase.OPENING
        _production_opening_thread_id = thread_id


def _abort_production_opening_slot() -> None:
    global _production_slot_phase, _production_opening_thread_id
    thread_id = threading.get_ident()
    with _production_slot_guard:
        if (
            _production_slot_phase is _ProductionSlotPhase.OPENING
            and _production_opening_thread_id == thread_id
        ):
            _production_slot_phase = _ProductionSlotPhase.AVAILABLE
            _production_opening_thread_id = None


def _commit_production_active_slot(lifecycle: ProductionProcessLifecycle) -> None:
    global _production_slot_phase, _active_production_process_lifecycle, _production_opening_thread_id
    thread_id = threading.get_ident()
    with _production_slot_guard:
        if (
            _production_slot_phase is not _ProductionSlotPhase.OPENING
            or _production_opening_thread_id != thread_id
        ):
            raise ProductionProcessLifecycleError(
                "production process lifecycle opening slot was not reserved by this thread",
            )
        _production_slot_phase = _ProductionSlotPhase.ACTIVE
        _active_production_process_lifecycle = lifecycle
        _production_opening_thread_id = None


def _release_active_production_slot(lifecycle: ProductionProcessLifecycle) -> None:
    global _production_slot_phase, _active_production_process_lifecycle
    with _production_slot_guard:
        if _active_production_process_lifecycle is lifecycle:
            _active_production_process_lifecycle = None
            _production_slot_phase = _ProductionSlotPhase.AVAILABLE


def _release_uncommitted_production_lifecycle_resources(
    *,
    lifecycle: ProductionProcessLifecycle | None,
    continuation_store: SerializedSealExecutionContinuationStateStore | None,
    owned_invocation: SQLiteProviderInvocationStore | None,
    continuation_closed: bool,
    invocation_closed: bool,
) -> list[BaseException]:
    """Close opener-owned stores; prefer lifecycle.close when a partial root exists."""
    if lifecycle is not None:
        try:
            lifecycle.close_process_resources()
        except BaseException as exc:
            return cleanup_errors_from_close_failure(exc)
        return []
    return release_lifecycle_owned_stores(
        continuation_store=continuation_store,
        owned_invocation=owned_invocation,
        continuation_closed=continuation_closed,
        invocation_closed=invocation_closed,
    )


def open_production_process_lifecycle(
    *,
    agent_distribution_db: Path,
    continuation_state_file: Path,
    deployment_adapter: RuntimeDeploymentAdapter,
    governance: ProductionControlPlaneGovernance,
    trusted_artifact_parent_roots: tuple[Path, ...],
    missing_file_policy: MissingContinuationFilePolicy = "fail",
    provider_invocation_db: Path | None = None,
    provider_invocation_store: ProviderInvocationStore | None = None,
    application_composition: AgentCapabilityApplicationComposition | None = None,
    serving_reconciliation_targets: tuple[tuple[str, str], ...] = (),
    composition_builder: Callable[
        [
            SerializedSealExecutionContinuationStateStore,
            ProviderInvocationStore,
            ProductionProcessComposition,
        ],
        ProductionProcessComposition,
    ]
    | None = None,
    owned_governed_external_work_host_store_bundle: (
        OwnedGovernedExternalWorkHostStoreBundle | None
    ) = None,
) -> ProductionProcessLifecycle:
    """
    Open one sanctioned production process lifecycle for supported A1 co-located deployment.

    Requires a genuine ``RuntimeDeploymentAdapter``, durable ``SqliteAgentDistributionStoreBundle``,
    and explicit production governance — no reference fake adapter or harness policy fallback.
    """
    require_production_control_plane_governance(governance)
    if provider_invocation_db is not None and provider_invocation_store is not None:
        raise ProductionProcessLifecycleError(
            "provider_invocation_db and provider_invocation_store are mutually exclusive",
        )

    _reserve_production_opening_slot()

    owned_invocation: SQLiteProviderInvocationStore | None = None
    resolved_invocation: ProviderInvocationStore | None = provider_invocation_store
    try:
        if provider_invocation_db is not None:
            owned_invocation = open_provider_invocation_store_at(provider_invocation_db)
            resolved_invocation = owned_invocation
        continuation_store = open_production_execution_continuation_state_store(
            continuation_state_file,
            missing_file_policy=missing_file_policy,
        )
    except BaseException as startup_exc:
        _abort_production_opening_slot()
        cleanup_errors = release_lifecycle_owned_stores(
            continuation_store=None,
            owned_invocation=owned_invocation,
            continuation_closed=True,
            invocation_closed=False,
        )
        raise_startup_failure_with_cleanup(
            startup_exc,
            cleanup_errors,
            resource_scope="production",
        )
        raise AssertionError("unreachable startup cleanup")

    durable_runtime = None
    lifecycle: ProductionProcessLifecycle | None = None
    try:
        durable_runtime = build_durable_production_agent_platform_runtime(
            agent_distribution_db,
        )
        agent_platform_runtime = durable_runtime.agent_platform_runtime
        base_composition = ProductionProcessComposition(
            agent_platform_runtime=agent_platform_runtime,
            agent_capability_runtime=None,
        )
        lifecycle_services = wire_production_lifecycle_services(
            base_composition,
            durable_store_bundle=durable_runtime.distribution_store_bundle,
            deployment_adapter=deployment_adapter,
            trusted_artifact_parent_roots=trusted_artifact_parent_roots,
        )
        if composition_builder is not None:
            if resolved_invocation is None:
                raise ProductionProcessLifecycleError(
                    "composition_builder requires provider_invocation_db or "
                    "provider_invocation_store",
                )
            composition = composition_builder(
                continuation_store,
                resolved_invocation,
                base_composition,
            )
        else:
            composition = assemble_durable_production_process_composition(
                agent_platform_runtime=agent_platform_runtime,
                application_composition=application_composition,
                provider_invocation_store=resolved_invocation,
                execution_continuation_state_store=continuation_store,
            )
        if composition.agent_platform_runtime is not agent_platform_runtime:
            raise ProductionProcessLifecycleError(
                "production composition must use durable agent_platform_runtime instance",
            )
        if composition.execution_continuation_state_store is not continuation_store:
            raise ProductionProcessLifecycleError(
                "composition must bind opened execution_continuation_state_store",
            )
        if resolved_invocation is not None:
            if composition.provider_invocation_store is not resolved_invocation:
                raise ProductionProcessLifecycleError(
                    "composition must bind resolved provider_invocation_store",
                )
        if owned_governed_external_work_host_store_bundle is not None:
            bound_bundle = composition.governed_external_work_host_store_bundle
            if bound_bundle is None:
                raise ProductionProcessLifecycleError(
                    "composition must bind governed_external_work_host_store_bundle "
                    "when owned_governed_external_work_host_store_bundle is supplied",
                )
            if (
                bound_bundle is not
                owned_governed_external_work_host_store_bundle.host_store_bundle
            ):
                raise ProductionProcessLifecycleError(
                    "composition governed_external_work_host_store_bundle must match "
                    "owned_governed_external_work_host_store_bundle instance",
                )
            if not bound_bundle.is_durable:
                raise ProductionProcessLifecycleError(
                    "owned governed external work host store bundle is not durable",
                )
        lifecycle_services = wire_production_lifecycle_services(
            composition,
            durable_store_bundle=durable_runtime.distribution_store_bundle,
            deployment_adapter=deployment_adapter,
            trusted_artifact_parent_roots=trusted_artifact_parent_roots,
        )
        launcher = ProductionLifecycleLauncher(
            composition,
            services=lifecycle_services,
            mutation_authorization_boundary=governance.mutation_authorization_boundary,
            environment_tenant_resolver=governance.environment_tenant_resolver,
        )
        lifecycle = ProductionProcessLifecycle(
            composition=composition,
            launcher=launcher,
            governance=governance,
            lifecycle_services=lifecycle_services,
            durable_agent_distribution_db=agent_distribution_db,
            deployment_adapter=deployment_adapter,
            _owned_continuation_store=continuation_store,
            _owned_invocation_store=owned_invocation,
            _owned_governed_external_work_host_store_bundle=(
                owned_governed_external_work_host_store_bundle
            ),
            registry_projection_rehydrator=durable_runtime.registry_projection_rehydrator,
        )
        serving_store = durable_runtime.distribution_store_bundle.serving_store
        reconciliation_targets = (
            resolve_mandatory_production_serving_reconciliation_targets(
                serving_store,
                serving_reconciliation_targets,
            )
        )
        for app_id, env_id in reconciliation_targets:
            try:
                lifecycle.reconcile_serving_environment(app_id, env_id)
            except ProductionServingReconciliationError as reconcile_exc:
                _abort_production_opening_slot()
                cleanup_errors = _release_uncommitted_production_lifecycle_resources(
                    lifecycle=lifecycle,
                    continuation_store=continuation_store,
                    owned_invocation=owned_invocation,
                    continuation_closed=False,
                    invocation_closed=False,
                )
                raise_startup_failure_with_cleanup(
                    reconcile_exc,
                    cleanup_errors,
                    resource_scope="production",
                )
                raise AssertionError("unreachable reconciliation cleanup")
        _commit_production_active_slot(lifecycle)
        return lifecycle
    except BaseException as startup_exc:
        _abort_production_opening_slot()
        if lifecycle is not None and lifecycle._process_closed:
            raise startup_exc
        cleanup_errors = _release_uncommitted_production_lifecycle_resources(
            lifecycle=lifecycle,
            continuation_store=continuation_store,
            owned_invocation=owned_invocation,
            continuation_closed=False,
            invocation_closed=False,
        )
        raise_startup_failure_with_cleanup(
            startup_exc,
            cleanup_errors,
            resource_scope="production",
        )
        raise AssertionError("unreachable startup cleanup")


__all__ = [
    "ProductionLifecycleLauncher",
    "ProductionProcessLifecycle",
    "ProductionProcessLifecycleError",
    "ProductionServingReconciliationError",
    "ProductionServingReconciliationResult",
    "open_production_process_lifecycle",
    "reconcile_production_serving_environment_on_process_open",
]
