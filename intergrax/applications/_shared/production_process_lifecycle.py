# © Artur Czarnecki. All rights reserved.

"""Sanctioned production process lifetime — composition + durable store ownership (PROD-Q-R1-R7-D2)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
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
    ProductionLifecycleServices,
    require_production_control_plane_governance,
    wire_production_lifecycle_services,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
)
from intergrax.applications._shared.reference_production_process_lifecycle import (
    _cleanup_errors_from_close_failure,
    _raise_primary_with_cleanup_errors,
    _raise_startup_failure_with_cleanup,
    _release_lifecycle_owned_stores,
)
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
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

_active_production_process_lifecycle_guard = threading.Lock()
_active_production_process_lifecycle: "ProductionProcessLifecycle | None" = None


class ProductionProcessLifecycleError(ValueError):
    """Invalid production process lifecycle wiring."""


class ProductionLifecycleLauncher(ReferenceProductionLifecycleLauncher):
    """Production-qualified deploy/project/activate orchestration (same semantics, production wiring only)."""


@dataclass
class ProductionProcessLifecycle:
    """One sanctioned production composition root, launcher, governance, and owned durable stores."""

    composition: ProductionProcessComposition
    launcher: ProductionLifecycleLauncher
    governance: ProductionControlPlaneGovernance
    lifecycle_services: ProductionLifecycleServices
    durable_agent_distribution_db: Path
    deployment_adapter: RuntimeDeploymentAdapter
    _owned_continuation_store: SerializedSealExecutionContinuationStateStore = field(
        repr=False,
    )
    _owned_invocation_store: SQLiteProviderInvocationStore | None = field(
        default=None,
        repr=False,
    )
    _continuation_closed: bool = field(default=False, repr=False)
    _invocation_closed: bool = field(default=False, repr=False)
    _process_closed: bool = field(default=False, repr=False)

    @property
    def execution_continuation_state_store(self) -> ExecutionContinuationStateStore:
        return self._owned_continuation_store

    @property
    def provider_invocation_store(self) -> ProviderInvocationStore | None:
        return self.composition.provider_invocation_store

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
        self._process_closed = True
        global _active_production_process_lifecycle
        with _active_production_process_lifecycle_guard:
            if _active_production_process_lifecycle is self:
                _active_production_process_lifecycle = None
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
            cleanup_errors = _cleanup_errors_from_close_failure(cleanup_exc)
            _raise_primary_with_cleanup_errors(
                body_exc,
                cleanup_errors,
                message=(
                    "production process lifecycle context exited with an error; "
                    "errors while closing resources"
                ),
            )
        return False


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
    composition_builder: Callable[
        [
            SerializedSealExecutionContinuationStateStore,
            ProviderInvocationStore,
            ProductionProcessComposition,
        ],
        ProductionProcessComposition,
    ]
    | None = None,
    allow_duplicate_process_lifecycle: bool = False,
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
    global _active_production_process_lifecycle
    with _active_production_process_lifecycle_guard:
        if (
            not allow_duplicate_process_lifecycle
            and _active_production_process_lifecycle is not None
            and not _active_production_process_lifecycle._process_closed
        ):
            raise ProductionProcessLifecycleError(
                "duplicate production process lifecycle is not permitted in one process",
            )

    owned_invocation: SQLiteProviderInvocationStore | None = None
    resolved_invocation: ProviderInvocationStore | None = provider_invocation_store
    if provider_invocation_db is not None:
        owned_invocation = open_provider_invocation_store_at(provider_invocation_db)
        resolved_invocation = owned_invocation

    try:
        continuation_store = open_production_execution_continuation_state_store(
            continuation_state_file,
            missing_file_policy=missing_file_policy,
        )
    except BaseException as startup_exc:
        cleanup_errors = _release_lifecycle_owned_stores(
            continuation_store=None,
            owned_invocation=owned_invocation,
            continuation_closed=True,
            invocation_closed=False,
        )
        _raise_startup_failure_with_cleanup(startup_exc, cleanup_errors)
        raise AssertionError("unreachable startup cleanup")

    durable_runtime = None
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
            composition = create_reference_production_process_composition(
                application_composition=application_composition,
                provider_invocation_store=resolved_invocation,
                execution_continuation_state_store=continuation_store,
            )
            composition = ProductionProcessComposition(
                agent_platform_runtime=agent_platform_runtime,
                agent_capability_runtime=composition.agent_capability_runtime,
                provider_invocation_store=composition.provider_invocation_store,
                execution_continuation_state_store=continuation_store,
                serving_runtime_handoff=composition.serving_runtime_handoff,
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
        )
        with _active_production_process_lifecycle_guard:
            if (
                not allow_duplicate_process_lifecycle
                and _active_production_process_lifecycle is not None
                and not _active_production_process_lifecycle._process_closed
            ):
                raise ProductionProcessLifecycleError(
                    "duplicate production process lifecycle is not permitted in one process",
                )
            _active_production_process_lifecycle = lifecycle
        return lifecycle
    except BaseException as startup_exc:
        cleanup_errors = _release_lifecycle_owned_stores(
            continuation_store=continuation_store,
            owned_invocation=owned_invocation,
            continuation_closed=False,
            invocation_closed=False,
        )
        _raise_startup_failure_with_cleanup(startup_exc, cleanup_errors)
        raise AssertionError("unreachable startup cleanup")


__all__ = [
    "ProductionLifecycleLauncher",
    "ProductionProcessLifecycle",
    "ProductionProcessLifecycleError",
    "open_production_process_lifecycle",
]
