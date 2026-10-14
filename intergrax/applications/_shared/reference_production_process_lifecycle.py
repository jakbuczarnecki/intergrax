# © Artur Czarnecki. All rights reserved.

"""Reference production process lifetime — composition + durable store ownership (PROD-Q-R1-R6 / PIS-02).

Opens the sanctioned durable continuation store and SQLite ``ProviderInvocationStore`` once per
process, retains both on ``ProductionProcessComposition``, and closes them at the authorized
process boundary. Application factories and FastAPI shutdown hooks are consumers only — they must
not close composition-owned stores.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from intergrax.applications._shared.process_lifecycle_resource_cleanup import (
    cleanup_errors_from_close_failure,
    raise_primary_with_cleanup_errors,
    raise_startup_failure_with_cleanup,
    release_lifecycle_owned_stores,
)
from intergrax.applications._shared.production_execution_continuation_store import (
    MissingContinuationFilePolicy,
    SerializedSealExecutionContinuationStateStore,
    open_production_execution_continuation_state_store,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    ReferenceProductionControlPlaneGovernance,
    wire_governed_reference_production_launcher,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
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


def _close_owned_resource(
    close: Callable[[], None],
    *,
    mark_closed: Callable[[], None],
) -> BaseException | None:
    try:
        close()
    except BaseException as exc:
        return exc
    mark_closed()
    return None


def _raise_grouped_exceptions(
    message: str,
    exceptions: list[BaseException],
    *,
    from_exc: BaseException | None = None,
) -> None:
    if all(isinstance(exc, Exception) for exc in exceptions):
        raise ExceptionGroup(message, exceptions) from from_exc
    raise BaseExceptionGroup(message, exceptions) from from_exc


class ReferenceProductionProcessLifecycleError(ValueError):
    """Invalid reference production process lifecycle wiring."""


@dataclass
class ReferenceProductionProcessLifecycle:
    """One process composition root, launcher, and owned durable stores."""

    composition: ProductionProcessComposition
    launcher: ReferenceProductionLifecycleLauncher
    governance: ReferenceProductionControlPlaneGovernance
    _owned_continuation_store: SerializedSealExecutionContinuationStateStore = field(
        repr=False,
    )
    _owned_invocation_store: SQLiteProviderInvocationStore | None = field(
        default=None,
        repr=False,
    )
    _continuation_closed: bool = field(default=False, repr=False)
    _invocation_closed: bool = field(default=False, repr=False)

    @property
    def execution_continuation_state_store(self) -> ExecutionContinuationStateStore:
        return self._owned_continuation_store

    @property
    def provider_invocation_store(self) -> ProviderInvocationStore | None:
        return self.composition.provider_invocation_store

    def close_process_resources(self) -> None:
        """Release lifecycle-owned continuation and invocation resources exactly once."""
        cleanup_errors: list[BaseException] = []
        if not self._continuation_closed:
            err = _close_owned_resource(
                self._owned_continuation_store.close,
                mark_closed=lambda: setattr(self, "_continuation_closed", True),
            )
            if err is not None:
                cleanup_errors.append(err)
        owned_invocation = self._owned_invocation_store
        if owned_invocation is not None and not self._invocation_closed:
            err = _close_owned_resource(
                owned_invocation.close,
                mark_closed=lambda: setattr(self, "_invocation_closed", True),
            )
            if err is not None:
                cleanup_errors.append(err)
        if cleanup_errors:
            if len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            _raise_grouped_exceptions(
                "errors while closing reference production process lifecycle resources",
                cleanup_errors,
            )

    def __enter__(self) -> ReferenceProductionProcessLifecycle:
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
                    "reference production process lifecycle context exited with an error; "
                    "errors while closing resources"
                ),
            )
        return False


def open_reference_production_process_lifecycle(
    *,
    continuation_state_file: Path,
    env: ApplicationEnvironmentProfile,
    missing_file_policy: MissingContinuationFilePolicy = "fail",
    provider_invocation_db: Path | None = None,
    provider_invocation_store: ProviderInvocationStore | None = None,
    application_composition: AgentCapabilityApplicationComposition | None = None,
    composition_builder: Callable[
        [
            SerializedSealExecutionContinuationStateStore,
            ProviderInvocationStore,
        ],
        ProductionProcessComposition,
    ]
    | None = None,
    service_id: str = "reference-production-lifecycle",
    tenant_id: str | None = None,
) -> ReferenceProductionProcessLifecycle:
    """
    Open continuation + invocation persistence, construct composition + governed launcher.

    When ``provider_invocation_db`` is set, this lifecycle opens and owns the SQLite store.
    An externally supplied ``provider_invocation_store`` is never replaced; the lifecycle does
    not close caller-owned stores.

    The returned lifecycle object must stay open for the full serve lifetime and
    ``close_process_resources()`` (or context-manager exit) must run at process shutdown.
    """
    if provider_invocation_db is not None and provider_invocation_store is not None:
        raise ReferenceProductionProcessLifecycleError(
            "provider_invocation_db and provider_invocation_store are mutually exclusive; "
            "supply one binding only",
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
        cleanup_errors = release_lifecycle_owned_stores(
            continuation_store=None,
            owned_invocation=owned_invocation,
            continuation_closed=True,
            invocation_closed=False,
        )
        raise_startup_failure_with_cleanup(
            startup_exc,
            cleanup_errors,
            resource_scope="reference production",
        )

    try:
        if composition_builder is not None:
            if resolved_invocation is None:
                raise ReferenceProductionProcessLifecycleError(
                    "composition_builder requires provider_invocation_db or "
                    "provider_invocation_store",
                )
            composition = composition_builder(continuation_store, resolved_invocation)
        else:
            composition = create_reference_production_process_composition(
                application_composition=application_composition,
                provider_invocation_store=resolved_invocation,
                execution_continuation_state_store=continuation_store,
            )
        bound_continuation = composition.execution_continuation_state_store
        if bound_continuation is not continuation_store:
            raise ReferenceProductionProcessLifecycleError(
                "composition must bind the opened execution_continuation_state_store "
                "instance on ProductionProcessComposition",
            )
        if resolved_invocation is not None:
            bound_invocation = composition.provider_invocation_store
            if bound_invocation is not resolved_invocation:
                raise ReferenceProductionProcessLifecycleError(
                    "composition must bind the resolved provider_invocation_store "
                    "instance on ProductionProcessComposition",
                )
        launcher, governance = wire_governed_reference_production_launcher(
            composition,
            env,
            service_id=service_id,
            tenant_id=tenant_id,
        )
        return ReferenceProductionProcessLifecycle(
            composition=composition,
            launcher=launcher,
            governance=governance,
            _owned_continuation_store=continuation_store,
            _owned_invocation_store=owned_invocation,
        )
    except BaseException as startup_exc:
        cleanup_errors = release_lifecycle_owned_stores(
            continuation_store=continuation_store,
            owned_invocation=owned_invocation,
            continuation_closed=False,
            invocation_closed=False,
        )
        raise_startup_failure_with_cleanup(
            startup_exc,
            cleanup_errors,
            resource_scope="reference production",
        )


__all__ = [
    "ReferenceProductionProcessLifecycle",
    "ReferenceProductionProcessLifecycleError",
    "open_reference_production_process_lifecycle",
]
