# © Artur Czarnecki. All rights reserved.

"""Reference production process lifetime — composition + launcher continuation ownership (PROD-Q-R1-R6).

Opens the sanctioned durable continuation store once per process, retains it on
``ProductionProcessComposition``, and closes it at the authorized process boundary.
Application factories and FastAPI shutdown hooks are consumers only — they must not
close the composition-owned store.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from intergrax.applications._shared.production_agent_capability_runtime import (
        AgentCapabilityApplicationComposition,
    )


class ReferenceProductionProcessLifecycleError(ValueError):
    """Invalid reference production process lifecycle wiring."""


@dataclass
class ReferenceProductionProcessLifecycle:
    """One process composition root, launcher, and owned continuation store."""

    composition: ProductionProcessComposition
    launcher: ReferenceProductionLifecycleLauncher
    governance: ReferenceProductionControlPlaneGovernance
    _owned_continuation_store: SerializedSealExecutionContinuationStateStore = field(
        repr=False,
    )
    _continuation_closed: bool = field(default=False, repr=False)

    @property
    def execution_continuation_state_store(self) -> ExecutionContinuationStateStore:
        return self._owned_continuation_store

    def close_process_resources(self) -> None:
        """Release composition-owned continuation writer lease exactly once."""
        if self._continuation_closed:
            return
        self._owned_continuation_store.close()
        self._continuation_closed = True

    def __enter__(self) -> ReferenceProductionProcessLifecycle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.close_process_resources()
        return False


def open_reference_production_process_lifecycle(
    *,
    continuation_state_file: Path,
    env: ApplicationEnvironmentProfile,
    missing_file_policy: MissingContinuationFilePolicy = "fail",
    provider_invocation_store: ProviderInvocationStore | None = None,
    application_composition: AgentCapabilityApplicationComposition | None = None,
    composition_builder: Callable[
        [SerializedSealExecutionContinuationStateStore],
        ProductionProcessComposition,
    ]
    | None = None,
    service_id: str = "reference-production-lifecycle",
    tenant_id: str | None = None,
) -> ReferenceProductionProcessLifecycle:
    """
    Open continuation persistence, construct composition + governed launcher, retain store.

    The returned lifecycle object must stay open for the full serve lifetime and
    ``close_process_resources()`` (or context-manager exit) must run at process shutdown.
    """
    store = open_production_execution_continuation_state_store(
        continuation_state_file,
        missing_file_policy=missing_file_policy,
    )
    try:
        if composition_builder is not None:
            composition = composition_builder(store)
        else:
            composition = create_reference_production_process_composition(
                application_composition=application_composition,
                provider_invocation_store=provider_invocation_store,
                execution_continuation_state_store=store,
            )
        bound = composition.execution_continuation_state_store
        if bound is not store:
            raise ReferenceProductionProcessLifecycleError(
                "composition_builder must bind the opened execution_continuation_state_store "
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
            _owned_continuation_store=store,
        )
    except Exception:
        store.close()
        raise


__all__ = [
    "ReferenceProductionProcessLifecycle",
    "ReferenceProductionProcessLifecycleError",
    "open_reference_production_process_lifecycle",
]
