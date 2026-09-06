# © Artur Czarnecki. All rights reserved.

"""Default runtime inspection provider composition (P1.4)."""

from __future__ import annotations

from intergrax.applications._shared.runtime_inspection.providers.capability_dependency import (
    capability_dependency_inspection_provider,
)
from intergrax.applications._shared.runtime_inspection.providers.execution_binding import (
    execution_binding_inspection_provider,
)
from intergrax.applications._shared.runtime_inspection.providers.execution_environment import (
    execution_environment_inspection_provider,
)
from intergrax.applications._shared.runtime_inspection.providers.profile_revision import (
    profile_revision_inspection_provider,
)
from intergrax.applications.contracts.runtime_inspection.provider import (
    RuntimeInspectionProvider,
)


def default_runtime_inspection_providers() -> tuple[RuntimeInspectionProvider, ...]:
    """Explicit immutable provider set — no global mutable registry."""
    return (
        profile_revision_inspection_provider(),
        execution_binding_inspection_provider(),
        execution_environment_inspection_provider(),
        capability_dependency_inspection_provider(),
    )
