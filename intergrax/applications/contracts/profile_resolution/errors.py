# © Artur Czarnecki. All rights reserved.

"""Profile resolution failure contracts (P1.1)."""

from __future__ import annotations

from intergrax.applications.contracts.profile_resolution.layer import ProfileLayer


class ProfileResolutionError(RuntimeError):
    """Resolution failed before execution may proceed."""


class ProfileLayerConflictError(ProfileResolutionError):
    """Duplicate canonical layer input."""

    def __init__(self, layer: ProfileLayer) -> None:
        super().__init__(f"duplicate profile layer input: {layer.value}")
        self.layer = layer


class ProfileOverrideRejectedError(ProfileResolutionError):
    """Mandatory override could not be satisfied."""

    def __init__(self, path: str, layer: ProfileLayer, reason: str) -> None:
        super().__init__(f"profile override rejected at {path} from {layer.value}: {reason}")
        self.path = path
        self.layer = layer
        self.reason = reason


class EffectiveProfileRevisionError(RuntimeError):
    """Effective profile revision lifecycle failure."""


class EffectiveProfileRevisionConflictError(EffectiveProfileRevisionError):
    """Append-only store rejected duplicate revision identity."""


class MissingPinnedEffectiveProfileRevisionError(EffectiveProfileRevisionError):
    """Required pinned revision is absent — fail closed."""

    def __init__(self, *, tenant_id: str, execution_id: str) -> None:
        super().__init__(
            f"missing pinned effective profile revision for execution {execution_id!r} "
            f"in tenant {tenant_id!r}"
        )
        self.tenant_id = tenant_id
        self.execution_id = execution_id


class EffectiveProfileActivationError(EffectiveProfileRevisionError):
    """Effective profile activation lifecycle failure."""


class EffectiveProfileActivationConflictError(EffectiveProfileActivationError):
    """CAS activation conflict — expected active revision does not match current."""


class EffectiveProfileActivationRevisionNotFoundError(EffectiveProfileActivationError):
    """Candidate revision is absent from canonical revision store."""

    def __init__(self, *, revision_id: str, scope: EffectiveProfileRevisionScope) -> None:
        super().__init__(
            f"candidate revision {revision_id!r} not found for scope "
            f"{scope.application_id!r}/{scope.tenant_id!r}",
        )
        self.revision_id = revision_id
        self.scope = scope


class EffectiveProfileActivationScopeMismatchError(EffectiveProfileActivationError):
    """Candidate revision scope does not match activation scope."""


class EffectiveProfileActivationRejectedError(EffectiveProfileActivationError):
    """Candidate failed required activation eligibility checks."""


class EffectiveProfileActivationPersistenceError(EffectiveProfileActivationError):
    """Active pointer persistence failed — prior active remains authoritative."""


class MissingActiveEffectiveProfileRevisionError(EffectiveProfileActivationError):
    """No active revision is published for scope — fail closed for new admission."""

    def __init__(self, *, scope: EffectiveProfileRevisionScope) -> None:
        super().__init__(
            f"no active effective profile revision for scope "
            f"{scope.application_id!r}/{scope.tenant_id!r}",
        )
        self.scope = scope
