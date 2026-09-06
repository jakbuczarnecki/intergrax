# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral effective execution environment contracts (P1.8)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.tools.core.contracts import ToolContract, ToolIsolationRequirement


class FilesystemAccess(StrEnum):
    NONE = "none"
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"


class NetworkAccess(StrEnum):
    NONE = "none"
    RESTRICTED = "restricted"
    ALLOWED = "allowed"


class ProcessExecution(StrEnum):
    DENIED = "denied"
    SANDBOXED = "sandboxed"


class PrivilegeMode(StrEnum):
    STANDARD = "standard"
    PRIVILEGED = "privileged"


class ExecutionEnvironmentProviderKind(StrEnum):
    NONE = "none"
    LOCAL = "local"
    HOSTED = "hosted"


class ExecutionEnvironmentProviderRef(BaseModel):
    """Safe immutable provider identity — no session handles or credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    provider_kind: ExecutionEnvironmentProviderKind


class ExecutionEnvironmentProvenance(BaseModel):
    """Explainability for narrowing decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_contribution: str
    requirement_contribution: str
    provider_contribution: str
    decision: str
    reason_codes: tuple[str, ...] = Field(default_factory=tuple)


class ProfileIsolationAuthority(BaseModel):
    """Isolation ceiling derived from ``ApplicationEnvironmentProfile`` — not authority itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filesystem_access: FilesystemAccess = FilesystemAccess.NONE
    network_access: NetworkAccess = NetworkAccess.NONE
    process_execution: ProcessExecution = ProcessExecution.DENIED
    privilege_mode: PrivilegeMode = PrivilegeMode.STANDARD
    sandbox_configured: bool = False


class ExecutionEnvironmentRequirement(BaseModel):
    """Minimal runtime requirement for one operation — not configuration authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sandbox_required: bool = False
    filesystem_access: FilesystemAccess = FilesystemAccess.NONE
    network_access: NetworkAccess = NetworkAccess.NONE
    process_execution: ProcessExecution = ProcessExecution.DENIED
    privilege_mode: PrivilegeMode = PrivilegeMode.STANDARD

    @classmethod
    def none(cls) -> ExecutionEnvironmentRequirement:
        return cls()

    @classmethod
    def from_tool_isolation(cls, isolation: ToolIsolationRequirement) -> ExecutionEnvironmentRequirement:
        if isolation is ToolIsolationRequirement.SANDBOX:
            return cls(
                sandbox_required=True,
                filesystem_access=FilesystemAccess.WORKSPACE_WRITE,
                network_access=NetworkAccess.NONE,
                process_execution=ProcessExecution.SANDBOXED,
            )
        return cls.none()

    @classmethod
    def from_tool_contract(cls, contract: ToolContract) -> ExecutionEnvironmentRequirement:
        return cls.from_tool_isolation(contract.isolation_requirement)


class SandboxProviderCapabilities(BaseModel):
    """Provider-neutral capability projection — enforced facts, not intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_ref: ExecutionEnvironmentProviderRef
    filesystem_access: FilesystemAccess
    network_access: NetworkAccess
    process_execution: ProcessExecution
    supports_sandboxed_exec: bool
    supports_workspace_write: bool
    supports_network_isolation: bool | None = None


class EffectiveExecutionEnvironment(BaseModel):
    """Immutable resolved runtime semantics — derived projection only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filesystem_access: FilesystemAccess
    network_access: NetworkAccess
    process_execution: ProcessExecution
    privilege_mode: PrivilegeMode
    sandbox_required: bool
    provider_ref: ExecutionEnvironmentProviderRef
    provenance: ExecutionEnvironmentProvenance


class ExecutionEnvironmentResolutionFailureReason(StrEnum):
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    AUTHORITY_VIOLATION = "authority_violation"
    REQUIREMENT_UNSATISFIED = "requirement_unsatisfied"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_CAPABILITY_UNSATISFIED = "provider_capability_unsatisfied"


class ExecutionEnvironmentResolutionFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: ExecutionEnvironmentResolutionFailureReason
    message: str
    reason_codes: tuple[str, ...] = Field(default_factory=tuple)


class ExecutionEnvironmentResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: EffectiveExecutionEnvironment | None = None
    failure: ExecutionEnvironmentResolutionFailure | None = None

    def require_environment(self) -> EffectiveExecutionEnvironment:
        if self.environment is None:
            failure = self.failure
            detail = failure.message if failure is not None else "execution_environment_unresolved"
            raise ExecutionEnvironmentResolutionError(detail, failure=failure)
        return self.environment


class ExecutionEnvironmentResolutionError(RuntimeError):
    """Fail-closed resolution boundary."""

    def __init__(
        self,
        message: str,
        *,
        failure: ExecutionEnvironmentResolutionFailure | None = None,
    ) -> None:
        super().__init__(message)
        self.failure = failure


class ExecutionEnvironmentAuthorityUnavailableError(ExecutionEnvironmentResolutionError):
    """Effective profile authority is unavailable for resolution."""


class ExecutionEnvironmentAuthorityViolationError(ExecutionEnvironmentResolutionError):
    """Downstream requirement exceeds profile isolation authority."""


class ExecutionEnvironmentRequirementUnsatisfiedError(ExecutionEnvironmentResolutionError):
    """Resolved environment does not satisfy the stated requirement."""


class ExecutionEnvironmentProviderUnavailableError(ExecutionEnvironmentResolutionError):
    """Required sandbox substrate is unavailable."""


class ExecutionEnvironmentProviderCapabilityUnsatisfiedError(ExecutionEnvironmentResolutionError):
    """Selected provider lacks required capability."""
