# © Artur Czarnecki. All rights reserved.

"""Agent assembly validation at register time (Phase AS-1, AS-2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState

_AGENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class AgentAssemblyValidationResult:
    """Outcome of agent contract assembly validation."""

    valid: bool
    errors: tuple[str, ...] = ()


class AgentAssemblyError(ValueError):
    """Raised when an agent contract fails assembly validation."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def validate_contract_metadata(contract: AgentContract) -> AgentAssemblyValidationResult:
    """Validate declarative contract metadata required for agent assembly."""
    errors: list[str] = []

    agent_id = contract.id.strip()
    if not agent_id:
        errors.append("AgentContract.id must be non-empty")
    elif not _AGENT_ID_PATTERN.match(agent_id):
        errors.append(
            "AgentContract.id must match ^[a-z][a-z0-9_-]*$ "
            f"(got {contract.id!r})"
        )

    if not contract.name.strip():
        errors.append("AgentContract.name must be non-empty")

    if not contract.description.strip():
        errors.append("AgentContract.description must be non-empty")

    capabilities = [item.strip() for item in contract.capabilities if item.strip()]
    if not capabilities:
        errors.append("AgentContract.capabilities must declare at least one capability id")

    if (contract.skills or contract.extra_tools) and contract.allowed_tools:
        errors.append(
            "AgentContract.allowed_tools must be empty at author time when "
            "skills or extra_tools are declared; resolution happens in AgentRegistry.register"
        )

    return AgentAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def validate_lifecycle_metadata(contract: AgentContract) -> AgentAssemblyValidationResult:
    """Validate lifecycle metadata consistency on AgentContract (Phase AS-2)."""
    errors: list[str] = []

    if contract.production_eligible:
        if not (contract.owner_team or "").strip():
            errors.append("production_eligible agents require owner_team")
        if not (contract.owner_contact or "").strip():
            errors.append("production_eligible agents require owner_contact")
        if not (contract.runbook_ref or "").strip():
            errors.append("production_eligible agents require runbook_ref")

    return AgentAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def validate_agent_assembly(contract: AgentContract) -> AgentAssemblyValidationResult:
    """Run full agent assembly validation."""
    metadata = validate_contract_metadata(contract)
    lifecycle = validate_lifecycle_metadata(contract)
    errors = (*metadata.errors, *lifecycle.errors)
    return AgentAssemblyValidationResult(valid=not errors, errors=errors)


def assert_agent_assembly_valid(contract: AgentContract) -> None:
    """Raise :class:`AgentAssemblyError` when assembly validation fails."""
    result = validate_agent_assembly(contract)
    if not result.valid:
        raise AgentAssemblyError(result.errors)
