# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Package metadata ↔ runtime AgentContract parity validation (AGENT-CONSOLIDATION-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.agent_capability_metadata import AgentCapabilityDescriptor
from intergrax.contracts.agent_contract_meta import AgentContract


@dataclass(frozen=True, slots=True)
class AgentContractMetadataParityMismatch:
    """One architecture-field mismatch between package metadata and runtime contract."""

    field: str
    expected: object
    actual: object


class AgentContractMetadataParityError(ValueError):
    """Raised when package metadata and runtime AgentContract disagree on architecture fields."""

    def __init__(self, mismatches: tuple[AgentContractMetadataParityMismatch, ...]) -> None:
        self.mismatches = mismatches
        details = "; ".join(
            f"{item.field}: expected={item.expected!r} actual={item.actual!r}"
            for item in mismatches
        )
        super().__init__(f"agent contract metadata parity violation: {details}")


def _normalized_string_tuple(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted(item.strip() for item in values if item.strip()))


def _contract_skill_ids(contract: AgentContract) -> tuple[str, ...]:
    return _normalized_string_tuple([skill.skill_id for skill in contract.skills])


def _contract_extra_tool_ids(contract: AgentContract) -> tuple[str, ...]:
    return _normalized_string_tuple([tool.tool_id for tool in contract.extra_tools])


def validate_agent_contract_metadata_parity(
    *,
    descriptor: AgentCapabilityDescriptor,
    contract: AgentContract,
) -> None:
    """Validate package-owned metadata against an executable runtime AgentContract.

    Package-owned declarative metadata is the non-executable discovery declaration.
    Runtime ``AgentContract`` is the executable contract presented by an instantiated agent.
    For a published/installable package the two must have exact parity for the
    overlapping architecture fields. Package version is not AgentContract version.

    Compared fields: ``contract_id``, version, capabilities, declared skill ids,
    declared extra tool ids. ``allowed_tools`` and other runtime-only fields are ignored.
    """
    mismatches: list[AgentContractMetadataParityMismatch] = []

    if descriptor.contract_id != contract.id:
        mismatches.append(
            AgentContractMetadataParityMismatch(
                field="contract_id",
                expected=descriptor.contract_id,
                actual=contract.id,
            )
        )

    if descriptor.agent_version != contract.version:
        mismatches.append(
            AgentContractMetadataParityMismatch(
                field="agent_version",
                expected=descriptor.agent_version,
                actual=contract.version,
            )
        )

    descriptor_capabilities = _normalized_string_tuple(descriptor.capabilities)
    contract_capabilities = _normalized_string_tuple(contract.capabilities)
    if descriptor_capabilities != contract_capabilities:
        mismatches.append(
            AgentContractMetadataParityMismatch(
                field="capabilities",
                expected=descriptor_capabilities,
                actual=contract_capabilities,
            )
        )

    descriptor_skill_ids = _normalized_string_tuple(descriptor.skill_ids)
    contract_skill_ids = _contract_skill_ids(contract)
    if descriptor_skill_ids != contract_skill_ids:
        mismatches.append(
            AgentContractMetadataParityMismatch(
                field="skill_ids",
                expected=descriptor_skill_ids,
                actual=contract_skill_ids,
            )
        )

    descriptor_tool_ids = _normalized_string_tuple(descriptor.tool_ids)
    contract_tool_ids = _contract_extra_tool_ids(contract)
    if descriptor_tool_ids != contract_tool_ids:
        mismatches.append(
            AgentContractMetadataParityMismatch(
                field="tool_ids",
                expected=descriptor_tool_ids,
                actual=contract_tool_ids,
            )
        )

    if mismatches:
        raise AgentContractMetadataParityError(tuple(mismatches))
