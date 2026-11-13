# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent UPGRADE compatibility check (ADR-PDC-01, PDC-D1-R011).

Pure evaluation of a candidate package release against the active release of the
same installation slot and the configuration bound to that slot. Version policy is
the registry SemVer policy (``is_compatible_runtime``); this module is not a
registry, manager or lifecycle authority and mutates nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from intergrax.agent_distribution.binding import ApplicationAgentBinding
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.semver import SemVer
from intergrax.runtime.registry.semver_compat import is_compatible_runtime


class AgentUpgradeRejectionReason(StrEnum):
    """Typed, machine-readable reasons an upgrade candidate is rejected."""

    PACKAGE_LINE_MISMATCH = "package_line_mismatch"
    VERSION_UNPARSEABLE = "version_unparseable"
    NOT_AN_UPGRADE = "not_an_upgrade"
    PACKAGE_VERSION_INCOMPATIBLE = "package_version_incompatible"
    PRIOR_CONTRACT_UNRESOLVED = "prior_contract_unresolved"
    CONTRACT_REMOVED = "contract_removed"
    CONTRACT_VERSION_INCOMPATIBLE = "contract_version_incompatible"
    CAPABILITY_REMOVED = "capability_removed"
    BOUND_TOOL_REMOVED = "bound_tool_removed"


@dataclass(frozen=True, slots=True)
class UpgradeAdmissionViolation:
    reason: AgentUpgradeRejectionReason
    subject: str
    detail: str


@dataclass(frozen=True, slots=True)
class UpgradeAdmissionReport:
    violations: tuple[UpgradeAdmissionViolation, ...] = ()

    @property
    def admitted(self) -> bool:
        return not self.violations

    @property
    def primary_reason(self) -> AgentUpgradeRejectionReason | None:
        return self.violations[0].reason if self.violations else None


def _contract_tool_ids(contract: AgentContract) -> frozenset[str]:
    tool_ids: set[str] = set()
    for skill in contract.skills:
        tool_ids.update(skill.tool_ids)
    for tool in contract.extra_tools:
        tool_ids.add(tool.tool_id)
    return frozenset(tool_ids)


def _union_tool_ids(contracts: Mapping[str, AgentContract]) -> frozenset[str]:
    tool_ids: set[str] = set()
    for contract in contracts.values():
        tool_ids.update(_contract_tool_ids(contract))
    return frozenset(tool_ids)


def _semver_key(value: str) -> tuple[int, int, int] | None:
    try:
        parsed = SemVer.parse(value)
    except ValueError:
        return None
    return (parsed.major, parsed.minor, parsed.patch)


def evaluate_agent_upgrade_admission(
    *,
    prior_identity: AgentPackageIdentity,
    candidate_identity: AgentPackageIdentity,
    prior_contracts: Mapping[str, AgentContract | None],
    candidate_contracts: Mapping[str, AgentContract],
    bindings: Sequence[ApplicationAgentBinding],
) -> UpgradeAdmissionReport:
    """Check the candidate release against the active release and its bound configuration.

    ``prior_contracts`` maps every contract id the active package declares to its stored
    ``AgentContract`` authority, or ``None`` when the store cannot resolve it (fail closed).
    """
    violations: list[UpgradeAdmissionViolation] = []

    if prior_identity.distribution_package_id != candidate_identity.distribution_package_id:
        violations.append(
            UpgradeAdmissionViolation(
                reason=AgentUpgradeRejectionReason.PACKAGE_LINE_MISMATCH,
                subject=candidate_identity.distribution_package_id,
                detail=(
                    "candidate belongs to package line "
                    f"{candidate_identity.distribution_package_id!r}, slot holds "
                    f"{prior_identity.distribution_package_id!r}"
                ),
            )
        )
        return UpgradeAdmissionReport(tuple(violations))

    prior_key = _semver_key(prior_identity.package_version)
    candidate_key = _semver_key(candidate_identity.package_version)
    if prior_key is None or candidate_key is None:
        violations.append(
            UpgradeAdmissionViolation(
                reason=AgentUpgradeRejectionReason.VERSION_UNPARSEABLE,
                subject=candidate_identity.package_version,
                detail="package versions must be SemVer to evaluate an upgrade",
            )
        )
        return UpgradeAdmissionReport(tuple(violations))

    if candidate_key <= prior_key:
        violations.append(
            UpgradeAdmissionViolation(
                reason=AgentUpgradeRejectionReason.NOT_AN_UPGRADE,
                subject=candidate_identity.package_version,
                detail=(
                    f"candidate {candidate_identity.package_version} is not newer than "
                    f"active {prior_identity.package_version}; use rollback to go back"
                ),
            )
        )
        return UpgradeAdmissionReport(tuple(violations))

    package_compat = is_compatible_runtime(
        prior_identity.package_version,
        candidate_identity.package_version,
    )
    if not package_compat.compatible:
        violations.append(
            UpgradeAdmissionViolation(
                reason=AgentUpgradeRejectionReason.PACKAGE_VERSION_INCOMPATIBLE,
                subject=candidate_identity.package_version,
                detail=package_compat.reason,
            )
        )

    resolved_prior: dict[str, AgentContract] = {}
    for contract_id, prior_contract in sorted(prior_contracts.items()):
        if prior_contract is None:
            violations.append(
                UpgradeAdmissionViolation(
                    reason=AgentUpgradeRejectionReason.PRIOR_CONTRACT_UNRESOLVED,
                    subject=contract_id,
                    detail="active package contract authority could not be resolved",
                )
            )
            continue
        resolved_prior[contract_id] = prior_contract

    for contract_id, prior_contract in sorted(resolved_prior.items()):
        candidate_contract = candidate_contracts.get(contract_id)
        if candidate_contract is None:
            violations.append(
                UpgradeAdmissionViolation(
                    reason=AgentUpgradeRejectionReason.CONTRACT_REMOVED,
                    subject=contract_id,
                    detail="candidate package no longer declares this contract",
                )
            )
            continue
        try:
            contract_compat = is_compatible_runtime(
                prior_contract.version,
                candidate_contract.version,
            )
        except ValueError:
            violations.append(
                UpgradeAdmissionViolation(
                    reason=AgentUpgradeRejectionReason.VERSION_UNPARSEABLE,
                    subject=contract_id,
                    detail="contract versions must be SemVer to evaluate an upgrade",
                )
            )
            continue
        if not contract_compat.compatible:
            violations.append(
                UpgradeAdmissionViolation(
                    reason=AgentUpgradeRejectionReason.CONTRACT_VERSION_INCOMPATIBLE,
                    subject=contract_id,
                    detail=(f"{prior_contract.version} -> {candidate_contract.version}: {contract_compat.reason}"),
                )
            )
        removed_capabilities = sorted(set(prior_contract.capabilities) - set(candidate_contract.capabilities))
        if removed_capabilities:
            violations.append(
                UpgradeAdmissionViolation(
                    reason=AgentUpgradeRejectionReason.CAPABILITY_REMOVED,
                    subject=contract_id,
                    detail="candidate dropped capabilities: " + ", ".join(removed_capabilities),
                )
            )

    prior_tools = _union_tool_ids(resolved_prior)
    candidate_tools = _union_tool_ids(candidate_contracts)
    for binding in sorted(bindings, key=lambda item: item.application_binding_id):
        if binding.tombstone or binding.policy_overrides is None:
            continue
        for tool_id in binding.policy_overrides.tool_allowlist:
            if tool_id in prior_tools and tool_id not in candidate_tools:
                violations.append(
                    UpgradeAdmissionViolation(
                        reason=AgentUpgradeRejectionReason.BOUND_TOOL_REMOVED,
                        subject=f"{binding.application_binding_id}:{tool_id}",
                        detail=("binding allow-lists a tool the candidate package no longer provides"),
                    )
                )

    return UpgradeAdmissionReport(tuple(violations))


__all__ = [
    "AgentUpgradeRejectionReason",
    "UpgradeAdmissionReport",
    "UpgradeAdmissionViolation",
    "evaluate_agent_upgrade_admission",
]
