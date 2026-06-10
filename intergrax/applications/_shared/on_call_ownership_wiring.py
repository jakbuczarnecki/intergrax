# © Artur Czarnecki. All rights reserved.

"""On-call ownership model wiring for production components (AUDIT-IDEAL-30.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.architecture.production_ownership import (
    ProductionOwnerMetadata,
    ProductionOwnershipDecision,
    ProductionOwnershipEvidence,
    evaluate_production_ownership,
)


@dataclass(frozen=True, slots=True)
class OnCallOwnershipRecord:
    agent_id: str
    team: str
    on_call: str
    approved: bool


@dataclass(frozen=True, slots=True)
class OnCallOwnershipRegistry:
    enabled: bool
    records: tuple[OnCallOwnershipRecord, ...]


def _record_from_contract(contract: AgentContract) -> OnCallOwnershipRecord:
    owner = ProductionOwnerMetadata(
        team=contract.owner_team or "",
        owner=contract.owner_contact or "",
        on_call=contract.on_call_contact or contract.owner_contact or "",
        escalation_channel=contract.owner_contact or "",
    )
    evidence = ProductionOwnershipEvidence(
        agent_id=contract.id,
        agent_version=contract.version,
        production_eligible=contract.production_eligible,
        owner=owner,
        runbook_ref=contract.runbook_ref or "",
    )
    decision: ProductionOwnershipDecision = evaluate_production_ownership(evidence)
    return OnCallOwnershipRecord(
        agent_id=contract.id,
        team=owner.team,
        on_call=owner.on_call,
        approved=decision.approved,
    )


def resolve_on_call_ownership_registry(
    env: ApplicationEnvironmentProfile,
    *,
    contracts: tuple[AgentContract, ...],
) -> OnCallOwnershipRegistry:
    """Validate on-call ownership for production-eligible agents on product hosts."""
    enabled = env.application_profile is ApplicationProfile.PRODUCT
    records = tuple(_record_from_contract(contract) for contract in contracts)
    return OnCallOwnershipRegistry(enabled=enabled, records=records)
