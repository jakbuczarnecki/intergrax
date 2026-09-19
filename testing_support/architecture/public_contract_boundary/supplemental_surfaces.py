# © Artur Czarnecki. All rights reserved.

"""Declarative supplemental public contract surfaces outside intergrax/**/contracts/**."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.architecture.public_contract_boundary.models import RemovalStage


@dataclass(frozen=True, slots=True)
class SupplementalPublicContractSurface:
    repo_relative_path: str
    owner_domain: str
    remediation_stage: RemovalStage


SUPPLEMENTAL_PUBLIC_CONTRACT_SURFACES: tuple[SupplementalPublicContractSurface, ...] = (
    SupplementalPublicContractSurface(
        repo_relative_path="intergrax/agents/agent_contract.py",
        owner_domain="agents",
        remediation_stage=RemovalStage.EBH_2B,
    ),
    SupplementalPublicContractSurface(
        repo_relative_path="intergrax/agents/uaep_protocol.py",
        owner_domain="agents",
        remediation_stage=RemovalStage.EBH_2B,
    ),
)
