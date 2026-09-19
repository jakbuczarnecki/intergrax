# © Artur Czarnecki. All rights reserved.

"""EBH-2A mechanical public contract dependency boundary gate."""

from testing_support.architecture.public_contract_boundary.evaluation import (
    evaluate_public_contract_dependency_boundary,
    format_gate_failure,
)
from testing_support.architecture.public_contract_boundary.models import (
    ContractDependencyDebtEntry,
    ContractDependencyViolation,
    DependencyRuleId,
    PublicContractBoundaryGateResult,
    RemovalStage,
)

from testing_support.architecture.public_contract_boundary.supplemental_surfaces import (
    SUPPLEMENTAL_PUBLIC_CONTRACT_SURFACES,
    SupplementalPublicContractSurface,
)

__all__ = (
    "ContractDependencyDebtEntry",
    "ContractDependencyViolation",
    "DependencyRuleId",
    "PublicContractBoundaryGateResult",
    "RemovalStage",
    "SUPPLEMENTAL_PUBLIC_CONTRACT_SURFACES",
    "SupplementalPublicContractSurface",
    "evaluate_public_contract_dependency_boundary",
    "format_gate_failure",
)
