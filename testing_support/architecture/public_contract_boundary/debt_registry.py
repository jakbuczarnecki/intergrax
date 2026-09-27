# © Artur Czarnecki. All rights reserved.

"""Declarative EBH-2 existing public-contract dependency debt (EBH-2A gate exceptions)."""

from __future__ import annotations

from testing_support.architecture.public_contract_boundary.models import (
    ContractDependencyDebtEntry,
    DependencyRuleId,
    RemovalStage,
)

PUBLIC_CONTRACT_DEPENDENCY_DEBT: tuple[ContractDependencyDebtEntry, ...] = (
    ContractDependencyDebtEntry(finding_id="EBH2A-D-069", source_module="intergrax.contracts.acp_budget_enforcement", forbidden_import_module="intergrax.utils", rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION, removal_stage=RemovalStage.EBH_2B,),
    ContractDependencyDebtEntry(finding_id="EBH2A-D-076", source_module="intergrax.contracts.idempotency_store", forbidden_import_module="intergrax.tools.execution_models", rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION, removal_stage=RemovalStage.EBH_2B,),
    ContractDependencyDebtEntry(finding_id="EBH2A-D-077", source_module="intergrax.contracts.reasoning_profile", forbidden_import_module="intergrax.llm_adapters.registry.profile", rule_id=DependencyRuleId.FORBIDDEN_REGISTRY_NAMESPACE, removal_stage=RemovalStage.EBH_2B,),
    ContractDependencyDebtEntry(finding_id="EBH2A-D-079", source_module="intergrax.contracts.runtime_environment", forbidden_import_module="intergrax.llm_adapters.registry.profile", rule_id=DependencyRuleId.FORBIDDEN_REGISTRY_NAMESPACE, removal_stage=RemovalStage.EBH_2B,),
    ContractDependencyDebtEntry(finding_id="EBH2A-D-080", source_module="intergrax.contracts.runtime_environment", forbidden_import_module="intergrax.llm_adapters.routing", rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION, removal_stage=RemovalStage.EBH_2B,),
    ContractDependencyDebtEntry(finding_id="EBH2A-D-095", source_module="intergrax.knowledge.contracts.validation", forbidden_import_module="intergrax.core.security", rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION, removal_stage=RemovalStage.EBH_3,),
    ContractDependencyDebtEntry(finding_id="EBH2A-D-125", source_module="intergrax.queueing.contracts.task_queue", forbidden_import_module="intergrax.queueing.task_priority", rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION, removal_stage=RemovalStage.EBH_3,),
    ContractDependencyDebtEntry(finding_id="EBH2A-D-140", source_module="intergrax.contracts.execution.suspended_operation.reentry", forbidden_import_module="intergrax.tools.execution_models", rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION, removal_stage=RemovalStage.EBH_3,),
    ContractDependencyDebtEntry(finding_id="EBH2A-D-141", source_module="intergrax.contracts.execution_bound_catalog_tool_invocation", forbidden_import_module="intergrax.tools.execution_models", rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION, removal_stage=RemovalStage.EBH_3,),
    ContractDependencyDebtEntry(finding_id="EBH2A-D-142", source_module="intergrax.contracts.execution_bound_catalog_tool_invocation", forbidden_import_module="intergrax.tools.invocation_wiring", rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION, removal_stage=RemovalStage.EBH_3,),
)
