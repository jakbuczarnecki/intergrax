# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bootstrap platform workflow plugins (SELF-HEALING R2)."""

from __future__ import annotations

from intergrax.contracts.self_healing.workflow.registry import (
    SelfHealingPlanBuilderRegistry,
    SelfHealingRollbackRegistry,
    SelfHealingValidationRegistry,
    SelfHealingWorkflowPluginDescriptor,
)
from intergrax.runtime.self_healing.workflow.default_plan_builder import PlatformDefaultSelfHealingPlanBuilder
from intergrax.runtime.self_healing.workflow.platform_rollback import PlatformDefaultRollbackProvider
from intergrax.runtime.self_healing.workflow.platform_validation import PlatformEvidenceValidationProvider


def _descriptor(plugin_id: str, *, capabilities: tuple[str, ...]) -> SelfHealingWorkflowPluginDescriptor:
    return SelfHealingWorkflowPluginDescriptor(
        plugin_id=plugin_id,
        version="1",
        namespace="platform",
        priority=10,
        capabilities=capabilities,
        tenant_scope=None,
        timeout_seconds=5.0,
    )


def register_platform_workflow_plugins(
    *,
    plan_builders: SelfHealingPlanBuilderRegistry,
    validation_registry: SelfHealingValidationRegistry,
    rollback_registry: SelfHealingRollbackRegistry,
    strategy_ids: tuple[str, ...],
    register_validation: bool = True,
) -> None:
    builder = PlatformDefaultSelfHealingPlanBuilder()
    desc = _descriptor(builder.builder_id, capabilities=("plan.build",))
    for strategy_id in strategy_ids:
        plan_builders.register(builder, desc, strategy_id=strategy_id)
    if register_validation:
        validation = PlatformEvidenceValidationProvider()
        validation_registry.register(
            validation,
            _descriptor(validation.provider_id, capabilities=("validate",)),
        )
    rollback = PlatformDefaultRollbackProvider()
    rollback_registry.register(
        rollback,
        _descriptor(rollback.provider_id, capabilities=("rollback",)),
    )


__all__ = ["register_platform_workflow_plugins"]
