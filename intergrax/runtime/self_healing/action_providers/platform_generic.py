# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Platform self-healing action provider — external operation translation (SELF-HEALING R1)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.external_operations.intent import (
    ExternalOperationIntent,
    ExternalOperationType,
    mint_external_operation_intent_id,
)
from intergrax.contracts.external_operations.provider import ProviderPayloadBounds
from intergrax.contracts.self_healing.decision import SelfHealingDecision, SelfHealingProposedAction


class PlatformSelfHealingActionProvider:
    @property
    def provider_id(self) -> str:
        return "self_healing.platform.generic"

    @property
    def supported_action_types(self) -> frozenset[str]:
        return frozenset(
            {
                "self_healing.retry.adjustment",
                "self_healing.capacity.protection",
                "self_healing.dependency.isolation",
                "self_healing.human.escalation",
                "self_healing.rollback.restore",
            },
        )

    @property
    def payload_bounds(self) -> ProviderPayloadBounds:
        return ProviderPayloadBounds(
            max_payload_bytes=2048,
            timeout_seconds=10.0,
            max_retries=0,
        )

    def translate(
        self,
        decision: SelfHealingDecision,
        action: SelfHealingProposedAction,
        *,
        tenant_id: str,
        task_id: TaskId,
        requested_by: str,
    ) -> ExternalOperationIntent:
        return ExternalOperationIntent(
            intent_id=mint_external_operation_intent_id(),
            tenant_id=tenant_id,
            task_id=task_id,
            operation_type=ExternalOperationType.CONNECTOR_UPDATE,
            target_resource=action.target_resource,
            requested_by=requested_by,
            justification=decision.justification,
            created_at=datetime.now(timezone.utc),
        )


__all__ = ["PlatformSelfHealingActionProvider"]
