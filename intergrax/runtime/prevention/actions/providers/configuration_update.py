# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Example preventive provider — configuration.update → external intent (PREVENTIVE R7)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.external_operations.intent import (
    ExternalOperationIntent,
    ExternalOperationType,
    mint_external_operation_intent_id,
)
from intergrax.contracts.external_operations.provider import ProviderPayloadBounds
from intergrax.contracts.preventive.actions.action_type import PreventiveActionType
from intergrax.contracts.preventive.actions.proposal import PreventiveActionProposal


class ConfigurationUpdatePreventiveActionProvider:
    @property
    def provider_id(self) -> str:
        return "preventive.configuration_update"

    @property
    def supported_action_types(self) -> frozenset[str]:
        short = f"{PreventiveActionType.CONFIGURATION_UPDATE.namespace}.{PreventiveActionType.CONFIGURATION_UPDATE.name}"
        return frozenset({PreventiveActionType.CONFIGURATION_UPDATE.qualified_id, short})

    @property
    def payload_bounds(self) -> ProviderPayloadBounds:
        return ProviderPayloadBounds(
            max_payload_bytes=2048,
            timeout_seconds=10.0,
            max_retries=0,
        )

    def translate(
        self,
        proposal: PreventiveActionProposal,
        *,
        task_id: str,
        requested_by: str,
    ) -> ExternalOperationIntent:
        return ExternalOperationIntent(
            intent_id=mint_external_operation_intent_id(),
            tenant_id=proposal.tenant_id,
            task_id=task_id,
            operation_type=ExternalOperationType.CONNECTOR_UPDATE,
            target_resource=proposal.target_resource,
            requested_by=requested_by,
            justification=proposal.justification,
            created_at=datetime.now(timezone.utc),
        )


__all__ = ["ConfigurationUpdatePreventiveActionProvider"]
