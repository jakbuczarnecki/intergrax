"""Lab dataset order access — canonical commerce row for in-memory proof runs."""

from __future__ import annotations

from decimal import Decimal

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.entities import (
    EnterpriseOrganization,
    OrderSnapshot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.failures import (
    ApplicationFailure,
    ApplicationFailureCode,
    MissingBusinessEntityError,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.application.references import (
    LabBusinessReferences,
)


class LabReferenceOrderAccess:
    """Returns the canonical Nordic Industrial order for the lab dataset."""

    def __init__(self, references: LabBusinessReferences | None = None) -> None:
        self._references = references or LabBusinessReferences()

    def load_order_for_execution(self, context: ScenarioExecutionContext) -> OrderSnapshot:
        expected = context.correlation_ids.get("order_logical_id")
        if expected != self._references.logical_order_id:
            raise MissingBusinessEntityError(
                ApplicationFailure(
                    code=ApplicationFailureCode.MISSING_BUSINESS_ENTITY,
                    message=f"order not found for logical id: {expected}",
                )
            )
        return OrderSnapshot(
            logical_order_id=self._references.logical_order_id,
            order_number=self._references.order_number,
            organization=EnterpriseOrganization(
                organization_key=self._references.organization_key,
                legal_name=self._references.organization_legal_name,
                account_reference=self._references.account_reference,
            ),
            amount=Decimal(self._references.order_amount),
            currency=self._references.currency,
            business_status=self._references.order_business_status,
        )
