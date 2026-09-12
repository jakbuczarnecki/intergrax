"""PostgreSQL adapter for external_sor tables — external reality ownership only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from psycopg import sql

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.contracts.persistence import (
    ExternalRealityPersistenceBundle,
    ExternalRealityPersistencePort,
)

if TYPE_CHECKING:
    import psycopg


class PostgreSqlExternalRealityStore(ExternalRealityPersistencePort):
    """Persists capture outcomes into external_sor schema via existing lab connection."""

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def persist_external_reality(self, bundle: ExternalRealityPersistenceBundle) -> None:
        with self._connection.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO external_sor.external_payment_effects (
                        external_payment_effect_id,
                        payment_intent_id,
                        external_effect_reference,
                        correlation_id,
                        requested_state,
                        observed_integration_state,
                        requested_at,
                        observed_at,
                        created_at
                    ) VALUES (
                        %(external_payment_effect_id)s,
                        %(payment_intent_id)s,
                        %(external_effect_reference)s,
                        %(correlation_id)s,
                        %(requested_state)s,
                        %(observed_integration_state)s,
                        %(requested_at)s,
                        %(observed_at)s,
                        %(created_at)s
                    )
                    ON CONFLICT (external_effect_reference) DO UPDATE SET
                        observed_integration_state = EXCLUDED.observed_integration_state,
                        observed_at = EXCLUDED.observed_at
                    """
                ),
                {
                    "external_payment_effect_id": bundle.external_payment_effect_id,
                    "payment_intent_id": bundle.payment_intent_id,
                    "external_effect_reference": bundle.external_effect_reference,
                    "correlation_id": bundle.correlation_id,
                    "requested_state": bundle.requested_state,
                    "observed_integration_state": bundle.observed_integration_state,
                    "requested_at": bundle.requested_at,
                    "observed_at": bundle.processed_at,
                    "created_at": bundle.processed_at,
                },
            )
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO external_sor.external_reality (
                        external_reality_id,
                        external_payment_effect_id,
                        correlation_id,
                        sor_transaction_ref,
                        terminal_outcome,
                        funds_captured,
                        truth_availability_state,
                        processed_at,
                        created_at
                    ) VALUES (
                        %(external_reality_id)s,
                        %(external_payment_effect_id)s,
                        %(correlation_id)s,
                        %(sor_transaction_ref)s,
                        %(terminal_outcome)s,
                        %(funds_captured)s,
                        %(truth_availability_state)s,
                        %(processed_at)s,
                        %(created_at)s
                    )
                    ON CONFLICT (external_payment_effect_id) DO UPDATE SET
                        terminal_outcome = EXCLUDED.terminal_outcome,
                        funds_captured = EXCLUDED.funds_captured,
                        truth_availability_state = EXCLUDED.truth_availability_state,
                        processed_at = EXCLUDED.processed_at
                    """
                ),
                {
                    "external_reality_id": bundle.external_reality_id,
                    "external_payment_effect_id": bundle.external_payment_effect_id,
                    "correlation_id": bundle.correlation_id,
                    "sor_transaction_ref": bundle.sor_transaction_ref,
                    "terminal_outcome": bundle.sor_truth.terminal_outcome,
                    "funds_captured": bundle.sor_truth.funds_captured,
                    "truth_availability_state": bundle.sor_truth.truth_availability_state,
                    "processed_at": bundle.processed_at,
                    "created_at": bundle.processed_at,
                },
            )
