"""PostgreSQL lab adapter — reads ``external_sor.external_reality`` by correlation id."""

from __future__ import annotations

from typing import TYPE_CHECKING

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealityLookupPort,
    ExternalRealitySnapshot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    ExternalRealityRecordMissing,
    ExternalRealitySourceUnavailable,
)

if TYPE_CHECKING:
    import psycopg


class PostgreSqlExternalRealityLookup(ExternalRealityLookupPort):
    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def lookup_by_correlation_id(self, correlation_id: str) -> ExternalRealitySnapshot:
        normalized = correlation_id.strip()
        if not normalized:
            raise ExternalRealityRecordMissing("empty_correlation_id")
        try:
            with self._connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        er.correlation_id,
                        epe.external_effect_reference,
                        er.terminal_outcome,
                        er.funds_captured,
                        er.truth_availability_state,
                        er.sor_transaction_ref
                    FROM external_sor.external_reality er
                    INNER JOIN external_sor.external_payment_effects epe
                        ON epe.external_payment_effect_id = er.external_payment_effect_id
                    WHERE er.correlation_id = %s
                    LIMIT 1
                    """,
                    (normalized,),
                )
                row = cur.fetchone()
        except Exception as exc:
            raise ExternalRealitySourceUnavailable(str(exc)) from exc

        if row is None:
            raise ExternalRealityRecordMissing(f"no_external_reality_for:{normalized}")

        return ExternalRealitySnapshot(
            correlation_id=str(row[0]),
            external_effect_reference=str(row[1]),
            terminal_outcome=str(row[2]),
            funds_captured=bool(row[3]),
            truth_availability_state=str(row[4]),
            sor_transaction_ref=str(row[5]) if row[5] is not None else None,
        )
