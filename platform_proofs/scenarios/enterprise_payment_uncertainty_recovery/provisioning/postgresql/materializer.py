"""Insert and delete scenario rows within PostgreSQL transactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.dataset_loader import (
    LoadedScenarioPackage,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.materialization import (
    MaterializedScenarioState,
    build_materialized_state,
    row_payloads,
)

if TYPE_CHECKING:
    import psycopg

from psycopg import sql

_INSERT_ORDER: tuple[str, ...] = (
    "commerce.organizations",
    "commerce.orders",
    "commerce.payment_intents",
    "external_sor.external_payment_effects",
    "external_sor.external_reality",
    "commerce.application_knowledge",
    "reconciliation.reconciliation_cases",
)

_DELETE_ORDER: tuple[str, ...] = (
    "reconciliation.resolution_records",
    "reconciliation.evidence_references",
    "reconciliation.investigation_attempts",
    "reconciliation.reconciliation_cases",
    "commerce.application_knowledge",
    "external_sor.external_reality",
    "external_sor.external_payment_effects",
    "commerce.payment_intents",
    "commerce.orders",
    "commerce.organizations",
)

_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "commerce.organizations": (
        "organization_id",
        "organization_key",
        "legal_name",
        "account_reference",
        "created_at",
        "updated_at",
    ),
    "commerce.orders": (
        "order_id",
        "organization_id",
        "order_number",
        "amount",
        "currency",
        "business_status",
        "created_at",
        "updated_at",
        "fulfillment_eligible_at",
    ),
    "commerce.payment_intents": (
        "payment_intent_id",
        "order_id",
        "intent_reference",
        "amount",
        "currency",
        "correlation_id",
        "attempt_ordinal",
        "application_status",
        "requested_at",
        "created_at",
        "updated_at",
    ),
    "external_sor.external_payment_effects": (
        "external_payment_effect_id",
        "payment_intent_id",
        "external_effect_reference",
        "correlation_id",
        "requested_state",
        "observed_integration_state",
        "requested_at",
        "observed_at",
        "created_at",
    ),
    "external_sor.external_reality": (
        "external_reality_id",
        "external_payment_effect_id",
        "correlation_id",
        "sor_transaction_ref",
        "terminal_outcome",
        "funds_captured",
        "truth_availability_state",
        "processed_at",
        "created_at",
    ),
    "commerce.application_knowledge": (
        "application_knowledge_id",
        "payment_intent_id",
        "external_payment_effect_id",
        "known_status",
        "order_payment_substate",
        "confirmation_received",
        "uncertainty_explicit",
        "inventory_reservation_knowledge",
        "observed_at",
        "created_at",
        "updated_at",
    ),
    "reconciliation.reconciliation_cases": (
        "reconciliation_case_id",
        "case_reference",
        "order_id",
        "payment_intent_id",
        "correlation_id",
        "variant_context",
        "resolution_state",
        "opened_at",
        "resolved_at",
        "created_at",
        "updated_at",
    ),
}


def _qualified_table(table: str) -> sql.Composed:
    schema, table_name = table.split(".", 1)
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table_name))


def _insert_statement(table: str) -> sql.Composed:
    columns = _TABLE_COLUMNS[table]
    column_idents = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
    return sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        _qualified_table(table),
        column_idents,
        placeholders,
    )


def materialize_package(
    conn: psycopg.Connection,
    package: LoadedScenarioPackage,
) -> MaterializedScenarioState:
    payloads = row_payloads(package)
    scenario_state = build_materialized_state(package)
    with conn.transaction():
        for table in _INSERT_ORDER:
            columns = _TABLE_COLUMNS[table]
            values = tuple(payloads[table][column] for column in columns)
            with conn.cursor() as cur:
                cur.execute(_insert_statement(table), values)
    return scenario_state


def cleanup_state(conn: psycopg.Connection, state: MaterializedScenarioState) -> None:
    predicates: dict[str, tuple[str, Any]] = {
        "reconciliation.resolution_records": (
            "reconciliation_case_id",
            state.reconciliation_case_id,
        ),
        "reconciliation.evidence_references": (
            "reconciliation_case_id",
            state.reconciliation_case_id,
        ),
        "reconciliation.investigation_attempts": (
            "reconciliation_case_id",
            state.reconciliation_case_id,
        ),
        "reconciliation.reconciliation_cases": (
            "reconciliation_case_id",
            state.reconciliation_case_id,
        ),
        "commerce.application_knowledge": (
            "application_knowledge_id",
            state.application_knowledge_id,
        ),
        "external_sor.external_reality": (
            "external_reality_id",
            state.external_reality_id,
        ),
        "external_sor.external_payment_effects": (
            "external_payment_effect_id",
            state.external_payment_effect_id,
        ),
        "commerce.payment_intents": (
            "payment_intent_id",
            state.payment_intent_id,
        ),
        "commerce.orders": ("order_id", state.order_id),
        "commerce.organizations": ("organization_id", state.organization_id),
    }
    with conn.transaction():
        for table in _DELETE_ORDER:
            column, value = predicates[table]
            delete_sql = sql.SQL("DELETE FROM {} WHERE {} = %s").format(
                _qualified_table(table),
                sql.Identifier(column),
            )
            with conn.cursor() as cur:
                cur.execute(delete_sql, (value,))


def state_records_present(conn: psycopg.Connection, state: MaterializedScenarioState) -> bool:
    checks: tuple[tuple[str, str, Any], ...] = (
        ("commerce.organizations", "organization_id", state.organization_id),
        ("commerce.orders", "order_id", state.order_id),
        ("commerce.payment_intents", "payment_intent_id", state.payment_intent_id),
        (
            "commerce.application_knowledge",
            "application_knowledge_id",
            state.application_knowledge_id,
        ),
        (
            "external_sor.external_payment_effects",
            "external_payment_effect_id",
            state.external_payment_effect_id,
        ),
        ("external_sor.external_reality", "external_reality_id", state.external_reality_id),
        (
            "reconciliation.reconciliation_cases",
            "reconciliation_case_id",
            state.reconciliation_case_id,
        ),
    )
    with conn.cursor() as cur:
        for table, column, value in checks:
            select_sql = sql.SQL("SELECT 1 FROM {} WHERE {} = %s").format(
                _qualified_table(table),
                sql.Identifier(column),
            )
            cur.execute(select_sql, (value,))
            if cur.fetchone() is None:
                return False
    return True
