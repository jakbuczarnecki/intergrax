"""Apply lab schema migration when tables are absent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.database.contract import (
    INITIAL_MIGRATION,
    read_sql,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.connection import (
    required_tables_present,
)

if TYPE_CHECKING:
    import psycopg


def ensure_lab_schema(conn: psycopg.Connection) -> None:
    if required_tables_present(conn):
        return
    sql = read_sql(INITIAL_MIGRATION)
    previous_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        conn.execute(sql)  # type: ignore[attr-defined]
    finally:
        conn.autocommit = previous_autocommit
