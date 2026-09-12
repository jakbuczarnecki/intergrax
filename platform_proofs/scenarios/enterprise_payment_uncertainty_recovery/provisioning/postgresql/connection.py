"""PostgreSQL lab connection helpers — adapter boundary only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.database.contract import (
    REQUIRED_TABLES,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.infrastructure.contract import (
    POSTGRES_ENV_FILE,
    parse_env_file,
    validate_postgres_env_contract,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.infrastructure.health_verification import (
    verify_postgres_infrastructure,
)

if TYPE_CHECKING:
    import psycopg


@dataclass(frozen=True, slots=True)
class PostgresConnectionSettings:
    host: str
    port: int
    user: str
    password: str
    dbname: str


def load_connection_settings(
    env_file: Path = POSTGRES_ENV_FILE,
) -> PostgresConnectionSettings | None:
    if not env_file.is_file():
        return None
    values = parse_env_file(env_file)
    if validate_postgres_env_contract(values):
        return None
    port_raw = values.get("ERL_QUAL_004_POSTGRES_HOST_PORT", "")
    if not port_raw.isdigit():
        return None
    return PostgresConnectionSettings(
        host="127.0.0.1",
        port=int(port_raw),
        user=values["POSTGRES_USER"],
        password=values["POSTGRES_PASSWORD"],
        dbname=values["POSTGRES_DB"],
    )


def postgres_lab_available(env_file: Path = POSTGRES_ENV_FILE) -> bool:
    return verify_postgres_infrastructure(env_file=env_file).ok


def connect(settings: PostgresConnectionSettings) -> psycopg.Connection:
    import psycopg

    return psycopg.connect(
        host=settings.host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        dbname=settings.dbname,
        autocommit=False,
    )


def required_tables_present(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema || '.' || table_name
            FROM information_schema.tables
            WHERE table_schema IN ('commerce', 'external_sor', 'reconciliation')
            """
        )
        found = {row[0] for row in cur.fetchall()}
    return REQUIRED_TABLES <= found
