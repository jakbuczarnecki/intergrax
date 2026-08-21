# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3B).

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.providers.relational_store.postgresql import create_postgresql_relational_store
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry

from platform_proofs.tools.iterative_sql_investigation.contracts import (
    PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
    SqlQueryInput,
    SqlQueryOutput,
)
from platform_proofs.tools.iterative_sql_investigation.sql_tool import ProofSqlQueryHandler, SqlValidationError

DSN_ENV = "INTERGRAX_PP_SQL_INVESTIGATION_DSN"
ADMIN_DSN_ENV = "INTERGRAX_PP_SQL_INVESTIGATION_ADMIN_DSN"

DEFAULT_ADMIN_DSN = (
    "postgresql://proof_admin:proof_admin_local@localhost:5435/iterative_sql_proof"
)
DEFAULT_RUNTIME_DSN = (
    "postgresql://proof_runtime:proof_runtime_local@localhost:5435/iterative_sql_proof"
)


@dataclass(frozen=True, slots=True)
class ProofSqlRuntime:
    store: object
    registry: ToolRegistry
    invoker: RuntimeToolInvoker

    def close(self) -> None:
        self.store.close()


def build_proof_sql_tool_contract() -> ToolContract:
    return ToolContract(
        tool_id=PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
        name=PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
        description=(
            "Execute a single bounded read-only SQL query against the proof PostgreSQL dataset."
        ),
        description_short="Bounded read-only SQL query.",
        input_schema=SqlQueryInput,
        output_schema=SqlQueryOutput,
        error_mapping={SqlValidationError: "validation_error", ValueError: "validation_error"},
        side_effects=False,
        category="platform_proof",
        risk_level=ToolRiskLevel.LOW,
        tags=("platform_proof", "sql", "read_only"),
    )


def build_proof_sql_runtime(*, dsn: str) -> ProofSqlRuntime:
    store = create_postgresql_relational_store(dsn=dsn, tenant_schema="proof")
    store.connect()
    registry = ToolRegistry()
    contract = build_proof_sql_tool_contract()
    registry.register(contract, ProofSqlQueryHandler(store))
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    return ProofSqlRuntime(store=store, registry=registry, invoker=invoker)
