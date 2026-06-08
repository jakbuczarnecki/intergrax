# © Artur Czarnecki. All rights reserved.

"""Postgres session/LTM backend spike — multi-tenant RFC (Phase MEM-DEPTH-5.6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class PostgresMemoryBackendConfig:
    """
    Configuration contract for a future Postgres-backed memory bundle.

    Ship blocked on §6.3 multi-tenant product decision — spike only.
    """

    dsn_env: str = "INTERGRAX_POSTGRES_MEMORY_DSN"
    schema: str = "intergrax_memory"
    pool_size: int = 5
    enable_session_storage: bool = True
    enable_user_ltm: bool = True


@dataclass(frozen=True, slots=True)
class PostgresMemoryBackendSpikeResult:
    configured: bool
    message: str
    config: Optional[PostgresMemoryBackendConfig] = None


def evaluate_postgres_memory_backend_spike(
    *,
    dsn_present: bool,
) -> PostgresMemoryBackendSpikeResult:
    config = PostgresMemoryBackendConfig()
    if not dsn_present:
        return PostgresMemoryBackendSpikeResult(
            configured=False,
            message=f"Postgres memory backend not configured — set {config.dsn_env}",
            config=config,
        )
    return PostgresMemoryBackendSpikeResult(
        configured=True,
        message="Postgres memory backend spike ready for integration bundle wiring",
        config=config,
    )
