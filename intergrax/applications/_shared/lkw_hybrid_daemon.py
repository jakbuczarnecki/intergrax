# © Artur Czarnecki. All rights reserved.

"""LKW hybrid daemon contract (CFG-14 / AUDIT-IDEAL-28.3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LkwHybridDaemonSpec(BaseModel):
    """Single-user hybrid daemon process model for Local Knowledge Workspace."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    process_name: str = "lkw-host"
    bind_host: str = "127.0.0.1"
    port: int = Field(default=8020, ge=1, le=65535)
    include_mcp: bool = True
    include_interactions: bool = True
    include_scheduler: bool = True
    data_home_env: str = "LKW_DATA_HOME"
    launcher_script: str = "scripts/maintenance/lkw-host.py"


def build_lkw_hybrid_daemon_spec(
    *,
    bind_host: str = "127.0.0.1",
    port: int = 8020,
    include_mcp: bool = True,
    include_interactions: bool = True,
    include_scheduler: bool = True,
) -> LkwHybridDaemonSpec:
    """Build the CFG-14 hybrid daemon specification."""
    return LkwHybridDaemonSpec(
        bind_host=bind_host,
        port=port,
        include_mcp=include_mcp,
        include_interactions=include_interactions,
        include_scheduler=include_scheduler,
    )
