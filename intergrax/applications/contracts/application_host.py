# © Artur Czarnecki. All rights reserved.

"""Tier-3 host profile and feature flags (shared by manifest and environment)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ApplicationProfile(str, Enum):
    """Scaffold / factory profile for Tier-3 hosts."""

    LAB = "lab"
    PRODUCT = "product"


class ApplicationFeatures(BaseModel):
    """Product/lab behaviors toggled by the Tier-3 host factory."""

    model_config = ConfigDict(extra="forbid")

    debug_surface: bool = Field(
        default=True,
        description="Expose /debug/* inspection routes (lab profile default)",
    )
    interaction_routes: bool = Field(
        default=True,
        description="Mount inbound interaction intake router",
    )
    long_running_scheduler: bool = Field(
        default=True,
        description="Start long-running scheduler on app startup",
    )
    openapi: bool | None = Field(
        default=None,
        description="Override OpenAPI exposure; None = profile default",
    )
    task_sandbox_default: bool = Field(
        default=False,
        description="Default Task metadata sandbox flag (Tier-1 isolation, not this host)",
    )
    durable_async_index_default: bool = Field(
        default=False,
        description="Use SQLite async task index by default (AUDIT-IDEAL-28.1)",
    )
    streaming_intake_enabled: bool = Field(
        default=False,
        description="Expose streaming/async intake parity on product hosts (AUDIT-IDEAL-3.2)",
    )
    checkpoint_introspection_enabled: bool = Field(
        default=False,
        description="Expose checkpoint introspection HTTP API for ops (AUDIT-IDEAL-8.2)",
    )
    modality_worker_pool_enabled: bool = Field(
        default=False,
        description="Route Plane C vision inference through worker pools (AUDIT-IDEAL-29.2)",
    )
    agent_simulator_enabled: bool = Field(
        default=False,
        description="Expose MVP evolution simulate/replay HTTP routes (AUDIT-IDEAL-27.3)",
    )
    trace_explorer_enabled: bool = Field(
        default=False,
        description="Expose read-only trace explorer HTTP routes on product hosts (AUDIT-IDEAL-27.1)",
    )
    graph_editor_enabled: bool = Field(
        default=False,
        description="Expose graph editor / visualization HTTP routes (AUDIT-IDEAL-27.4)",
    )

    @classmethod
    def lab_defaults(cls) -> ApplicationFeatures:
        return cls(
            debug_surface=True,
            interaction_routes=True,
            long_running_scheduler=True,
            openapi=None,
            task_sandbox_default=False,
        )

    @classmethod
    def product_defaults(cls) -> ApplicationFeatures:
        return cls(
            debug_surface=False,
            interaction_routes=False,
            long_running_scheduler=True,
            openapi=False,
            task_sandbox_default=False,
            durable_async_index_default=True,
            streaming_intake_enabled=True,
            checkpoint_introspection_enabled=True,
            modality_worker_pool_enabled=True,
            agent_simulator_enabled=True,
            trace_explorer_enabled=True,
            graph_editor_enabled=True,
        )
