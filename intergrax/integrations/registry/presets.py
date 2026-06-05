# © Artur Czarnecki. All rights reserved.

"""Named integration stacks for Tier-3 authoring (Phase DX-4.1)."""

from __future__ import annotations

from intergrax.integrations.registry.profile import IntegrationProfile


def lab_stack(*, enable_otel: bool = True) -> IntegrationProfile:
    """Reference lab harness stack (sqlite, log, lab_json, optional OTEL)."""
    return IntegrationProfile.lab_harness_preset(enable_otel=enable_otel)


def legal_stack() -> IntegrationProfile:
    """Legal product relational + vector + rerank preset."""
    return IntegrationProfile.legal_product()


def research_stack() -> IntegrationProfile:
    """Research product search + vector preset."""
    return IntegrationProfile.research_product()


def data_stack(*, enable_redis: bool = True, enable_qdrant: bool = False) -> IntegrationProfile:
    """Data-heavy harness: sqlite + optional redis/qdrant."""
    return IntegrationProfile.lab_harness_preset(
        enable_otel=False,
        enable_redis=enable_redis,
        enable_qdrant=enable_qdrant,
    )


def observability_stack(*, enable_otel: bool = True) -> IntegrationProfile:
    """Observability-first lab stack."""
    return IntegrationProfile.lab_harness_preset(enable_otel=enable_otel)
