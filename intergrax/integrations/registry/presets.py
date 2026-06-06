# © Artur Czarnecki. All rights reserved.

"""Named integration stacks for Tier-3 authoring (Phase DX-4.1, M.6 P4 follow-up)."""

from __future__ import annotations

from intergrax.integrations.registry.catalog_manifests import (
    DOCLING,
    DOPPLER,
    GITHUB_ACTIONS,
    GRAFANA,
    LAB_JSON,
    LOG,
    LOKI,
    OTEL,
    PGVECTOR,
    POSTGRESQL,
    SQLITE,
    TEMPO,
    UNLEASH,
)
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


def observability_stack(
    *,
    enable_otel: bool = True,
    enable_grafana_stack: bool = False,
) -> IntegrationProfile:
    """
    Observability-first lab stack.

    When ``enable_grafana_stack`` is True, binds ``grafana`` as primary observability backend
    and registers ``loki`` / ``tempo`` option slots for direct LogQL/TraceQL probes.
    """
    options: dict[str, dict[str, object]] = {}
    if enable_otel:
        options[OTEL.slug] = {}
    observability_backend = OTEL if enable_otel else None
    if enable_grafana_stack:
        observability_backend = GRAFANA
        options[LOKI.slug] = {}
        options[TEMPO.slug] = {}
        if enable_otel:
            options[OTEL.slug] = {}

    return IntegrationProfile(
        relational_store=SQLITE,
        notification_channel=LOG,
        interaction_surface=LAB_JSON,
        document_parser=DOCLING,
        observability_backend=observability_backend,
        options=options,
    )


def harness_production_stack(
    *,
    secrets_slug: str = "doppler",
    enable_grafana_stack: bool = True,
) -> IntegrationProfile:
    """
    Production-oriented harness integration stack (no business agents).

    Uses PostgreSQL + pgvector, catalog secrets backend, Grafana observability triad,
    Unleash feature flags, and GitHub Actions CI evidence reads.
    """
    allowed_secrets = {"doppler", "aws_secrets_manager", "vault"}
    normalized_secrets = secrets_slug.strip().lower()
    if normalized_secrets not in allowed_secrets:
        raise ValueError(f"Unsupported secrets slug for harness production stack: {secrets_slug!r}")

    integration = observability_stack(enable_otel=True, enable_grafana_stack=enable_grafana_stack)
    return integration.model_copy(
        update={
            "relational_store": POSTGRESQL,
            "vector_store": PGVECTOR,
            "secrets_store": normalized_secrets,
            "feature_flag": UNLEASH,
            "ci_cd": GITHUB_ACTIONS,
        }
    )
