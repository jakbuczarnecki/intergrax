# © Artur Czarnecki. All rights reserved.

"""Named integration stacks for Tier-3 authoring (Phase DX-4.1, M.6 P4 follow-up)."""

from __future__ import annotations

from intergrax.integrations.registry.catalog_manifests import (
    DOCLING,
    DOPPLER,
    DUCKDB,
    GITHUB,
    GITHUB_ACTIONS,
    GITLAB_CI,
    GRAFANA,
    KAFKA,
    LAB_JSON,
    LANGFUSE,
    LOG,
    LOKI,
    MINIO,
    OTEL,
    PGVECTOR,
    POSTGRESQL,
    PROMETHEUS,
    REDIS,
    REDPANDA,
    SQLITE,
    TEMPO,
    UNLEASH,
    VAULT,
)
from intergrax.integrations.core.manifest import IntegrationManifest
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


def harness_metrics_stack(
    *,
    enable_otel: bool = True,
    enable_grafana_stack: bool = True,
) -> IntegrationProfile:
    """Metrics-first harness stack (Prometheus + Grafana triad + OTEL)."""
    base = observability_stack(enable_otel=enable_otel, enable_grafana_stack=enable_grafana_stack)
    options = dict(base.options)
    options[PROMETHEUS.slug] = {}
    return base.model_copy(update={"observability_backend": PROMETHEUS, "options": options})


def harness_eval_stack(
    *,
    enable_minio: bool = True,
) -> IntegrationProfile:
    """Eval export stack (Langfuse traces + DuckDB analytics + optional MinIO artifacts)."""
    options: dict[str, dict[str, object]] = {DUCKDB.slug: {}}
    object_storage = MINIO if enable_minio else None
    if enable_minio:
        options[MINIO.slug] = {}
    return IntegrationProfile(
        relational_store=DUCKDB,
        observability_backend=LANGFUSE,
        object_storage=object_storage,
        notification_channel=LOG,
        interaction_surface=LAB_JSON,
        document_parser=DOCLING,
        options=options,
    )


def harness_async_stack(
    *,
    message_bus_slug: str = "redpanda",
    enable_redis: bool = True,
    enable_temporal: bool = False,
) -> IntegrationProfile:
    """Async harness stack (Redpanda/Kafka bus + Redis + optional Temporal)."""
    normalized = message_bus_slug.strip().lower()
    if normalized not in {"redpanda", "kafka"}:
        raise ValueError(f"Unsupported message bus for harness async stack: {message_bus_slug!r}")
    bus_manifest = REDPANDA if normalized == "redpanda" else KAFKA
    options: dict[str, dict[str, object]] = {}
    key_value_cache = REDIS if enable_redis else None
    if enable_redis:
        options[REDIS.slug] = {}
    if enable_temporal:
        options["temporal"] = {}
    return IntegrationProfile(
        relational_store=SQLITE,
        message_bus=bus_manifest,
        key_value_cache=key_value_cache,
        notification_channel=LOG,
        interaction_surface=LAB_JSON,
        document_parser=DOCLING,
        options=options,
    )


def harness_ci_stack(
    *,
    primary_ci: str = "github_actions",
    enable_gitlab_ci: bool = True,
    enable_circleci: bool = False,
) -> IntegrationProfile:
    """Multi-CI release evidence stack."""
    allowed = {"github_actions", "gitlab_ci", "circleci", "azure_pipelines", "codecov"}
    normalized = primary_ci.strip().lower()
    if normalized not in allowed:
        raise ValueError(f"Unsupported primary CI slug: {primary_ci!r}")
    manifest_by_slug = {
        "github_actions": GITHUB_ACTIONS,
        "gitlab_ci": GITLAB_CI,
    }
    ci_cd: IntegrationManifest | str = manifest_by_slug.get(normalized, normalized)
    options: dict[str, dict[str, object]] = {GITHUB.slug: {}}
    if enable_gitlab_ci:
        options[GITLAB_CI.slug] = {}
    if enable_circleci:
        options["circleci"] = {}
    return IntegrationProfile(
        relational_store=SQLITE,
        ci_cd=ci_cd,
        issue_tracker=GITHUB,
        notification_channel=LOG,
        interaction_surface=LAB_JSON,
        document_parser=DOCLING,
        options=options,
    )
