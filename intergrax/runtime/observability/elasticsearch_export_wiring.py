# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elasticsearch observability export operator wiring (OBS-VENDOR-5)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from intergrax.runtime.observability.operator_wiring import (
    ElasticsearchExportOperatorConfig,
    ObservabilityExportOperatorConfig,
    ObservabilityExportOperatorConfigError,
)

if TYPE_CHECKING:
    from intergrax.integrations.providers.observability_backend.elasticsearch.integration import (
        ElasticsearchObservabilityIntegration,
        ElasticsearchObservabilityTransport,
    )


def _require_enabled_elasticsearch_config(
    config: ObservabilityExportOperatorConfig,
) -> ElasticsearchExportOperatorConfig:
    if not config.enabled:
        raise ObservabilityExportOperatorConfigError("observability export is disabled")
    if config.backend_id != "elasticsearch":
        raise ObservabilityExportOperatorConfigError(
            f"elasticsearch export configuration requires backend_id 'elasticsearch', "
            f"got {config.backend_id!r}"
        )
    if config.elasticsearch is None:
        raise ObservabilityExportOperatorConfigError(
            "elasticsearch export configuration is required"
        )
    elasticsearch = config.elasticsearch
    if not elasticsearch.base_url.strip():
        raise ObservabilityExportOperatorConfigError("elasticsearch base_url is required")
    if not elasticsearch.index.strip():
        raise ObservabilityExportOperatorConfigError("elasticsearch index is required")
    return elasticsearch


def build_elasticsearch_observability_integration(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: ElasticsearchObservabilityTransport | None = None,
    http_client: Any | None = None,
    http_client_factory: Callable[..., Any] | None = None,
) -> ElasticsearchObservabilityIntegration:
    """Construct an Elasticsearch observability vendor integration from operator config."""
    from intergrax.integrations.providers.observability_backend.elasticsearch.bundle import (
        create_elasticsearch_observability_integration,
        create_elasticsearch_observability_transport,
    )
    from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
        ElasticsearchRetryPolicy,
    )

    elasticsearch = _require_enabled_elasticsearch_config(config)
    config_overrides: dict[str, object] = {
        "base_url": elasticsearch.base_url,
        "index": elasticsearch.index,
    }
    if elasticsearch.timeout_seconds is not None:
        config_overrides["timeout_seconds"] = elasticsearch.timeout_seconds

    retry_policy = ElasticsearchRetryPolicy(
        enabled=elasticsearch.retry_enabled,
        max_attempts=elasticsearch.retry_max_attempts,
        initial_backoff_seconds=elasticsearch.retry_initial_backoff_seconds,
        max_backoff_seconds=elasticsearch.retry_max_backoff_seconds,
    )

    failed_delivery_sink = None
    failed_delivery_file_path = (elasticsearch.failed_delivery_file_path or "").strip()
    if failed_delivery_file_path:
        from intergrax.integrations.providers.observability_backend.elasticsearch.failed_delivery import (
            FileElasticsearchFailedDeliverySink,
        )

        failed_delivery_sink = FileElasticsearchFailedDeliverySink(failed_delivery_file_path)

    active_transport = transport or create_elasticsearch_observability_transport(
        http_client=http_client,
        http_client_factory=http_client_factory,
        retry_policy=retry_policy,
        failed_delivery_sink=failed_delivery_sink,
        **config_overrides,
    )
    return create_elasticsearch_observability_integration(
        transport=active_transport,
        enabled=config.enabled,
    )


def _build_default_elasticsearch_observability_integration(
    config: ObservabilityExportOperatorConfig,
) -> object:
    return build_elasticsearch_observability_integration(config)
