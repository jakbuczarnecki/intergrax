# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Phase M.6 P4 integration slugs (28 providers)."""

from __future__ import annotations


def register_m6_p4_integrations(*, override: bool = False) -> None:
    from intergrax.integrations.providers.vector_store.pgvector.register import register_pgvector_integration
    from intergrax.integrations.providers.relational_store.duckdb.register import register_duckdb_integration
    from intergrax.integrations.providers.observability_backend.influxdb.register import register_influxdb_integration
    from intergrax.integrations.providers.relational_store.timescaledb.register import register_timescaledb_integration
    from intergrax.integrations.providers.observability_backend.grafana.register import register_grafana_integration
    from intergrax.integrations.providers.observability_backend.loki.register import register_loki_integration
    from intergrax.integrations.providers.observability_backend.tempo.register import register_tempo_integration
    from intergrax.integrations.providers.secrets_store.aws_secrets_manager.register import register_aws_secrets_manager_integration
    from intergrax.integrations.providers.secrets_store.azure_key_vault.register import register_azure_key_vault_integration
    from intergrax.integrations.providers.secrets_store.gcp_secret_manager.register import register_gcp_secret_manager_integration
    from intergrax.integrations.providers.secrets_store.doppler.register import register_doppler_integration
    from intergrax.integrations.providers.feature_flag.unleash.register import register_unleash_integration
    from intergrax.integrations.providers.feature_flag.launchdarkly.register import register_launchdarkly_integration
    from intergrax.integrations.providers.ci_cd.github_actions.register import register_github_actions_integration
    from intergrax.integrations.providers.message_bus.redpanda.register import register_redpanda_integration
    from intergrax.integrations.providers.object_storage.cloudflare_r2.register import register_cloudflare_r2_integration
    from intergrax.integrations.providers.graph_store.memgraph.register import register_memgraph_integration
    from intergrax.integrations.providers.graph_store.falkordb.register import register_falkordb_integration
    from intergrax.integrations.providers.notification_channel.incident_io.register import register_incident_io_integration
    from intergrax.integrations.providers.cloud_platform.kubernetes.register import register_kubernetes_integration
    from intergrax.integrations.providers.issue_tracker.servicenow.register import register_servicenow_integration
    from intergrax.integrations.providers.issue_tracker.bitbucket.register import register_bitbucket_integration
    from intergrax.integrations.providers.issue_tracker.asana.register import register_asana_integration
    from intergrax.integrations.providers.notification_channel.sendgrid.register import register_sendgrid_integration
    from intergrax.integrations.providers.notification_channel.mailgun.register import register_mailgun_integration
    from intergrax.integrations.providers.observability_backend.mlflow.register import register_mlflow_integration
    from intergrax.integrations.providers.object_storage.huggingface_hub.register import register_huggingface_hub_integration
    from intergrax.integrations.providers.model_serving_runtime.ollama.register import register_ollama_integration

    register_pgvector_integration(override=override)
    register_duckdb_integration(override=override)
    register_influxdb_integration(override=override)
    register_timescaledb_integration(override=override)
    register_grafana_integration(override=override)
    register_loki_integration(override=override)
    register_tempo_integration(override=override)
    register_aws_secrets_manager_integration(override=override)
    register_azure_key_vault_integration(override=override)
    register_gcp_secret_manager_integration(override=override)
    register_doppler_integration(override=override)
    register_unleash_integration(override=override)
    register_launchdarkly_integration(override=override)
    register_github_actions_integration(override=override)
    register_redpanda_integration(override=override)
    register_cloudflare_r2_integration(override=override)
    register_memgraph_integration(override=override)
    register_falkordb_integration(override=override)
    register_incident_io_integration(override=override)
    register_kubernetes_integration(override=override)
    register_servicenow_integration(override=override)
    register_bitbucket_integration(override=override)
    register_asana_integration(override=override)
    register_sendgrid_integration(override=override)
    register_mailgun_integration(override=override)
    register_mlflow_integration(override=override)
    register_huggingface_hub_integration(override=override)
    register_ollama_integration(override=override)
