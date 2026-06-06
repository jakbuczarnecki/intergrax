# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Phase M.6 P5 integration slugs (8 greenfield providers)."""

from __future__ import annotations


def register_m6_p5_integrations(*, override: bool = False) -> None:
    from intergrax.integrations.providers.ci_cd.gitlab_ci.register import register_gitlab_ci_integration
    from intergrax.integrations.providers.ci_cd.circleci.register import register_circleci_integration
    from intergrax.integrations.providers.ci_cd.azure_pipelines.register import register_azure_pipelines_integration
    from intergrax.integrations.providers.ci_cd.codecov.register import register_codecov_integration
    from intergrax.integrations.providers.notification_channel.mailpit.register import register_mailpit_integration
    from intergrax.integrations.providers.notification_channel.grafana_oncall.register import register_grafana_oncall_integration
    from intergrax.integrations.providers.cloud_platform.localstack.register import register_localstack_integration
    from intergrax.integrations.providers.observability_backend.opentelemetry_collector.register import (
        register_opentelemetry_collector_integration,
    )

    register_gitlab_ci_integration(override=override)
    register_circleci_integration(override=override)
    register_azure_pipelines_integration(override=override)
    register_codecov_integration(override=override)
    register_mailpit_integration(override=override)
    register_grafana_oncall_integration(override=override)
    register_localstack_integration(override=override)
    register_opentelemetry_collector_integration(override=override)
