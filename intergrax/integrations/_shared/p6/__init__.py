# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.6 P5 integration factories (harness depth wave)."""

from intergrax.integrations._shared.p6.factories import (
    create_azure_pipelines_ci_cd,
    create_circleci_ci_cd,
    create_codecov_ci_cd,
    create_gitlab_ci_ci_cd,
    create_grafana_oncall_notification_channel,
    create_localstack_cloud_platform,
    create_mailpit_notification_channel,
    create_opentelemetry_collector_observability_backend,
)

__all__ = [
    "create_azure_pipelines_ci_cd",
    "create_circleci_ci_cd",
    "create_codecov_ci_cd",
    "create_gitlab_ci_ci_cd",
    "create_grafana_oncall_notification_channel",
    "create_localstack_cloud_platform",
    "create_mailpit_notification_channel",
    "create_opentelemetry_collector_observability_backend",
]
