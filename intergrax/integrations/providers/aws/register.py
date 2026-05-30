# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register AWS in the integration catalog (Phase M.6)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.aws.bundle import create_aws_cloud_platform
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_aws_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.AWS.value,
            categories=(IntegrationCategory.CLOUD_PLATFORM,),
            factory=create_aws_cloud_platform,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_AWS",
            description=(
                "AWS cloud platform facade (IAM/STS auth, category slug defaults for S3/SQS/DynamoDB/ElastiCache)"
            ),
        ),
        override=override,
    )
