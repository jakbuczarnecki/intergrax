# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``aws`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="aws",
    categories=(IntegrationCategory.CLOUD_PLATFORM,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_AWS',
    description='AWS cloud platform facade (IAM/STS auth, category slug defaults for S3/SQS/DynamoDB/ElastiCache)',
)
