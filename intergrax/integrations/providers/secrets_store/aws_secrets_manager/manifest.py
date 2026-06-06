# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``aws_secrets_manager`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="aws_secrets_manager",
    categories=(IntegrationCategory.SECRETS_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_AWS_SECRETS_MANAGER',
    description='aws_secrets_manager integration (Phase M.6 P4)',
)
