# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``semgrep`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="semgrep",
    categories=(IntegrationCategory.SECURITY_SCANNER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_SEMGREP',
    description='semgrep integration (Phase M.6 P6)',
)
