# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``langfuse`` integration.

Metadata only. ``IntegrationCategory.OBSERVABILITY_BACKEND`` maps to runtime
``ObservabilityVendorIntegrationContract`` (``integration_kind=observability_vendor``).
"""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="langfuse",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_LANGFUSE',
    description='langfuse integration (Phase M.7)',
)
