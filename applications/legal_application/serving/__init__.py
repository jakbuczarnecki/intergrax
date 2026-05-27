# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
HTTP surface for Legal Agent on :mod:`intergrax.fastapi_core`.

Mount with :func:`mount_legal_agent_routes` after :func:`intergrax.fastapi_core.app_factory.create_app`.
"""

from legal_application.serving.fastapi_router import (
    DefaultLegalAgentService,
    LegalAgentService,
    LegalAgentServingConfig,
    LegalAgentServingFacade,
    LegalIdentitySource,
    create_legal_agent_router,
    legal_agent_router,
    mount_legal_agent_routes,
)
from legal_application.serving.runtime_bridge import LegalApiV1RuntimeMapper
from intergrax.runtime.nexus.policies.runtime_policies import DataCompliancePolicy

__all__ = [
    "DataCompliancePolicy",
    "DefaultLegalAgentService",
    "LegalAgentService",
    "LegalAgentServingConfig",
    "LegalAgentServingFacade",
    "LegalApiV1RuntimeMapper",
    "LegalIdentitySource",
    "create_legal_agent_router",
    "legal_agent_router",
    "mount_legal_agent_routes",
]
