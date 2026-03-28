# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
HTTP surface for Legal Agent on :mod:`intergrax.fastapi_core`.

Mount with :func:`mount_legal_agent_routes` after :func:`intergrax.fastapi_core.app_factory.create_app`.
Uses the same DI style as ``RunService`` (``dependency_overrides`` + :class:`LegalAgentService`).
Requires ``RequestContextMiddleware`` (same as core ``/runs``).
"""

from intergrax.agents_packages.legal_agent.serving.fastapi_router import (
    DefaultLegalAgentService,
    LegalAgentService,
    LegalAgentServingConfig,
    LegalAgentServingFacade,
    LegalIdentitySource,
    create_legal_agent_router,
    legal_agent_router,
    mount_legal_agent_routes,
)
from intergrax.agents_packages.legal_agent.serving.runtime_bridge import LegalApiV1RuntimeMapper

__all__ = [
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
