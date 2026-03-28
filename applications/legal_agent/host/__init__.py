# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deployable FastAPI host for the Legal product (Tier-2). Depends on ``intergrax`` (Tier-0/1) only."""

from legal_agent.host.factory import create_legal_backend_app
from legal_agent.host.settings import LegalBackendSettings

__all__ = ["LegalBackendSettings", "create_legal_backend_app"]
