# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 Legal product host: FastAPI shell composing ``intergrax`` (Tier-0/1) + Legal domain (Tier-2)."""

from legal_agent.host.factory import create_legal_backend_app
from legal_agent.host.settings import LegalBackendSettings

__all__ = ["LegalBackendSettings", "create_legal_backend_app"]
