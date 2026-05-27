# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 Legal product host: FastAPI shell composing intergrax + Legal agent."""

from legal_application.host.factory import create_legal_backend_app
from legal_application.host.settings import LegalBackendSettings

__all__ = ["LegalBackendSettings", "create_legal_backend_app"]
