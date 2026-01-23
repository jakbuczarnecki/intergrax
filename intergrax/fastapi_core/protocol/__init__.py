# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations


class ApiHeaders:
    """
    Canonical HTTP header names used by Intergrax API.

    This module defines the public wire-level contract.
    """
    API_KEY: str = "X-API-Key"
    AUTHORIZATION: str = "Authorization"
    REQUEST_ID: str = "X-Request-ID"
