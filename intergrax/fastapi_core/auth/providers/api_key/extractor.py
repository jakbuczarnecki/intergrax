# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Optional
from fastapi import Request

from intergrax.fastapi_core.protocol import ApiHeaders


class ApiKeyHeaderExtractor:
    """
    Extract raw API key from X-API-Key header.
    """

    def extract(self, request: Request) -> Optional[str]:
        return request.headers.get(ApiHeaders.API_KEY)
