# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional, Sequence

from fastapi import Request

from intergrax.fastapi_core.auth.extractors.api_key import ApiKeyExtractor
from intergrax.fastapi_core.auth.extractors.base import AuthHeaderExtractor
from intergrax.fastapi_core.auth.extractors.jwt_bearer import JwtBearerExtractor

class AuthResolver:
    """
    Resolve raw authentication credential from request
    using a sequence of extractors.
    """

    def __init__(self, extractors: Sequence[AuthHeaderExtractor]) -> None:
        self._extractors = extractors

    def resolve(self, request: Request) -> Optional[str]:
        for extractor in self._extractors:
            value = extractor.extract(request)
            if value:
                return value
        return None


DEFAULT_AUTH_RESOLVER = AuthResolver(
    extractors=[
        ApiKeyExtractor(),
        JwtBearerExtractor(),
    ]
)
