# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Optional
from fastapi import Request

from intergrax.fastapi_core.auth.api_key import ApiKeyConfig
from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.auth.provider import AuthProvider
from intergrax.fastapi_core.auth.providers.api_key.extractor import ApiKeyHeaderExtractor
from intergrax.fastapi_core.auth.providers.api_key.resolver import ApiKeyIdentityResolver


class ApiKeyAuthProvider(AuthProvider):
    """
    API key based authentication provider.
    """

    def __init__(self, config: ApiKeyConfig) -> None:
        self._extractor = ApiKeyHeaderExtractor()
        self._resolver = ApiKeyIdentityResolver(config)

    def authenticate(self, request: Request) -> Optional[AuthContext]:
        api_key = self._extractor.extract(request)
        if api_key is None:
            return None

        identity = self._resolver.resolve(api_key)
        if identity is None:
            return None

        return AuthContext(
            is_authenticated=True,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
            scopes=identity.scopes,
        )
