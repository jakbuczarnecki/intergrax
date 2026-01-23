# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from fastapi import Request

from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.auth.extractors.jwt_bearer import JwtBearerExtractor
from intergrax.fastapi_core.auth.provider import AuthProvider
from intergrax.fastapi_core.auth.providers.jwt.verifier import JwtVerifier, NoOpJwtVerifier


class JwtAuthProvider(AuthProvider):
    """
    JWT-based authentication provider.
    """

    def __init__(
        self,
        verifier: JwtVerifier | None = None,
    ) -> None:
        self._extractor = JwtBearerExtractor()
        self._verifier = verifier or NoOpJwtVerifier()

    def authenticate(self, request: Request) -> Optional[AuthContext]:
        token = self._extractor.extract(request)
        if token is None:
            return None

        claims = self._verifier.verify(token)
        if claims is None:
            return None

        return AuthContext(
            is_authenticated=True,
            tenant_id=claims.tenant_id,
            user_id=claims.subject,
            scopes=claims.scopes,
        )
