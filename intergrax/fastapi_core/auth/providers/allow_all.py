# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from starlette.requests import Request

from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.auth.provider import AuthProvider

class AllowAllAuthProvider(AuthProvider):
    def authenticate(self, request: Request) -> AuthContext:
        return AuthContext(
            is_authenticated=True,
            tenant_id="__trusted__",
            user_id="__trusted__",
            scopes=("*",),
        )
