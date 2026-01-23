# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional, Sequence

from fastapi import Request

from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.auth.provider import AuthProvider

class CompositeAuthProvider(AuthProvider):
    """
    Auth provider that tries multiple providers in order.

    The first provider that returns AuthContext wins.
    """

    def __init__(self, providers: Sequence[AuthProvider]) -> None:
        if not providers:
            raise ValueError("CompositeAuthProvider requires at least one provider")
        self._providers = providers

    def authenticate(self, request: Request) -> Optional[AuthContext]:
        for provider in self._providers:
            auth = provider.authenticate(request)
            if auth is not None:
                return auth
        return None
