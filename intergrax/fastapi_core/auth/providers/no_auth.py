# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from fastapi import Request

from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.auth.provider import AuthProvider


class NoAuthProvider(AuthProvider):
    """
    Default authentication provider.

    Treats all requests as unauthenticated.
    """

    def authenticate(self, request: Request) -> Optional[AuthContext]:
        return None
