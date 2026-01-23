# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Protocol, Optional

from fastapi import Request

from intergrax.fastapi_core.auth.context import AuthContext

class AuthProvider(Protocol):
    """
    Application-level authentication provider.

    Responsible for authenticating the request
    and producing an AuthContext.
    """

    def authenticate(self, request: Request) -> Optional[AuthContext]:
        """
        Return AuthContext if authentication succeeds,
        or None if the request is unauthenticated.
        """
        ...
