# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class JwtClaims:
    """
    Normalized JWT claims for JwtAuthProvider.
    """
    subject: Optional[str]
    tenant_id: Optional[str]
    scopes: Tuple[str, ...]


class JwtVerifier:
    """
    JWT verifier contract (provider-internal).
    """

    def verify(self, token: str) -> Optional[JwtClaims]:
        raise NotImplementedError


class NoOpJwtVerifier(JwtVerifier):
    """
    Stub verifier accepting any token.
    """

    def verify(self, token: str) -> Optional[JwtClaims]:
        return JwtClaims(
            subject=None,
            tenant_id=None,
            scopes=(),
        )
