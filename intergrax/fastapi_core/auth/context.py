# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AuthContext:
    """
    Authentication and authorization context resolved for a request.

    This is a transport-level identity contract, independent of
    runtime / agent identity models.
    """
    is_authenticated: bool
    tenant_id: Optional[str]
    user_id: Optional[str]
    scopes: tuple[str, ...]
