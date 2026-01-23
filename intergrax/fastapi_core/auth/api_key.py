# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class ApiKeyIdentity:
    """
    Identity resolved from an API key.
    """
    tenant_id: str
    user_id: Optional[str]
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class ApiKeyConfig:
    """
    Static API key configuration.

    This is an MVP in-memory config.
    Will be replaced by DB / secrets manager later.
    """
    keys: Mapping[str, ApiKeyIdentity]