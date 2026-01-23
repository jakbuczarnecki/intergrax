# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Optional
from intergrax.fastapi_core.auth.api_key import ApiKeyConfig, ApiKeyIdentity


class ApiKeyIdentityResolver:
    """
    Resolve ApiKeyIdentity from raw API key.
    """

    def __init__(self, config: ApiKeyConfig) -> None:
        self._keys = config.keys

    def resolve(self, api_key: str) -> Optional[ApiKeyIdentity]:
        return self._keys.get(api_key)
