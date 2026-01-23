# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from fastapi import Request

from intergrax.fastapi_core.auth.extractors.base import AuthHeaderExtractor
from intergrax.fastapi_core.protocol import ApiHeaders


class ApiKeyExtractor(AuthHeaderExtractor):
    def extract(self, request: Request) -> Optional[str]:
        return request.headers.get(ApiHeaders.API_KEY)
