# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from fastapi import Request

from intergrax.fastapi_core.auth.extractors.base import AuthHeaderExtractor
from intergrax.fastapi_core.protocol import ApiHeaders

class JwtBearerExtractor(AuthHeaderExtractor):
 
    PREFIX: str = "Bearer "

    def extract(self, request: Request) -> Optional[str]:        
        header = request.headers.get(ApiHeaders.AUTHORIZATION)
        if not header or not header.startswith(self.PREFIX):
            return None
        return header[len(self.PREFIX):]
