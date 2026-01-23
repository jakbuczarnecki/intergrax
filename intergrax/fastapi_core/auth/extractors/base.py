# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from fastapi import Request


class AuthHeaderExtractor(ABC):
    """
    Extract authentication material from HTTP request.

    Implementations must return a raw credential string
    or None if not applicable.
    """

    @abstractmethod
    def extract(self, request: Request) -> Optional[str]:
        raise NotImplementedError
