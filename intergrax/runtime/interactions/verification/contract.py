# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Inbound HTTP request verification contracts (§18, Phase H.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping


class InboundRequestVerifier(ABC):
    """Validates vendor webhook signatures before payload parsing."""

    @abstractmethod
    def verify(self, *, headers: Mapping[str, str], body: bytes) -> None:
        """Raise ``ValueError`` when verification fails."""


class NullInboundRequestVerifier(InboundRequestVerifier):
    """No-op verifier — default for laboratory paths."""

    def verify(self, *, headers: Mapping[str, str], body: bytes) -> None:
        _ = headers, body
