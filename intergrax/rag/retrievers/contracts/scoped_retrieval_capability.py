# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ScopedRetrievalCapability(Protocol):
    """Explicit scoped-retrieval capability — not inferred from retrieve() kwargs."""

    supports_scoped_retrieval: bool
