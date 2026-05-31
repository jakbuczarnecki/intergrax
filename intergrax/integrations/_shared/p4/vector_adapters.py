# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vespa vector facade for catalog registration."""

from __future__ import annotations

from typing import Any, Optional

from intergrax.integrations._shared.p3.vector_adapters import _VectorStoreFacade


class VespaVectorFacade(_VectorStoreFacade):
    def __init__(self, client: Any, *, collection: str, tenant_id: str) -> None:
        super().__init__(collection=collection, tenant_id=tenant_id)
        self._client = client
