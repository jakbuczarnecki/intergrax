# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.vector_store.contracts import (
    VectorStoreCountInput,
    VectorStoreCountOutput,
    VectorStoreDeleteInput,
    VectorStoreDeleteOutput,
    VectorStoreHealthInput,
    VectorStoreHealthOutput,
    VectorStoreListCollectionsInput,
    VectorStoreListCollectionsOutput,
)
from intergrax.tools.providers.vector_store.service import (
    vector_store_count,
    vector_store_delete,
    vector_store_health,
    vector_store_list_collections,
)


class VectorStoreCountHandler(ServiceToolHandler[VectorStoreCountInput, VectorStoreCountOutput]):
    _service = vector_store_count


class VectorStoreDeleteHandler(ServiceToolHandler[VectorStoreDeleteInput, VectorStoreDeleteOutput]):
    _service = vector_store_delete


class VectorStoreListCollectionsHandler(
    ServiceToolHandler[VectorStoreListCollectionsInput, VectorStoreListCollectionsOutput]
):
    _service = vector_store_list_collections


class VectorStoreHealthHandler(ServiceToolHandler[VectorStoreHealthInput, VectorStoreHealthOutput]):
    _service = vector_store_health
