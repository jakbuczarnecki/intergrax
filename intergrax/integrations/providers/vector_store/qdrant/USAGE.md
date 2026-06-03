# `qdrant` integration — usage

**Category:** ``vector_store``  
**Catalog factory:** ``create_qdrant_vector_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(vector_store="qdrant")
backend = profile.resolve(IntegrationCategory.VECTOR_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store

backend = create_qdrant_vector_store(**config_overrides)
```


## Environment variables

`INTERGRAX_QDRANT_URL` or `INTERGRAX_QDRANT_HOST`/`INTERGRAX_QDRANT_PORT`; optional `INTERGRAX_QDRANT_API_KEY`, `INTERGRAX_QDRANT_COLLECTION`, `INTERGRAX_QDRANT_TENANT_ID`, `INTERGRAX_QDRANT_METRIC`

## Example

```python
from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store

store = create_qdrant_vector_store(
    collection_name="intergrax-rag",
    tenant_id="tenant-a",
    host="localhost",
    port=6333,
)
store.add_documents(
    [Document(page_content="Intergrax overview", metadata={"source": "docs"})],
    [[0.01, 0.02, 0.03]],
    ids=["doc-1"],
)
hits = store.query([0.01, 0.02, 0.03], top_k=5)
store.delete(["doc-1"])
```

## Notes

Catalog bridge to ``intergrax/rag/`` — ``qdrant_client`` import only in ``opens.py``; RAG ``QdrantVectorStore`` unchanged.
