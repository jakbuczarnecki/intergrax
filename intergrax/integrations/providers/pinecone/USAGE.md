# `pinecone` integration — usage

**Category:** ``vector_store``  
**Catalog factory:** ``create_pinecone_vector_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(vector_store=IntegrationSlug.PINECONE)
backend = profile.resolve(IntegrationCategory.VECTOR_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.pinecone.bundle import create_pinecone_vector_store

backend = create_pinecone_vector_store(**config_overrides)
```


## Environment variables

`INTERGRAX_PINECONE_API_KEY`, `INTERGRAX_PINECONE_INDEX`; optional `INTERGRAX_PINECONE_TENANT_ID`, `INTERGRAX_PINECONE_COLLECTION`, `INTERGRAX_PINECONE_METRIC`

## Example

```python
from intergrax.integrations.providers.pinecone.bundle import create_pinecone_vector_store

from langchain_core.documents import Document

store = create_pinecone_vector_store(api_key="pc-...", index_name="intergrax-rag", tenant_id="tenant-a")
store.add_documents(
    [Document(page_content="Intergrax overview", metadata={"source": "docs"})],
    [[0.01, 0.02, 0.03]],
    ids=["doc-1"],
)
hits = store.query([0.01, 0.02, 0.03], top_k=5)
store.delete(["doc-1"])
```

## Notes

Catalog bridge to ``intergrax/rag/`` — Pinecone SDK import only in ``opens.py``; RAG implementation unchanged.
