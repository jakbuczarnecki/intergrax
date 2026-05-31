# `chroma` integration — usage

**Category:** ``vector_store``  
**Catalog factory:** ``create_chroma_vector_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(vector_store=IntegrationSlug.CHROMA)
backend = profile.resolve(IntegrationCategory.VECTOR_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.vector_store.chroma.bundle import create_chroma_vector_store

backend = create_chroma_vector_store(**config_overrides)
```


## Environment variables

`INTERGRAX_CHROMA_MODE` (`embedded`|`http`); optional `INTERGRAX_CHROMA_HOST`, `INTERGRAX_CHROMA_PORT`, `INTERGRAX_CHROMA_PERSIST_DIRECTORY`, `INTERGRAX_CHROMA_COLLECTION`, `INTERGRAX_CHROMA_TENANT_ID`

## Example

```python
from intergrax.integrations.providers.vector_store.chroma.bundle import create_chroma_vector_store

store = create_chroma_vector_store(
    collection_name="intergrax-rag",
    tenant_id="tenant-a",
    mode="embedded",
    persist_directory=None,
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

Catalog bridge to ``intergrax/rag/`` — ``chromadb`` import only in ``opens.py``; RAG ``ChromaVectorStore`` unchanged.
