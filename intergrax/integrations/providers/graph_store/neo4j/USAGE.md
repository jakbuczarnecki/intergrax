# `neo4j` integration — usage

**Category:** ``graph_store``  
**Catalog factory:** ``create_neo4j_graph_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(graph_store=IntegrationSlug.NEO4J)
backend = profile.resolve(IntegrationCategory.GRAPH_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.graph_store.neo4j.bundle import create_neo4j_graph_store

backend = create_neo4j_graph_store(**config_overrides)
```


## Environment variables

`INTERGRAX_NEO4J_URL`, `INTERGRAX_NEO4J_USER`, `INTERGRAX_NEO4J_PASSWORD`

## Example

```python
from intergrax.integrations.providers.graph_store.neo4j.bundle import create_neo4j_graph_store

graph = create_neo4j_graph_store(base_url="bolt://localhost:7687", user="neo4j", password="...")
```

## Notes

Agent memory / tool graphs and **GraphRAG** persistence. Requires ``neo4j`` driver. Category ``graph_store`` (existing — no new §5.2.4 category).

### GraphRAG (Tier-0 RAG)

Wire Neo4j as the RAG graph backend (distinct from the integration ``GraphStore`` Cypher contract):

```python
import os
os.environ["INTERGRAX_RAG_GRAPH_ENABLED"] = "1"
os.environ["INTERGRAX_RAG_GRAPH_STORE"] = "neo4j"

from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack

stack = create_default_rag_stack()  # resolves Neo4j via Integration Library when env is set
```

Or inject explicitly:

```python
from intergrax.integrations.providers.graph_store.neo4j.bundle import create_neo4j_graph_store
from intergrax.rag.graph.providers.neo4j_rag_graph_store import Neo4jRagGraphStore

graph = Neo4jRagGraphStore(create_neo4j_graph_store(**config_overrides))
stack = create_default_rag_stack(graph_store=graph)
```
