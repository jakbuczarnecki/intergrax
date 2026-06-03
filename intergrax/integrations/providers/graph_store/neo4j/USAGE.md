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

register_default_integrations()
profile = IntegrationProfile(graph_store="neo4j")
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

Agent memory / tool graphs. Requires ``neo4j`` driver. New category ``graph_store``.
