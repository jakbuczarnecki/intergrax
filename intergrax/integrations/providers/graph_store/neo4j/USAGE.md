# Neo4J (neo4j)

Category: `graph_store`

## Legacy facade

- `create_neo4j_graph_store()` remains backward-compatible.

## Contract-based integration

- `Neo4jGraphStoreIntegration` derives from the category-specific contract.
- Factory: `create_neo4j_graph_store_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
