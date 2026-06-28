# Confluence (confluence)

Category: `wiki_knowledge`

## Legacy facade

- `create_confluence_integration()` remains backward-compatible.

## Contract-based integration

- `ConfluenceWikiKnowledgeIntegration` derives from the category-specific contract.
- Factory: `create_confluence_wiki_knowledge_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
