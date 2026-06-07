# LocalIndexerAgent

Indexes user-local files into the Intergrax RAG vector store.

**Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)  
**Host:** [`applications/local_workspace_application/`](../../applications/local_workspace_application/)  
**Capability:** `local.workspace.index`

## Role in LKW pipeline

```text
source_paths → parse → chunk → embed → vector store
```

Downstream: `LocalSearchAgent` retrieves indexed chunks.

## Quick start

```bash
uv run pytest agents/local_indexer/tests -q
```

## Implementation status

| Wave | Scope |
|------|-------|
| LKW.0 | Scaffold + architecture (**Done**) |
| LKW.1 | UAEP ingest steps + smoke with real paths | Planned |

## Authoring

1. Implement domain logic in `steps/pipeline.py`
2. Adjust [`prompts/system.md`](prompts/system.md)
3. Register tools on contract — see [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md)

**Do not** import from `applications/` — Tier-2 only.
