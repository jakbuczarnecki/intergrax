# LocalSearchAgent

Semantic search over locally indexed documents.

**Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) · **Plan:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)  
**Host:** [`applications/local_workspace_application/`](../../applications/local_workspace_application/)  
**Capability:** `local.workspace.search` (default agent on LKW host)

## Role in LKW pipeline

```text
user query → rag.retrieve → ranked evidence + citations
```

Upstream: indexer. Downstream: synthesizer or direct answer.

## Quick start

```bash
uv run pytest agents/local_search/tests -q
```

## HTTP example (via host)

```bash
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{"message":"find documents about project Alpha","capability":"local.workspace.search"}'
```

## Implementation status

| Wave | Scope |
|------|-------|
| LKW.0 | Scaffold + architecture (**Done**) |
| LKW.1 | Retrieve + evidence formatting | Planned |

## Authoring

See [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md).
