# LKW Tier-2 — tool invocation pattern anchor

**Cursor read this file instead of** `uaep.py`, `boundary_demo_agent.py`, or other runtime internals when implementing LKW agent steps.

Canonical helpers: [`runtime_helpers.py`](runtime_helpers.py)

---

## Implementation point

Each LKW agent wires domain logic in:

```text
agents/<agent>/steps/<job>.py   ← edit here
agents/<agent>/<agent>_agent.py → act() calls run_*_job(step_ctx)
```

| Agent | Step module | Primary tool |
|-------|-------------|--------------|
| `local_indexer` | `steps/index_job.py` | `rag.ingest_document` |
| `local_search` | `steps/search_job.py` | `rag.retrieve` |
| `local_synthesizer` | `steps/synthesize_job.py` | `workspace.write_file` |

---

## Minimal pattern (copy into `steps/<job>.py`)

```python
from intergrax.contracts.agent_step_context import AgentStepContext
from lkw_shared.tool_catalog import RAG_INGEST_TOOL_ID
from lkw_shared.runtime_helpers import (
    allowlist_roots,
    exec_ctx_from_step,
    invoke_catalog_tool,
    parse_metadata_list,
    request_metadata,
    validate_allowlisted_files,
)
from lkw_shared.read_allowlist import require_read_allowlist_roots

STEP_ID = "local_indexer_step"


async def run_index_job(step_ctx: AgentStepContext) -> dict[str, object]:
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = request_metadata(exec_ctx)
    source_paths = parse_metadata_list(metadata, "source_paths")
    if not source_paths:
        return {"summary": "...", "ingest_summary": {"used": False, "reason": "source_paths_missing"}}

    roots = allowlist_roots(exec_ctx)
    require_read_allowlist_roots(roots if roots else None)
    validated, rejected = validate_allowlisted_files(source_paths, roots)

    ingested = []
    if exec_ctx is not None:
        for path in validated:
            ingested.append(
                await invoke_catalog_tool(
                    exec_ctx,
                    tool_name=RAG_INGEST_TOOL_ID,
                    agent_id=step_ctx.agent_id,
                    step_id=STEP_ID,
                    tool_input={"source_path": str(path), "metadata": metadata},
                )
            )
    return {"summary": "...", "ingest_summary": {"used": bool(ingested), "ingested": ingested, "rejected_paths": rejected}}
```

---

## Rules

- **Do not** import RAG ingest pipeline internals — use catalog tools only.
- **Do not** mutate user files — read via allowlisted paths only.
- Allowlist: `validate_allowlisted_files` + `allowlist_roots` (LKW.3 / `INTERGRAX_ALLOWED_READ_ROOTS`).
- `exec_ctx` is available as `step_ctx.metadata["uaep_exec_ctx"]` (set by `acp_uaep_shim`).

---

## Cursor stop condition

When the prompt cites this file or agent `ARCHITECTURE.md` § **Pattern anchor**, stop searching and edit the cited `steps/<job>.py` only.
