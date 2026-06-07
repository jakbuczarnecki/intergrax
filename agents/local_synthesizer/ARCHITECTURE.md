# LocalSynthesizerAgent — architecture

**Capability:** `local.workspace.synthesize`  
**Host:** [`applications/local_workspace_application/`](../../applications/local_workspace_application/)  
**Status:** Scaffold — domain steps pending Wave LKW.1

---

## Purpose

Transform retrieved evidence into user-requested deliverables — emails, reports, cost estimates, project summaries — and write results **only** to the shadow workspace. Never mutates the user's original files.

---

## Responsibilities

| In scope | Out of scope |
|----------|--------------|
| LLM synthesis from evidence (graph handoff or metadata) | Indexing or retrieval |
| Template-driven output (email, report, estimate) | Direct writes to user home directories |
| Shadow workspace artifacts via `workspace.write_file` | Unsandboxed risky shell commands |
| Optional HITL for sensitive exports | |

---

## Inputs

| Source | Field | Description |
|--------|-------|-------------|
| Task message | Deliverable instructions | e.g. „przygotuj mail do klienta” |
| Task metadata | `synthesis_template` | `email` \| `report` \| `estimate` \| `custom` |
| Task metadata | `evidence` | Chunks from search agent (graph handoff) |
| Task metadata | `shadow_workspace` | `True` — required for writes |
| Task metadata | `style_guidelines` | Optional user tone/format rules |

---

## Outputs

| Field | Description |
|-------|-------------|
| `artifact_paths` | Relative paths in shadow workspace |
| `shadow_workspace_id` | Nexus metadata for download/debug |
| `preview` | Short text preview for UI |

---

## UAEP pipeline

```text
steps/pipeline.py
  1. load_evidence         → memory / graph context
  2. select_template
  3. llm_synthesize
  4. write_artifacts       → workspace.write_file
  5. finalize_metadata
```

Enable shadow workspace on the Nexus task:

```python
Task(..., metadata={"shadow_workspace": True})
```

---

## Tools

- `workspace.write_file`, `workspace.read_file`, `workspace.list_files`
- `memory.read`
- LLM via `RuntimeConfig` (no direct vendor SDK in agent)

---

## Prompts

[`prompts/system.md`](prompts/system.md)

---

## Tests

```bash
uv run pytest agents/local_synthesizer/tests -q
```

---

## References

- Shadow workspace: [`docs/AGENT_CREATION_GUIDE.md` Appendix B](../../docs/AGENT_CREATION_GUIDE.md#appendix-b--shadow-workspace-and-sandbox)
- LKW architecture: [`applications/local_workspace_application/ARCHITECTURE.md`](../../applications/local_workspace_application/ARCHITECTURE.md)
