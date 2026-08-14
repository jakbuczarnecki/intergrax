# LocalSynthesizerAgent — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

**Capability:** `local.workspace.synthesize`  
**Host:** [`applications/local_workspace_application`](../../applications/local_workspace_application/)
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
`on_next_step` / cognitive pattern hooks
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

## Pattern anchor (Cursor — read instead of runtime grep)

| Item | Location |
|------|----------|
| Generic `invoke_tool` helpers | [`intergrax/agents/authoring/runtime_tool_helpers.py`](../../../../../intergrax/agents/authoring/runtime_tool_helpers.py) |
| Workspace write tool id | [`intergrax/tools/providers/workspace/service.py`](../../../../../intergrax/tools/providers/workspace/service.py) |
| **Implementation point** | [`steps/synthesize_job.py`](steps/synthesize_job.py) — `run_synthesize_job` |

Do **not** read `uaep.py` or `boundary_demo` to discover tool invocation for this agent.

---

## Integrations, tools, and skills

### Integrations (indirect)

| Slot | Role |
|------|------|
| runtime `ShadowWorkspace` | Artifact writes — **not** an integration slug; bound via `ToolWiringContext.shadow_workspace` |
| `relational_store` | `sqlite` — task memory for evidence handoff |
| LLM adapter | Resolved from `ApplicationEnvironmentProfile.llm_profile` / env — no vendor SDK in agent |

### Tools

| tool_id | Role |
|---------|------|
| `workspace.write_file` | Primary — save draft/report to shadow |
| `workspace.read_file` | Read back for revision loops |
| `workspace.list_files` | List deliverables |
| `workspace.search` | Grep within shadow artifacts |
| `memory.read` | Load evidence from search step / graph handoff |

### Skills (planned LKW.2)

| `skill_id` | `tool_ids` | Policy |
|------------|------------|--------|
| `local.workspace.synthesize` | `workspace.*`, `memory.read` | `policy_fragment_id` for HITL on financial/legal drafts (TBD) |

Requires task `metadata={"shadow_workspace": True}`.

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

- Shadow workspace: [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md` Appendix B](../../guides/AGENT_CREATION_GUIDE.md#appendix-b--shadow-workspace-and-sandbox)
- LKW architecture: [`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../../applications/local_workspace_application/docs/ARCHITECTURE.md)
