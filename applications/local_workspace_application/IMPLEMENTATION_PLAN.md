# Local Knowledge Workspace (LKW) — Implementation Plan

**Derived from:** [`ARCHITECTURE.md`](ARCHITECTURE.md) §15 (do not diverge without updating architecture first)

Status: **LKW.0 Done** · **LKW.3 Done** (T6) · **Active queue: LKW.1**

Platform register: [`docs/intergrax_runtime_architecture.md` §6.3a LKW.*](../../docs/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated)

Principle: **local backend daemon** · **thin frontends** · **Slack optional** · **shadow writes only**

---

## 0. Product boundary reminder

| | Backend (`lkw-host`) | Frontend (clients) |
|---|---------------------|-------------------|
| **Runs on** | localhost daemon | Tray / Cursor / Slack / curl |
| **Contains** | Nexus, agents, RAG, index | UI + HTTP calls only |
| **Must not** | — | RAG, LLM, direct file index |

See architecture §4.

---

## 1. Wave queue

| ID | Title | Depends | Status | Priority |
|----|-------|---------|--------|----------|
| LKW.0 | Scaffold + architecture v2 | — | **Done** | — |
| LKW.1 | Domain UAEP: ingest + search + synthesize stub | LKW.0 | **Active** | Critical |
| LKW.2 | Graph pipeline + `local.workspace.*` skills | LKW.1 | Planned | High |
| LKW.3 | `filesystem.*` + allowlist | LKW.0 | **Done** | — |
| LKW.4 | Background ingest queue (`message_bus`) | LKW.1 | Planned | Medium |
| LKW.5 | `LKW_DATA_HOME` + Chroma persistence | LKW.1 | Planned | High |
| LKW.6 | OS daemon + interaction intake router | LKW.1 | Planned | High |
| LKW.6b | Slack Socket Mode (optional) | LKW.6 | Planned | Medium |
| LKW.7 | File watcher + incremental index | LKW.4, LKW.5 | Planned | Medium |
| LKW.8 | Tray thin client | LKW.6 | Deferred | Low |

---

## 2. Active wave — LKW.1

### Tasks

| ID | Task | Module | Owner |
|----|------|--------|-------|
| LKW.1.1 | Indexer steps: path validation + `rag.ingest_document` loop | `agents/local_indexer/steps/pipeline.py` | Tier-2 |
| LKW.1.2 | Search steps: `rag.retrieve` + evidence formatting | `agents/local_search/steps/pipeline.py` | Tier-2 |
| LKW.1.3 | Synthesizer stub: shadow `workspace.write_file` | `agents/local_synthesizer/steps/pipeline.py` | Tier-2 |
| LKW.1.4 | Acceptance test: fixture doc ingest → search cites source | `applications/.../tests/` or `tests/acceptance/` | Tier-3 |

### Acceptance criteria

- [ ] `POST /v1/local_workspace/run` with `metadata.source_paths` + `capability=local.workspace.index` completes
- [ ] Follow-up search returns answer referencing ingested content
- [ ] Synthesize with `shadow_workspace: true` writes artifact under shadow root
- [ ] **No Slack** required for above
- [ ] `uv run pytest` agent + host smoke green

### Out of scope (LKW.1)

- Tray UI, Slack, file watcher, OS service installer
- `local.workspace.*` skill bundle (LKW.2)

---

## 3. Next waves (summary)

Full task breakdown: **ARCHITECTURE.md §15.2**.

| ID | Key deliverables |
|----|------------------|
| LKW.2 | `intergrax/skills/providers/local/`, `skill_ids` on contracts, `graph_spec` pipeline |
| LKW.5 | `LKW_DATA_HOME` in settings, Chroma path under `data/chroma/` |
| LKW.6 | `scripts/lkw-host`, systemd/launchd/Windows unit, `wire_interaction_intake_service` |
| LKW.6b | Socket Mode → `/lkw` mapping; HITL notify |
| LKW.7 | `host/indexer_worker.py`, watcher + queue |
| LKW.8 | `clients/lkw-tray/` — HTTP-only client |

---

## 4. End-to-end validation scenarios

| ID | Scenario | Waves |
|----|----------|-------|
| E1 | Install → pick folders → index | LKW.5, LKW.6, LKW.8 |
| E2 | Search at desk via MCP | LKW.1 |
| E3 | Pipeline report | LKW.2 |
| E4 | Auto-index new file | LKW.7 |
| E5 | Slack search (optional) | LKW.6b |

---

## 5. Verification commands

```bash
# Host + agents (every PR touching LKW)
uv run pytest applications/local_workspace_application/local_workspace_application_tests -q
uv run pytest agents/local_indexer/tests agents/local_search/tests agents/local_synthesizer/tests -q

# Dev run (backend only)
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

---

## 6. Per-agent plans

- [`agents/local_indexer/IMPLEMENTATION_PLAN.md`](../../agents/local_indexer/IMPLEMENTATION_PLAN.md)
- [`agents/local_search/IMPLEMENTATION_PLAN.md`](../../agents/local_search/IMPLEMENTATION_PLAN.md)
- [`agents/local_synthesizer/IMPLEMENTATION_PLAN.md`](../../agents/local_synthesizer/IMPLEMENTATION_PLAN.md)

---

## 7. Platform alignment

- Harness maintenance: platform **§6.1** only
- LKW product: platform **`docs/plan/PLATFORM_FOUNDATION.md` §6.3a** — update when wave scope changes
- One wave per PR unless operator batches explicitly
