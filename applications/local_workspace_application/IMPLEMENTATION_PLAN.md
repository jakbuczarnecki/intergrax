# Local Knowledge Workspace (LKW) — Implementation Plan

**The implementation map** for this Tier-3 product environment — phases, status, gaps, and verification.

Status: **LKW.0 Done** (2026-06-07) — Wave **LKW.1** active

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Platform register: [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md` §6.3a LKW.*](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63a-business-backlog-register-consolidated)  
Agents: [`local_indexer`](../../agents/local_indexer/IMPLEMENTATION_PLAN.md) · [`local_search`](../../agents/local_search/IMPLEMENTATION_PLAN.md) · [`local_synthesizer`](../../agents/local_synthesizer/IMPLEMENTATION_PLAN.md)

Principle: **product reprioritization after harness GA** · **read-heavy, shadow writes only**

---

## Documentation model

| Topic | Where |
|-------|--------|
| Product purpose, four-layer stack, I/O | **ARCHITECTURE.md** |
| Wave schedule, LKW.* IDs | **This file** (local detail) + platform **§6.3a** (canonical register) |
| Per-agent steps | `agents/local_*/IMPLEMENTATION_PLAN.md` |

Do not duplicate the full LKW register — update platform **§6.3a** when wave scope changes.

---

## 0. Scope at a glance

| Field | Value |
|-------|-------|
| Package | `local_workspace_application` |
| Profile | product (business) |
| Agents | `local_indexer`, `local_search`, `local_synthesizer` |
| Primary capabilities | `local.workspace.index`, `.search`, `.synthesize` |

---

## 1. Implementation queue (waves)

| ID | Wave | Task | Status | Priority | Notes |
|----|------|------|--------|----------|-------|
| LKW.0 | 0 | Scaffold + architecture baseline | **Done** | High | Agents + host |
| LKW.1 | 1 | Ingest + search smoke on explicit paths | Planned | **High** | Active queue |
| LKW.2 | 2 | Multi-agent pipeline graph | Planned | High | After LKW.1 |
| LKW.3 | 3 | Tier-0 `filesystem.*` read tools | Planned | Medium | Allowlist policy |
| LKW.4 | 4 | Background ingest queue | Planned | Medium | After LKW.2 |
| LKW.6 | 6 | Local OS daemon + interaction intake | Planned | High | Host extension |
| LKW.6b | 6b | Slack Socket Mode intake | Planned | Medium | After LKW.6 |
| LKW.7 | 7 | File watcher + incremental index | Planned | Medium | After LKW.3 |
| LKW.8 | 8 | Tray / file-picker UI | Deferred | Low | Product shell |

---

## 2. Verification

```bash
uv run pytest applications/local_workspace_application/local_workspace_application_tests -q
uv run pytest agents/local_indexer/tests agents/local_search/tests agents/local_synthesizer/tests -q
```

Local run:

```bash
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8093
```

---

## 3. Platform alignment

LKW is the **first business product** after harness maturity — tracked in platform plan **§6.3a**.
Harness-only work remains in platform **§6.1**; do not mix scopes in a single PR.
