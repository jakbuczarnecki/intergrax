# Intergrax Assistant — Implementation Plan

**The implementation map** for this Tier-3 application — phases, status, gaps, and verification.

**Status:** Working draft (2026-06-08) — lab profile, architecture baseline v1

**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
**ADR:** [`adr/ADR-INTERGRAX_ASSISTANT-001.md`](adr/ADR-INTERGRAX_ASSISTANT-001.md)
**Platform plan:** [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../docs/project/architecture/intergrax_runtime_architecture.md)

**Principle:** compose Tier-0 · no business logic in Nexus · manifest-driven wiring

---

## 0. Scope at a glance

| Field | Value |
|-------|-------|
| Package | `intergrax_assistant_application` |
| Profile | `lab` |
| Route prefix | `/v1/intergrax_assistant` |
| Default port | `8096` |
| Hub agent | `intergrax_assistant` (`platform.assist`) |
| Specialists | Echo, Legal, Research, Summary — env opt-in |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| IAA-0 | Architecture + ADR-001 hub-and-spoke | **Done** | Critical | This iteration |
| IAA-1 | Environment profile — harness lab + engine planner + LLM from env | **Done** | Critical | `host/environment_profile.py` |
| IAA-2 | Dynamic manifest — specialist roster flags | **Done** | High | `manifest.py` + `settings.py` |
| IAA-3 | Host smoke tests green | **Done** | High | `tests/host` |
| IAA-4 | Deploy triad present | **Done** | High | `docker`, `BUILD_AND_DEPLOY.md` |
| IAA-5 | Dedicated `/chat` route + session contract | Planned | High | Mirror `legal_application` serving |
| IAA-6 | Hub UAEP — conversational tool loop + file attachments | Planned | High | `agents/intergrax_assistant/steps` |
| IAA-7 | Workspace RAG collections per user | Planned | Medium | Tier-0 RAG wiring |
| IAA-8 | `ApplicationGraphSpec` for common delegation paths | Planned | Medium | `environment_profile.graph_spec` |
| IAA-9 | Product profile promotion (`--profile product`) | Planned | Low | After HITL/security review |

---

## 2. Verification

```bash
uv run pytest applications/intergrax_assistant_application/tests -q
uv run pytest agents/intergrax_assistant/tests -q
```

Local run:

```bash
cp applications/intergrax_assistant_application/.env.example applications/intergrax_assistant_application/.env
uv run uvicorn intergrax_assistant_application.host.main:app --host 127.0.0.1 --port 8096
curl -s http://127.0.0.1:8096/v1/intergrax_assistant/agents
```

---

## 3. Platform alignment

Explicit product reprioritization — harness chat lab for architecture experimentation.  
Platform maintenance queue: [`§6.1`](../../../docs/project/architecture/intergrax_runtime_architecture.md).
Business backlog pattern: [`§6.3a`](../../../docs/project/architecture/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated) (IAA tracked locally, not in platform §6.3a until promoted).
