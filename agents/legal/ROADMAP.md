# Roadmap: Intergrax + Legal → production backend + factory (Tier‑3)

**North Star:** a production-ready Legal backend any client (Flutter, web, mobile, desktop) integrates through a **stable REST/OpenAPI** surface—with auth, sessions, attachments, RAG ingest, history, and observability—plus an **Agent Factory** in Intergrax to spin up other domains on the same skeleton.

**Concrete step-by-step implementation (tasks, integrations, SaaS checklist):** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Architecture layers (no Nexus fork)

| Layer | Role |
|-------|------|
| Intergrax L1/L2 | RAG primitives, session stores, runtime, tracing, budget, governance—**no “shop” product logic**. |
| Tier‑2 `legal` | Pipeline, config, governance, runtime bridge — **`agents/legal/`** |
| Tier‑3 **product host** | `legal_application.host`: FastAPI entrypoint, env DI, mounts `fastapi_core` + `mount_legal_agent_routes`, deploy wiring. |
| Tier‑3 **agent factory** (Intergrax) | Contracts + templates: `AgentSpec`, registry, config factory, contract tests, mount helpers—**no Legal domain inside factory**. |

**Rule:** the product backend **composes** the framework; it does not fork Nexus or duplicate the Legal pipeline as a second engine.

## Where the Tier‑3 shell lives today

| Roadmap ID | Deliverable | Repo location |
|------------|-------------|---------------|
| A1 | Product shell entrypoint, settings | `applications/legal_application/host/main.py`, `settings.py` |
| A2 | `create_app` + Legal mount + wiring | `applications/legal_application/host/factory.py`, `wiring.py` |
| A3 | Prod identity (`context_only` + API keys) | `LegalBackendSettings`, `ApiKeyConfig` in `factory.py` |
| A4 | Runbook | `README.md`, `HOST_README.md`, repo `.env.example` |

**ASGI app:** `legal_application.host.main:app`

**Phase A MVP “done for front wire-up”** when an external client can call **`POST /v1/legal/chat`** with a token and stable JSON—upload/RAG async can follow in later phases.

## Phases B–F (tracking)

- **B — Sessions & history:** DB-backed `SessionManager` / storage adapter in the host; optional `GET …/sessions/{id}/messages`; migrations.
- **C — Files + RAG (sync MVP):** `POST /v1/…/attachments`, blob metadata, tenant-scoped vector store, link `attachment_id` to chat requests.
- **D — Async ingest:** queue port, `ingest_attachment` job, `GET /v1/jobs/{id}` (or webhook later).
- **E — Operations:** rate limits, OTEL hooks, secrets profiles, runbooks.
- **F — Commercial packaging:** SaaS vs enterprise vs vertical = mostly config + deploy, not a new runtime.

## Parallel track: Agent Factory (Intergrax)

After Legal host hardening, generalize repeatable wiring into `intergrax/agent_kit` (and follow-on modules): `AgentSpec`, `register_tier2_agent`, `mount_agent_routes`, templates, contract tests. **Factory must not import Legal domain code.**

## Critical dependencies

- Per-tenant vector / collection strategy for RAG.
- JWT (or chosen auth) integrated with `RequestContext` when moving beyond API-key MVP.
- Keep factory free of Legal-specific imports.

---

*This file is the working map from internal planning to concrete PRs; update checkboxes or status in PR descriptions as phases land.*
