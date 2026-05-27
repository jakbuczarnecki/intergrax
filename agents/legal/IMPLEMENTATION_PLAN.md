# Legal backend — implementation plan (step-by-step)

This is the **operational** companion to [ROADMAP.md](ROADMAP.md): ordered work, integrations, and components to reach a **self-hosted service / SaaS-ready** Legal API. Check items off as PRs merge.

**Legend:** `[ ]` not started · `[~]` in progress · `[x]` done (update in your branch/PR).

---

## Phase A — Product shell MVP (front can call chat)

Goal: external client runs **`POST /v1/legal/chat`** with stable JSON + OpenAPI in non-prod; prod path uses **`identity_source=context_only`** and API keys.

| Step | Task | Notes / integration |
|------|------|---------------------|
| A.1 | `[x]` Tier-3 host package | `legal_application.host` — `main.py`, `factory.py`, `settings.py`, `wiring.py` |
| A.2 | `[ ]` **Phase A DoD checklist** | Document + script: curl/Postman from repo root with `PYTHONPATH=applications`, prod env + API key |
| A.3 | `[ ]` **Dockerfile** for API | Single image: `uvicorn legal_application.host.main:app`, env-only config |
| A.4 | `[ ]` **docker-compose (optional)** | API + placeholder `postgres` / `redis` stubs for local dev (even if not wired yet) |
| A.5 | `[ ]` **Auth decision recorded** | MVP: API key only **or** add **JWT** (`JwtAuthProvider` + verifier) → `RequestContext`; update `.env.example` |
| A.6 | `[ ]` **CORS / allowed_hosts** prod defaults | Align with first real front origin |

**Exit:** any thin client can chat with token; no file upload required.

---

## Phase B — Persistent sessions & history

Goal: **`session_id` survives restarts**; UI can load conversation without replaying full LLM history only via chat.

| Step | Task | Notes / integration |
|------|------|---------------------|
| B.1 | `[ ]` **Session storage adapter** | Wire `SessionManager` (or equivalent in `fastapi_core` / Nexus) to **SQLite (already optional)** or **Postgres** |
| B.2 | `[ ]` **DB schema + migrations** | Alembic (or chosen tool) in host or `infra/`; tenant_id on session rows |
| B.3 | `[ ]|``GET /v1/legal/sessions/{id}/messages`** (or under `/v1/...`) | Thin read model for front; authz: tenant owns session |
| B.4 | `[ ]` **Lifecycle** | `lifespan` in host: open/close DB pool, optional migration check in dev |

**Exit:** restart server does not lose sessions (except deliberate in-memory dev).

---

## Phase C — Attachments + RAG (sync MVP)

Goal: **upload small file → chunk/embed → chat references `attachment_id` → RAG on**.

| Step | Task | Notes / integration |
|------|------|---------------------|
| C.1 | `[ ]` **Blob storage port** | Interface: `put/get/delete` — MVP: local disk path per env; prod: **S3-compatible** |
| C.2 | `[ ]` **`POST /v1/legal/attachments`** (or `/v1/files`) | Multipart or JSON+presigned URL later; returns `attachment_id`, size, content-type |
| C.3 | `[ ]` **Metadata store** | DB table: tenant_id, user_id, attachment_id, storage_key, status |
| C.4 | `[ ]` **Ingest pipeline (sync)** | Reuse platform **AttachmentIngestionService** / RAG path; **tenant-scoped** collection/namespace |
| C.5 | `[ ]` **Chat contract** | `LegalChatRequestV1.attachments` populated; compliance + SKU unchanged |

**Exit:** E2E small PDF: upload → chat with attachment → grounded answer (deterministic tests where possible).

---

## Phase D — Async ingest & jobs

Goal: large files **do not block** HTTP workers.

| Step | Task | Notes / integration |
|------|------|---------------------|
| D.1 | `[ ]` **Queue port** | **Redis + RQ / Celery / arq** — pick one aligned with Intergrax distributed roadmap |
| D.2 | `[ ]` **Job model + DB** | `pending | running | succeeded | failed`, error message, correlation id |
| D.3 | `[ ]` **Worker process** | Same image different CMD or sidecar; consumes `ingest_attachment` |
| D.4 | `[ ]` **`GET /v1/jobs/{id}`** | Authz by tenant; optional webhook later |

**Exit:** large upload returns `job_id`; UI polls until success before attaching to chat.

---

## Phase E — Operations & production hardening

Goal: safe multi-tenant default, observable, limitable.

| Step | Task | Notes / integration |
|------|------|---------------------|
| E.1 | `[ ]` **Rate limiting** | `RateLimitPolicy` on `/v1/legal/chat` (and upload) **per tenant/API key** |
| E.2 | `[ ]` **Structured logging + correlation** | Request id through `RequestContext`; JSON logs in prod |
| E.3 | `[ ]` **Metrics / tracing** | OTEL hooks where `fastapi_core` exposes them; RED metrics for chat/jobs |
| E.4 | `[ ]` **Secrets & environments** | Stage/prod profiles; no secrets in image; reference Vault/KMS when applicable |
| E.5 | `[ ]` **Runbook** | Backup DB, blob retention, RPO/RTO, upgrade procedure |

**Exit:** ops can run and debug service without reading Python internals.

---

## Phase F — SaaS vs enterprise packaging

Goal: **same binary**, different deploy and policy—not a second runtime.

| Step | Task | Notes / integration |
|------|------|---------------------|
| F.1 | `[ ]` **Multi-tenant guardrails** | Enforce tenant on every route; no cross-tenant IDs in responses |
| F.2 | `[ ]` **Provisioning hooks (optional)** | Stub API or admin path: create tenant, default quotas (billing system out of scope or separate service) |
| F.3 | `[ ]` **Plan / quota config** | Env or DB: max attachments, RAG size, RPM — wire to rate limit + job rejection |
| F.4 | `[ ]` **Enterprise variant** | Single-tenant deploy doc: dedicated DB, SSO ingress, optional CMK for blobs |

**Exit:** you can honestly sell “managed Legal API” or “private instance” with a config matrix, not a fork.

---

## Parallel: Agent Factory (Intergrax, Tier‑0/1)

Do **after** Legal host proves B + C (sync) so patterns are real.

| Step | Task |
|------|------|
| FF.1 | `[ ]` `AgentSpec` (Pydantic): `agent_id`, profile builder, route prefix, compliance defaults |
| FF.2 | `[ ]` `register_tier2_agent` + generic `mount_agent_routes` (Protocol for service facade) |
| FF.3 | `[ ]` Template package + cookiecutter/make target |
| FF.4 | `[ ]` Contract tests: tenant isolation, compliance surface, session round-trip |

**Rule:** `intergrax` factory code **must not** import `legal` or `legal_application`.

---

## Dependency graph (simplified)

```
A (shell + auth) → B (sessions) → C (sync RAG) → D (async) → E (ops) → F (SaaS packaging)
                                                                    ↑
Factory (FF*) parallel after C stable ─────────────────────────────┘
```

---

## What is *not* in this repo alone

- **Billing / payments** — usually external or thin BFF; stub webhooks only if needed.
- **IdP / SSO** — enterprise doc + integration at ingress or JWT issuer.
- **Global CDN** — front static assets; API stays origin-facing.

Update this file when steps split across PRs or priorities change.
