# Intergrax Harness Environment

**Last updated:** 2026-06-01 · Phase U (Harness production hardening)

Operator and author guide for the **lab harness stack** — Tier-0 integrations, Tier-1 Nexus, Tier-3 `lab_application` wiring, platform skills, and observability. Business agents (Problem Radar, Vendor Discovery) are **Phase K** and out of scope here.

**Related:** [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase S · [SKILLS.md](SKILLS.md) · [INTEGRATIONS.md](INTEGRATIONS.md) · Architecture [§5.3](intergrax_runtime_architecture.md#53-harness-ai-alignment-conceptual-model)

---

## What “harness environment” means

```text
Tier-3  lab_application  →  IntegrationProfile + ToolProfile + SkillProfile + policy bundle
Tier-1  NexusLoop        →  UAEP, trace, context budget, delegation, ToolRuntime
Tier-0  integrations + tools + skills + llm_adapters
Tier-2  agents/          →  echo, signoff_probe, research, legal (existing reference agents)
```

Phase S completes **environment** readiness so any new agent uses Integration → Tool → Skill → Agent without further platform gaps.

---

## Lab harness stable integration stack

These slugs are **`stable`** in the catalog (see `intergrax/integrations/registry/harness_lab_stack.py`):

| Slug | Category | Role in lab |
|------|----------|-------------|
| `sqlite` | relational_store | Trace, events, checkpoints, experiments (when DB paths set) |
| `postgresql` | relational_store | Tier-2 product apps (stable; optional in lab preset) |
| `redis` | key_value_cache | Optional distributed cache / rate limits |
| `qdrant` | vector_store | Production RAG vector backend (when enabled in profile) |
| `slack` | notification + interaction | Product webhooks (optional) |
| `sentry` | observability_backend | Error capture (harness vendor profile) |
| `otel` | observability_backend | OTLP-oriented facade |
| `lab_json` | interaction_surface | `POST /v1/interactions/intake` lab JSON |
| `log` | notification_channel | Default lab notifications |

**Regression:** `pytest tests/unit/integrations/test_harness_lab_stable_stack.py -m gate`

---

## Integration profiles

| Profile | Factory | Use when |
|---------|---------|----------|
| `IntegrationProfile.lab_harness_preset()` | **Default** (lab app) | sqlite + log + lab_json + docling + OTEL (disable via `LAB_OTEL_ENABLED=false`) |
| `IntegrationProfile.lab()` | Legacy alias | sqlite + log + lab_json + docling (no OTEL) |
| `IntegrationProfile.harness_environment()` | Alias | Same as `lab_harness_preset(enable_otel=True)` |
| `IntegrationProfile.harness_lab()` | `LAB_HARNESS=true` | LangSmith + Sentry + PagerDuty vendor harness (M.9) |

Optional preset flags: `enable_redis`, `enable_qdrant` on `lab_harness_preset()`.

Lab wiring: `applications/lab_application/host/integration_wiring.py` → `wire_lab_integrations()`.

---

## OTLP / observability (S-Ops.2)

Environment variables (see `applications/lab_application/.env.example`):

| Variable | Purpose |
|----------|---------|
| `LAB_OTEL_ENABLED` | Use `IntegrationProfile.harness_environment()` (OTEL primary backend) |
| `INTERGRAX_OTEL_ENDPOINT` | Collector URL (default `http://localhost:4318`) |
| `INTERGRAX_OTEL_SERVICE_NAME` | Service name tag (default `intergrax`) |

Without a collector, the OTEL adapter uses a **noop exporter** — safe for CI and local gate tests.

**Trace and metrics (debug API):**

```bash
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
# POST /v1/lab/run  →  GET /debug/tasks/{id}/trace?include_runtime=true
# GET /debug/tasks/{id}/metrics
```

Runtime events: `GET /debug/tasks/{id}/events` when SQLite runtime events DB is wired.

**Context engineering events** (Tier-1): `CONTEXT_ASSEMBLED`, `CONTEXT_TRIMMED` — see architecture §28.1.

---

## Skill preset (lab)

`lab_skill_profile()` enables bundles: **`harness`**, **`legal`**, **`research`**.

### Platform harness skills (Phase S-H.1)

| skill_id | Tools | Purpose |
|----------|-------|---------|
| `harness.tool_smoke` | `rag.retrieve`, `websearch.query` | Tool catalog smoke |
| `harness.context_demo` | `rag.retrieve` | Context budget exercises |
| `harness.trace_read` | `sandbox.exec` | Isolated diagnostics |

Reference harness agents **must** set `AgentContract.skill_ids` — echo and signoff_probe use `harness.tool_smoke`; do not duplicate tool lists in agent code.

---

## Tool preset (lab)

Default enabled tools: `rag.retrieve`, `websearch.query`.

`sandbox.exec` is enabled **only** when a sandbox session is passed into `wire_lab_tools(sandbox_session=...)` (Phase U-Sec.3). Skills may still declare `sandbox.exec` for harness exercises — wire a session before expecting successful invocations.

With `LAB_HARNESS=true`, also: `errors.capture`, `observability.query_traces`, `pagerduty.trigger_incident`, etc.

---

## Security surfaces (Phase U)

| Variable | Default | Effect |
|----------|---------|--------|
| `INTERGRAX_HARNESS_API_KEY` | unset | When set, `POST /v1/lab/run`, `/debug/*`, `/v1/interactions/*`, and MCP (wrapper) require `X-Api-Key` or `Authorization: Bearer <key>` |
| `LAB_INCLUDE_MCP` | `false` | MCP mount is opt-in |
| `LAB_STRICT_HARNESS` | `false` | Reference agents use `production_mode=True`, governance service, and `trace_db_path` on `RuntimeConfig` |

Lab reference agents implement `HarnessReferenceAgent` + `UAEPAgent`; manifest bindings set `requires_uaep=True` (research agents excluded until UAEP migration).

---

## Policy bundle

`build_runtime_policy_bundle()` on lab registry build — typed `BudgetPolicy` / `PlanLoopPolicy` slots on `RuntimePolicyBundle`. Applied via `build_lab_agent_runtime_context()` (Phase U-Pol.1). Read order: architecture §42.11.5.

---

## Legacy RAG stack (U-Leg.2)

`intergrax.rag.answers` was **removed**. Use `intergrax.rag.retrieval.RetrievalService`. Archived code: `intergrax/legacy/rag_answers/` (notebooks only).

## Verification commands

```bash
uv run pytest -m gate -q
python scripts/check_harness_no_getattr.py
python scripts/check_tools_agent_imports.py
python scripts/check_tools_agent_run.py
python scripts/check_legacy_tool_plan_booleans.py
uv run pytest tests/unit/integrations/test_harness_lab_stable_stack.py -q
uv run pytest tests/unit/skills/test_harness_skill_bundle.py -q
uv run pytest tests/acceptance/agent_os/test_lab_application.py -m gate -q
```

Optional strict profile (CI `harness-strict` job):

```bash
set LAB_STRICT_HARNESS=true
set INTERGRAX_HARNESS_API_KEY=your-key
uv run pytest tests/unit/applications/test_lab_strict_harness.py -m gate -q
```

---

## Product agents and applications (end of plan — not default next)

Business agents (K.1 Problem Radar, K.2 Vendor Discovery) and new Tier-3 **product** applications are **last** in the [implementation plan](INTERGRAX_IMPLEMENTATION_PLAN.md) (§4.0 Band 3, **§6.3**). Harness work uses §6.1 only. Product work starts only after an explicit prioritization decision — not because Phase U or §4.1 is Done.
