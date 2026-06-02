# Intergrax Harness Environment

**Last updated:** 2026-06-02 · Phase V **Done** (typed contracts + governance artifacts; L3/L4 CI closeout pending)

Operator and author guide for the **lab harness stack** — Tier-0 integrations, Tier-1 Nexus, Tier-3 `lab_application` wiring, platform skills, and observability. Business agents (Problem Radar, Vendor Discovery) are **Phase K** and out of scope here.

**Related:** [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase S/U/V · [SKILLS.md](SKILLS.md) · [INTEGRATIONS.md](INTEGRATIONS.md) · Architecture [§5.3](intergrax_runtime_architecture.md#53-harness-ai-alignment-conceptual-model) and [§53](intergrax_runtime_architecture.md#53-harness-architecture-hardening-addendum-post-u)

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

## Post-U continuation (Phase V)

Phase S/T/U established a production-configurable harness baseline.
Phase V architecture hardening and **Phase W-ML** harness contracts are **complete** in harness-only scope: modality tools (including `ml.batch_predict`), `VisionProfile`/`SpeechProfile`, `ModalityExecutionProfile` with `in_process` / `thread_pool` / `celery` offload (`INTERGRAX_MODALITY_EXECUTION`, optional `INTERGRAX_MODALITY_CELERY_BROKER_URL`), per-tool `modality_metrics` on trace + run export aggregation, lab + optional legal modality wiring, Triton/HF adapters, skills `harness.vision_qa` and `harness.modality_smoke`. Default continuation is **operational L3/L4 stability window**; Tier-3 hosts may additionally wire the shared `message_bus` Celery bundle for non-modality workloads.

Primary Phase V tracks impacting the harness environment:

- capability graph + compatibility gates (`V-CG.*`)
- context/prompt/evaluation regression discipline (`V-CE.*`, `V-PE.*`, `V-EVAL.*`)
- security/data and cost/resource governance hardening (`V-SEC.*`, `V-COST.*`)
- architecture metrics and debt governance (`V-AM.*`)

Current baseline delivered (Phase V V1):

- `V-CG.1` typed capability graph schema
- `V-CG.2` graph lineage report
- `V-CG.3` blast-radius impact report
- `V-CG.4` compatibility guard (report + `--enforce`)
- `V-AM.1` architecture metrics baseline contract
- `V-AM.2` metrics pipeline snapshots, trend, and gate result contracts
- `V-ALG.1` agent certification gate contract
- `V-ALG.2` promotion flow contract (`dev -> staging -> production`) with evidence checks
- `V-ALG.3` lifecycle transition governance (`production -> deprecated -> retired`) with migration windows
- `V-ALG.4` production ownership guard contract (owner/on-call/escalation + runbook)
- `V-EVAL.1` unified evaluation mode contracts (`offline`, `online`, `shadow`, `human`)
- `V-EVAL.2` golden dataset + scenario library + versioned evaluation asset bundle contracts
- `V-EVAL.3` automated evaluators (rule-based + LLM-judge) and persisted evaluator outputs
- `V-EVAL.4` evaluation registry trend/comparison report contracts
- `V-AM.3` governance and observability coverage report contracts
- `V-AM.4` architecture debt governance policy and periodic review cadence contracts
- `V-SEC.1` prompt injection defense profile and adversarial deny-path tests
- `V-SEC.2` tool injection defense policy (allowed tool IDs, blocked argument tokens, capability match controls)
- `V-SEC.3` retrieval poisoning defense with trust-score quarantine flow
- `V-SEC.4` tenant isolation verification and security audit trail checks
- `V-COST.1` multi-scope budget envelope governance (`tenant`, `application`, `agent`, `model`, `tool`)
- `V-COST.2` quota enforcement with deterministic `allow/degrade/deny` behavior
- `V-COST.3` spend/token forecast and anomaly detection reports
- `V-COST.4` optimization recommendations with policy guardrails
- `V-CE.1` context relevance/freshness/confidence scoring contracts
- `V-CE.2` duplicate suppression and context quality threshold governance
- `V-PE.1` prompt registry governance (owner/version/risk metadata)
- `V-PE.2` layered prompt composition model (`system`, `policy`, `task`, `context`)
- `V-CE.3` context regression benchmark baseline vs current comparison
- `V-CE.4` retrieval effectiveness metrics (`precision@k`, `recall@k`)
- `V-PE.3` deterministic policy injection overlays with trace records
- `V-PE.4` prompt regression and adversarial test suite contracts
- `V-MA.1` multi-agent coordination pattern catalog
- `V-MA.2` coordination pattern selection matrix (risk/latency/cost/complexity)
- `V-MA.3` pattern-specific multi-agent acceptance contracts
- `V-KG.1` Graph-RAG architecture contract (nodes/edges/types)
- `V-KG.2` hybrid retrieval path (vector + keyword + graph fusion)
- `V-KG.3` graph-backed explainability trace provenance fields
- report artifacts script: `scripts/phase_v_foundations_report.py`
- graph guard script: `scripts/phase_v_capability_graph_guard.py`
- governance artifacts script: `scripts/phase_v_governance_report.py`
- `V-V6.1` bounded adaptive loop governance (`adaptive_governance.py`)
- `V-V6.2` L3/L4 maturity gate evidence (`maturity_gate_evidence.py`)
- `V-V6.3` CI closeout gate: `scripts/phase_v_closeout_gate.py` (`--enforce`, `--enforce-l4`)

Execution references in the implementation plan:

- **Phase V stream map:** `Phase V — Harness Architecture Hardening (post-U)`
- **Execution order:** `Phase V — Execution matrix (dependencies and order)`
- **Acceptance thresholds:** `Phase V — KPI thresholds and acceptance metrics`
- **Maturity gates:** `Phase V — L3/L4 gate evidence (architecture maturity)`

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
uv run python scripts/phase_v_foundations_report.py
uv run python scripts/phase_v_capability_graph_guard.py
uv run python scripts/phase_v_governance_report.py
uv run python scripts/phase_v_closeout_gate.py --enforce --enforce-l4
```

Optional strict profile (CI `harness-strict` job):

```bash
set LAB_STRICT_HARNESS=true
set INTERGRAX_HARNESS_API_KEY=your-key
uv run pytest tests/unit/applications/test_lab_strict_harness.py -m gate -q
```

---

## Product agents and applications (end of plan — not default next)

Business agents (K.1 Problem Radar, K.2 Vendor Discovery) and new Tier-3 **product** applications are **last** in the [implementation plan](INTERGRAX_IMPLEMENTATION_PLAN.md) (§4.0 Band 3, **§6.3**). Harness work uses **§6.1 + §6.2 (Phase V)**. Product work starts only after an explicit prioritization decision — not because Phase U, §4.1, or initial Phase V waves are active.
