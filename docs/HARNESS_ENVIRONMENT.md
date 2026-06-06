# Intergrax Harness Environment

**Last updated:** 2026-06-02 · Phase V + ORCH + TS + INT + RAG + CTX + PE **Done**; gate **623**

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

**Tier-3 observability wiring (Phase OBS):** `wire_application_observability()` maps `ObservabilityProfile` to `NexusObservabilityStores`; `build_harness_host_runtime()` validates assembly via `observability_assembly_resolver`. Author map: [`AGENT_CREATION_GUIDE.md` Appendix Q](AGENT_CREATION_GUIDE.md#appendix-q--observability-control-plane-closeout). CI: `scripts/check_harness_observability_wiring.py`.

**Tier-3 reliability wiring (Phase REL):** `wire_application_reliability()` maps `ReliabilityProfile` to `IdempotencyStore` and `IntegrationCircuitBreakerConfig`; `materialize_runtime_config()` applies idempotency to `RuntimeConfig`. Author map: [`AGENT_CREATION_GUIDE.md` Appendix R](AGENT_CREATION_GUIDE.md#appendix-r--reliability-control-plane-closeout). CI: `scripts/check_harness_reliability_wiring.py`.

**Tier-3 security wiring (Phase SEC):** `wire_application_security()` maps `ApplicationSecurityProfile` to V-SEC middleware; `build_harness_host_runtime()` validates assembly via `security_assembly_resolver`. Author map: [`AGENT_CREATION_GUIDE.md` Appendix S](AGENT_CREATION_GUIDE.md#appendix-s--security-control-plane-closeout). CI: `scripts/check_harness_security_wiring.py`.

**Tier-3 cost wiring (Phase COST):** `wire_application_cost()` maps `CostProfile` to `BudgetPolicy` / `RunBudget`; `wire_policy_bundle()` merges cost governance into `RuntimePolicyBundle`. Author map: [`AGENT_CREATION_GUIDE.md` Appendix T](AGENT_CREATION_GUIDE.md#appendix-t--cost-governance-control-plane-closeout). CI: `scripts/check_harness_cost_wiring.py`.

**Tier-3 evaluation wiring (Phase EVAL):** `wire_application_evaluation()` maps `EvaluationProfile` to `OnlineEvaluationRegistry` / governance bridge; `wire_policy_bundle()` merges `evaluation_governance` into `RuntimePolicyBundle`. Author map: [`AGENT_CREATION_GUIDE.md` Appendix U](AGENT_CREATION_GUIDE.md#appendix-u--evaluation-control-plane-closeout). CI: `scripts/check_harness_evaluation_wiring.py`.

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
| `harness.reliability_smoke` | `observability.query_traces`, `rag.retrieve` | Reliability / ops smoke (W-OPS.8) |
| `harness.policy_smoke` | `rag.retrieve`, `websearch.query` | Policy bundle smoke (W-OPS.8) |
| `harness.stack_demo` | requires `harness.tool_smoke` | `requires_skills` chain demo (W-OPS.9) |

Reference harness agents **must** set `AgentContract.skill_ids` — echo and signoff_probe use `harness.tool_smoke`; do not duplicate tool lists in agent code.

---

## Tool preset (lab)

Default enabled tools: `rag.retrieve`, `websearch.query`.

`sandbox.exec` is enabled **only** when a sandbox session is passed into `wire_lab_tools(sandbox_session=...)` (Phase U-Sec.3). Skills may still declare `sandbox.exec` for harness exercises — wire a session before expecting successful invocations.

With `LAB_HARNESS=true`, also: `errors.capture`, `observability.query_traces`, `pagerduty.trigger_incident`, etc.

---

## Harness SLO catalog (Phase W-OPS.4)

Operational L3 evidence is separate from `phase_v_closeout_gate` (contract CI). Use `scripts/phase_w_ops_evidence.py` (CI: non-enforcing) and set `W_OPS_RELEASE_CYCLES` or append cycles to `build/architecture_hardening/release_cycles.json` after release board sign-off.

**Lab stack health (W-OPS.10):** `health_check_harness_lab_stack()` probes every `HARNESS_LAB_STABLE_SLUGS` catalog entry via `health_check_catalog_slugs` with circuit breaker protection.

**Shadow evaluation (W-OPS.11):** set `RuntimeRequest.metadata["harness_shadow_eval"]` to `{"scenario_id": "...", "passed": true, "score": 1.0}`; `RuntimeEngine` records `HarnessShadowEvalRecordedDiagV1` trace step and appends to `build/architecture_hardening/online_evaluation_observations.json`. After a release, export trends: `uv run python scripts/export_harness_shadow_eval_trend.py --release-id <id>` → `shadow_evaluation_trend_report.json` (snapshots in `evaluation_release_snapshots.json`).

**Release cycles (W-OPS.5):** after a gate-green harness release, run `uv run python scripts/record_harness_release_cycle.py --cycle-id <id> [--verify-gate]`. Evidence script reads `build/architecture_hardening/release_cycles.json` (or `W_OPS_RELEASE_CYCLES`).

| SLI | Target (harness lab) | Measurement |
|-----|----------------------|-------------|
| Trace completeness | ≥ 99% runs have `GET /debug/tasks/{id}/trace` payload | SQLite trace store + acceptance tests |
| Lab run success rate | ≥ 99% gate `agent_os` acceptance | `pytest tests/acceptance/agent_os -m gate` |
| Tool idempotency | Duplicate side-effect invoke returns cached result | `tests/unit/runtime/tools/test_idempotent_invoker.py` |
| Integration resilience | Circuit opens after repeated backend failures | `tests/unit/integrations/test_integration_circuit_breaker.py` |
| Cost per run (lab) | No unbounded token growth week-over-week | `GET /debug/tasks/{id}/metrics` + V-COST envelopes |

**Incident budget (rolling 30d):** ≤ 2 Sev-2 harness regressions; ≤ 1 unresolved gate red > 24h.

**Runtime event ops filters (Phase DX-5.7):** Every `RuntimeEventType` maps to an `ExecutionPhase` and a stable ops filter token (`ops:alert`, `ops:hitl`, `trace:step`, …). Source of truth: `intergrax.runtime.events.phase_coverage` (`EVENT_PHASE_COVERAGE`, `EVENT_OPS_FILTER_HINTS`). Canon table: [architecture §42.1.5](intergrax_runtime_architecture.md#4215-runtime-event-catalog-ops-filters). Gate: `test_all_runtime_event_types_have_ops_filter_hint`.

**Runbook stubs (owner: harness-platform):**

| Scenario | Action |
|----------|--------|
| Gate red after harness PR | Re-run `pytest -m gate`; check `check_plugin_catalog.py` and `phase_v_closeout_gate.py` |
| Provider outage | Open integration circuit; degrade tool to read-only paths; notify on-call |
| Policy false positive | Use HITL override path; file policy fragment fix in `RuntimePolicyBundle` |
| Stuck long-running task | Inspect checkpoint store; resume via scheduler or abort via debug API |

---

## Security surfaces (Phase U)

| Variable | Default | Effect |
|----------|---------|--------|
| `INTERGRAX_HARNESS_API_KEY` | unset (dev) | **Required** when `INTERGRAX_ENV=stage` or `prod`, or `LAB_STRICT_HARNESS=true` (W-OPS.7). When set, lab/debug/interaction/MCP routes require `X-Api-Key` or `Authorization: Bearer <key>` |
| `INTERGRAX_ENV` | `dev` | `stage` / `staging` / `prod` select API environment; non-dev requires harness API key |
| `LAB_INCLUDE_MCP` | `false` | MCP mount is opt-in |
| `LAB_STRICT_HARNESS` | `false` | Reference agents use `production_mode=True`, governance service, and `trace_db_path` on `RuntimeConfig` |
| `INTERGRAX_MODALITY_EXECUTION` | `in_process` | Set `celery` + `INTERGRAX_MODALITY_CELERY_BROKER_URL` for Tier-3 modality scale-out (W-OPS.12) |

Lab reference agents implement `HarnessReferenceAgent` + `UAEPAgent`; manifest bindings set `requires_uaep=True` (research agents excluded until UAEP migration).

---

## Policy bundle

`build_runtime_policy_bundle()` on lab registry build — typed `BudgetPolicy` / `PlanLoopPolicy` slots on `RuntimePolicyBundle`. Applied via `build_lab_agent_runtime_context()` (Phase U-Pol.1). Read order: architecture §42.11.5.

---

## Harness control plane (authoring)

Governance, policy, and observability are **composable control-plane layers** — configured via `ApplicationEnvironmentProfile`, `RuntimePolicyBundle`, hooks, and plugin entry points; enforced by Nexus on every run.

| Need | Where |
|------|--------|
| Full control-plane map (profiles, bundles, hooks, EP groups) | [`AGENT_CREATION_GUIDE.md` Appendix H](AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) |
| Operator policy read order | Architecture [§42.11.5](intergrax_runtime_architecture.md#42115-how-to-read-policy-for-a-run-operator) |
| Policy rule handler plugins (`intergrax.policy_rules`) | [`EXTENSION_AUTHOR_GUIDE.md` §10](EXTENSION_AUTHOR_GUIDE.md#10-policy-rule-handler-plugins-phase-dx-58) |
| Audit layers (policy §5, observability §21) | [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) |
| Observability wire-time closeout (§21) | [`AGENT_CREATION_GUIDE.md` Appendix Q](AGENT_CREATION_GUIDE.md#appendix-q--observability-control-plane-closeout) · [Phase OBS](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-obs--observability-control-plane-closeout) |
| Reliability wire-time closeout (§22) | [`AGENT_CREATION_GUIDE.md` Appendix R](AGENT_CREATION_GUIDE.md#appendix-r--reliability-control-plane-closeout) · [Phase REL](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-rel--reliability-control-plane-closeout) |
| Security wire-time closeout (§23) | [`AGENT_CREATION_GUIDE.md` Appendix S](AGENT_CREATION_GUIDE.md#appendix-s--security-control-plane-closeout) · [Phase SEC](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-sec--security-control-plane-closeout) |
| Cost wire-time closeout (§24) | [`AGENT_CREATION_GUIDE.md` Appendix T](AGENT_CREATION_GUIDE.md#appendix-t--cost-governance-control-plane-closeout) · [Phase COST](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-cost--cost-governance-control-plane-closeout) |

**Modularity:** swap observability backend via `IntegrationProfile.observability_backend`; add policy via YAML + EP handlers; enable V-SEC defenses via `ApplicationSecurityProfile` — without changing Tier-2 agent code.

**Orchestration:** graph execution, delegation, handoff, hooks — [`AGENT_CREATION_GUIDE.md` Appendix I](AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane). Wired via `orchestration_wiring.py` + `graph_spec_to_plan.py` (Phase ORCH **Done**). Multi-agent quick start: [Appendix C](AGENT_CREATION_GUIDE.md#appendix-c--multi-agent-graphs).

---

## Post-U continuation (Phase V)

Phase S/T/U established a production-configurable harness baseline.
Phase V architecture hardening and **Phase W-ML** harness contracts are **complete** in harness-only scope: modality tools (including `ml.batch_predict`), `VisionProfile`/`SpeechProfile`, `ModalityExecutionProfile` with `in_process` / `thread_pool` / `celery` offload (`INTERGRAX_MODALITY_EXECUTION`, optional `INTERGRAX_MODALITY_CELERY_BROKER_URL`), typed `ModalityInvocationCounters` (`media_bytes`, `tts_characters`, `ml_predictions`) surfaced on trace and run export, `wire_modality_extras()` Celery `message_bus` task registration, lab + optional legal modality wiring, Triton/HF adapters, skills `harness.vision_qa` and `harness.modality_smoke`, capability graph modality compatibility edges. Default continuation is **operational L3/L4 stability window**.

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
- `V-SEC.4` tenant isolation verification and security audit trail checks (runtime: `TenantSecurityMiddleware` on `BEFORE_TASK_INTAKE`; optional `TaskContext.metadata["resource_tenant_id"]` for resource-bound checks via `task_security_context.py`)
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
- `V-V6.3` CI closeout gate: `scripts/phase_v_closeout_gate.py` (`--enforce`, `--enforce-l4`; prints `l4_governance_passed` vs `l4_runtime_passed`)
- **W-ADAPT-5** adaptive verify closeout: `scripts/phase_w_adapt_closeout_gate.py` (`--enforce-l4-runtime`)
- **W-ADAPT-5** ops runbooks: [`runbook/adaptive/rollback_profile.md`](../runbook/adaptive/rollback_profile.md), [`approve_policy_learning.md`](../runbook/adaptive/approve_policy_learning.md), [`shadow_failure_triage.md`](../runbook/adaptive/shadow_failure_triage.md)
- **W-ADAPT-5** evidence artifacts: `build/adaptive_harness/verification_report.json`, `build/adaptive_harness/l4_runtime_evidence.json`

### Adaptive Harness Intelligence ops (W-ADAPT-7)

Lab and reference apps ship with `AdaptiveProfile(enabled=False, mode="observe")` by default. Enable closed-loop behavior only in controlled environments.

| Variable / flag | Purpose |
|-----------------|--------|
| `AdaptiveProfile.enabled` | Master switch for adaptive stores, executor, and signal collector |
| `AdaptiveProfile.mode` | `observe` \| `recommend` \| `shadow` \| `canary` \| `apply` |
| `AdaptiveProfile.signal_store_path` | SQLite path for harness outcome signals |
| `AdaptiveProfile.proposal_store_path` | SQLite path for adaptation engine runs |
| `AdaptiveProfile.debug_readonly_routes` | Mount `/debug/adaptive/signals` and `/debug/adaptive/proposals` on lab host |
| `INTERGRAX_BUSINESS_OUTCOME_WEBHOOK_SECRET` | HMAC secret for optional Tier-3 `business_outcome` webhook payloads |

Reports and closeout:

```bash
uv run python scripts/phase_w_adapt_report.py --patterns-output build/adaptive_harness/process_patterns.json
uv run python scripts/phase_w_adapt_closeout_gate.py --enforce-l4-runtime
```

Authoring guide: [`AGENT_CREATION_GUIDE.md` Appendix V](AGENT_CREATION_GUIDE.md#appendix-v--adaptive-harness-control-plane-closeout).

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
python scripts/check_legacy_modules_removed.py
python scripts/check_agent_skill_resolution.py
python scripts/check_harness_registry_resolution.py
python scripts/check_harness_capability_graph_wiring.py
python scripts/check_legacy_tool_plan_booleans.py
uv run pytest tests/unit/integrations/test_harness_lab_stable_stack.py -q
uv run pytest tests/unit/skills/test_harness_skill_bundle.py -q
uv run pytest tests/acceptance/agent_os/test_lab_application.py -m gate -q
uv run python scripts/phase_v_foundations_report.py
uv run python scripts/phase_v_capability_graph_guard.py
uv run python scripts/phase_v_governance_report.py
uv run python scripts/phase_v_closeout_gate.py --enforce --enforce-l4
uv run python scripts/phase_w_adapt_closeout_gate.py --enforce-l4-runtime
uv run python scripts/phase_w_ops_evidence.py
```

Operational L3 (after two stable release cycles):

```bash
set W_OPS_RELEASE_CYCLES=2
uv run python scripts/phase_w_ops_evidence.py --enforce
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
