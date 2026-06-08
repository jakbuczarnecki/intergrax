# Implementation Phases — Platform Quality

**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Phase W-OPS — Operational Harness Maturity (IDEAL L3 ops)

**Status:** **Done** (2026-06-06) — W-OPS.1–W-OPS.15 delivered including W-OPS.10 lab stack health probes; **operational L3** sign-off still requires `W_OPS_RELEASE_CYCLES>=2` (or `build/architecture_hardening/release_cycles.json`) via `phase_w_ops_evidence.py --enforce`.  
**Source:** Harness maturity audit (2026-06-02; conversation) · [IDEAL_HARNESS_AI_ARCHITECTURE.md](IDEAL_HARNESS_AI_ARCHITECTURE.md) §12.3–§12.4 · [guides/HARNESS_ENVIRONMENT.md](guides/HARNESS_ENVIRONMENT.md)  
**Prerequisites:** Phases **V**, **P-Ext**, **W-ML**, §4.1 **Done**.  
**Goal:** Close the gap between **L3 CI evidence** (`maturity_gate_evidence`, relaxed thresholds) and **L3 operational** (IDEAL critical areas Policy/Reliability/Observability ≥ 3 with release evidence).  
**Out of scope:** K.1, K.2, new product Tier-3 apps, domain/product skills (Band 3 · §6.3).

**Audit verdict (harness-only):** Intergrax is **L2+ scalable harness** with strong Tier-0 catalogs and Nexus §42; default implementation queue is **§6.1 + §6.2w**, not product agents.

#### W-OPS — Deliverables

| # | Deliverable | Status | Priority | Location / acceptance |
|---|-------------|--------|----------|------------------------|
| W-OPS.0 | Plan traceability from maturity audit | **Done** | — | This phase + §6.2w + doc model row |
| W-OPS.1 | **Side-effect idempotency** — `IdempotentToolInvoker` + `idempotency_key` on `ToolExecutionRequest` | **Done** | **Critical** | `runtime/tools/idempotent_invoker.py`; gate `test_idempotent_invoker.py` |
| W-OPS.2 | **Integration circuit breaker** — `IntegrationCircuitBreaker` in `integrations/_shared/` | **Done** | **Critical** | `IntegrationDependencyError`; `test_integration_circuit_breaker.py` |
| W-OPS.3 | **Reliability gate tests** — long-running scheduler / checkpoint in gate | **Done** | High | `test_long_running_scheduler_j4.py` (`pytest -m gate`) |
| W-OPS.4 | **SLO catalog + incident budget** — harness SLIs + runbook stubs | **Done** | **Critical** | `guides/HARNESS_ENVIRONMENT.md` § Harness SLO catalog |
| W-OPS.5 | **L3-ops evidence artifact** — distinct from V-V6 CI gate | **Done** | **Critical** | `phase_w_ops_evidence.py`; `record_harness_release_cycle.py`; `release_cycles.json` |
| W-OPS.6 | **`tenant_id` on execution path** — required on `RuntimeRequest`; trace/events scoped | **Done** | High | `runtime/nexus/engine/runtime.py`; `RuntimeState.tenant_id` |
| W-OPS.7 | **Mandatory harness auth** — stage/prod/strict require `INTERGRAX_HARNESS_API_KEY` | **Done** | High | `LabApplicationSettings.requires_harness_api_key`; `test_lab_harness_api_key_required.py` |
| W-OPS.8 | **`harness.*` skill expansion** — `harness.reliability_smoke`, `harness.policy_smoke` | **Done** | Medium | `skills/providers/harness/manifests.py` |
| W-OPS.9 | **`requires_skills` adoption** — `harness.stack_demo` | **Done** | Medium | `test_harness_requires_skills_demo.py` |
| W-OPS.10 | **Harness lab stack health** — per-slug probes + circuit breaker | **Done** | Medium | `health_check_catalog_slugs`, `harness_lab_health.py`; `test_harness_lab_health.py` |
| W-OPS.11 | **Online evaluation path** — shadow observations → evaluation trends | **Done** | Medium | `online_evaluation_trend.py`, `export_harness_shadow_eval_trend.py`; file registry + RuntimeEngine hook |
| W-OPS.12 | **W-ML Celery scale-out (optional)** — env-driven via `wire_modality_extras` | **Done** | Low | `INTERGRAX_MODALITY_EXECUTION=celery`; documented in HARNESS_ENVIRONMENT |
| W-OPS.13 | **ToolsAgent removal roadmap** — CI blocks new imports; module frozen | **Done** | Low | `check_tools_agent_imports.py`, `check_tools_agent_run.py` |
| W-OPS.14 | **Typed Tier-3 wiring** — `load_callable` uses module namespace (no `getattr`) | **Done** | Low | `applications/_shared/wiring.py` |
| W-OPS.15 | **Architecture metrics enforcement (phased)** — tightened V-V6 thresholds | **Done** | Low | `maturity_gate_evidence.collect_harness_governance_signals` |

#### W-OPS — Execution waves (dependency order)

```text
Wave W-OPS-0 (governance):  W-OPS.0  — Done (audit → plan)
Wave W-OPS-P0 (critical):   W-OPS.1 → W-OPS.2 → W-OPS.3 → W-OPS.4 → W-OPS.5 → W-OPS.6 → W-OPS.7
Wave W-OPS-P1 (extend):     W-OPS.8 → W-OPS.9 → W-OPS.10 → W-OPS.11 → W-OPS.12 (optional)
Wave W-OPS-P2 (hygiene):    W-OPS.13 → W-OPS.14 → W-OPS.15
```

**IDEAL §12.3 gate:** Do not declare **operational L3** until W-OPS-P0 is **Done** and W-OPS.5 records **two consecutive release cycles** within SLO/incident budget (W-OPS.4).

**Delivery rule:** One **W-OPS.\*** ID per PR → update this table + paydown log → `pytest -m gate` + harness audit scripts (§6.1).

#### W-OPS — Paydown log

| Date | W-OPS ID | Summary |
|------|----------|---------|
| 2026-06-02 | W-OPS.0 | Maturity audit → Phase W-OPS + §6.2w execution order in implementation plan |
| 2026-06-06 | W-OPS.1–W-OPS.15 | Circuit breaker, idempotency gate, SLO docs, ops evidence script, staging API key, harness skills, online eval, wiring/metrics |
| 2026-06-02 | OPS-L3.1 | `phase_w_ops_evidence.py` Windows pytest argv + shadow trend probe; `--enforce` green |
| 2026-06-02 | REG / §6.1 | `doctor --ci` green: research `ToolEnablementProfile` protocol; lab factory via `bootstrap_lab_integration_wiring` |
| 2026-06-03 | W-OPS.10–W-OPS.11 | Lab stack health by catalog slug; shadow eval wired in `RuntimeEngine`; CI `phase_w_ops_evidence.py`; gate **470** |
| 2026-06-03 | W-OPS.5/11 | File-backed shadow eval registry; `record_harness_release_cycle.py`; extended ops evidence checks |
| 2026-06-03 | §6.1 / N.9 | Product scaffold `legal_product()` manifest + catalog bootstrap; gate **470** |
| 2026-06-03 | W-OPS.11 | Shadow eval trend export + `--verify-gate` on release cycle recorder |
| — | — | *(append row per merged PR)* |

---

## Phase FAUDIT-32 — Full architecture audit closeout

**Status:** **Done** (2026-06-06) — 32-layer audit (`scope: C`) + **23/23 FAUDIT remediation** implemented → [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed)  
**Source:** [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §8  
**Traceability:** **Appendix M** (layer scorecard + gap → FAUDIT ID matrix)

**Audit verdict (2026-06-06, pre-remediation snapshot):** Harness **control-plane wiring closeouts** (ORCH, TS, INT, RAG, CTX, PE, AS, REG, CG, OBS, REL, SEC, COST, EVAL, W-ADAPT, M-LLM-R) are **Done** as documented — but **closeout ≠ full layer maturity**. Per-layer inspection at audit time showed **12/32 layers at L3+**, **19/32 at L2**, **1 Critical** tier-boundary violation, **~20 High** residuals — all routed to **FAUDIT.\*** and **closed** via [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed) + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed).

**Post-remediation (2026-06-06):** **0 Critical** open; tier CI gate green; **23/23 FAUDIT** + follow-up Done. Layer maturity uplift (L2→L3 depth) remains incremental maintenance — see Appendix M.

**Gate evidence (verify step):** `uv run pytest -m gate -q` → **901 passed**; `check_harness_no_getattr.py`, `check_intergrax_no_applications_imports.py`, `check_harness_prompt_golden_catalog.py`, `check_agents_lifecycle_metadata.py` → **OK**.

### FAUDIT-32 — Layer scorecard (summary)

| # | Layer | Score | Crit | High | Plan accurate? |
|---|-------|-------|------|------|----------------|
| 1 | Strategic Harness Model | L3 | 0 | 0 | Yes |
| 2 | Tier Model and Dependency Boundaries | L2 | **1** | 1 | **Partial** |
| 3 | Interface and Task Intake | L2 | 0 | 2 | Partial |
| 4 | Identity, Trust and Tenancy | L2 | 0 | 2 | Partial |
| 5 | Policy and Governance | L3 | 0 | 2 | Partial |
| 6 | LLM and Model Adapter Layer | L3 | 0 | 1 | Yes |
| 7 | Reasoning, Planning and Cognition | L2 | 0 | 1 | Partial |
| 8 | Execution Runtime and Agent OS | L3 | 0 | 0 | Yes |
| 9 | Orchestration, Scheduler and Execution Graph | L3 | 0 | 1 | Partial |
| 10 | Subagents and Multi-Agent Coordination | L2 | 0 | 2 | Partial |
| 11 | Tool Layer | L3 | 0 | 1 | Yes |
| 12 | Skill Layer | L3 | 0 | 0 | Yes |
| 13 | Integration Layer | L3 | 0 | 0 | Yes |
| 14 | RAG and Retrieval Layer | L3 | 0 | 0 | Yes |
| 15 | Memory Layer | L2 | 0 | 2 | Partial |
| 16 | Context Engineering Layer | L3 | 0 | 0 | Yes |
| 17 | Prompt Engineering and Prompt Registry | L2 | 0 | 1 | **No** |
| 18 | Agent Assembly and Agent Contracts | L2 | 0 | 1 | Yes |
| 19 | Registry Architecture | L2 | 0 | 2 | **No** |
| 20 | Capability Graph Architecture | L2 | 0 | 2 | **No** |
| 21 | Observability and Telemetry | L2 | 0 | 2 | **No** |
| 22 | Error Handling and Reliability | L2 | 0 | 1 | **No** |
| 23 | Security and Data Governance | L2 | 0 | 2 | **No** |
| 24 | Cost and Resource Governance | L2 | 0 | 1 | **No** |
| 25 | Evaluation and Benchmarking | L2 | 0 | 1 | **No** |
| 26 | Testing, CI and Architecture Gates | L3 | 0 | 0 | Yes |
| 27 | Developer Experience, Scaffold and Lab | L3 | 0 | 1 | Yes |
| 28 | Product Environment and Tier-3 Applications | L3 | 0 | 2 | Partial |
| 29 | Modality, Vision, Audio and Dedicated ML | L3 | 0 | 1 | Yes |
| 30 | Operational Excellence and SLOs | L3 | 0 | 2 | Partial |
| 31 | Agent Lifecycle Governance | L2 | 0 | 2 | Partial |
| 32 | Architecture Governance and Documentation Loop | L3 | 0 | 1 | Yes |

**Plan accuracy note:** Rows marked **No** or **Partial** mean the phase closeout register claims **Done** for **wiring/bridge** work, but FAUDIT found **High** gaps vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` / `INTEGRAX_HARNESS_AUDIT_MAP.md` §8 — tracked as **FAUDIT.\*** residuals, not reopening closed closeout phases.

### FAUDIT-32 — Remediation register (implementation queue → §6.1ah)

| ID | Layer | Gap | Severity | Module / acceptance |
|----|-------|-----|----------|-------------------|
| FAUDIT-TIER.1 | §2 | Tier-0 imports `applications/*` in `capability_graph_applications.py` | **Critical** | Move manifest catalog to Tier-3 injection or static metadata; zero `from applications` under `intergrax/` |
| FAUDIT-TIER.2 | §2 | No CI gate for `intergrax/` → `applications/` imports | High | `scripts/check_intergrax_no_applications_imports.py` in §6.1 |
| FAUDIT-INTAKE.1 | §3 | No canonical `TaskEnvelope`; `Task` + `RuntimeRequest` split | High | Typed envelope alias or consolidation; plan W-OPS.6 naming sync |
| FAUDIT-INTAKE.2 | §3 | Worker≡HTTP intake parity test matrix incomplete | High | Acceptance test: CLI/worker/HTTP same `Task` shape |
| FAUDIT-ID.1 | §4 | No user/service/agent identity distinction | High | Identity contracts + propagation to delegation |
| FAUDIT-ID.2 | §4 | `DelegationSpec` lacks permission scope audit | High | Scope field + trace on child runs |
| FAUDIT-POL.1 | §5 | No pre-LLM / pre-output policy hooks in runtime | High | PolicyEngine extension points documented + wired |
| FAUDIT-LLM.1 | §6 | No policy-driven model routing / fallback chain | High | Router module or AdaptiveProfile integration |
| FAUDIT-COG.1 | §7 | No universal `DecisionRecord` per UAEP step | High | Typed decision artifact + trace event |
| FAUDIT-ORCH.1 | §9 | No graph backpressure beyond parallel cap | High | Queue depth / shed policy in `GraphExecutor` |
| FAUDIT-SUB.1 | §10 | No formal `SubtaskContract`; `inherit_tool_policy=True` default | High | Contract type + safer delegation defaults |
| FAUDIT-MEM.1 | §15 | Entity graph memory absent; STM retention partial | High | Route to MEM-9.* / new MEM row if scoped |
| FAUDIT-PE.1 | §17 | No golden prompt content regression in CI | High | Golden YAML fixtures + gate test |
| FAUDIT-REG.1 | §19 | `HarnessRegistrySnapshot` omits agents + eval registry | High | Extend snapshot + assembly resolver |
| FAUDIT-CG.1 | §20 | Capability graph seed skips `prompt:*` nodes | High | `_seed_node_ids()` parity |
| FAUDIT-CG.2 | §20 | Blast-radius not enforced at release | High | `phase_v_capability_graph_guard.py` impact check |
| FAUDIT-OBS.1 | §21 | `RuntimeEventType` missing `LLM_CALL` / `POLICY_DECISION` | High | Canon event catalog + bridge from trace |
| FAUDIT-REL.1 | §22 | Shallow error taxonomy (`VALIDATION_ERROR` only + 2) | High | Expand classifier per AUDIT_MAP §22 |
| FAUDIT-SEC.1 | §23 | No `DataClassification` model | High | Security profile + enforcement hooks |
| FAUDIT-COST.1 | §24 | Per-tenant cost attribution not mandatory in NexusLoop | High | Budget gate on main path |
| FAUDIT-EVAL.1 | §25 | `require_baseline_for_release` not CI-enforced | High | `phase_v_closeout_gate.py` eval baseline check |
| FAUDIT-ALG.1 | §31 | Lifecycle states ≠ AUDIT_MAP catalog; weak agent adoption | High | Align or document ADR; scaffold defaults |
| FAUDIT-OPS.1 | §30 | `release_cycles.json` not in repo; W-OPS.5 artifact policy unclear | High | Document committed vs generated artifact |

**Delivery rule:** One **FAUDIT.\*** ID per PR → update §6.1ah + Appendix M paydown log → gate green.

**Explicitly out of scope (audit-and-fix):** source code or test changes during this audit pass; K.1/K.2; new product Tier-3 apps.

---

