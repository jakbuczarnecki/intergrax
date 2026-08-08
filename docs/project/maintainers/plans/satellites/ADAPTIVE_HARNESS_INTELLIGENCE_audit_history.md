# ADAPTIVE_HARNESS_INTELLIGENCE — audit history + LC closeout

**Parent hub:** [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](../ADAPTIVE_HARNESS_INTELLIGENCE.md)

## Phase W-ADAPT — Adaptive Harness Intelligence (L4 runtime)

**Status:** **Done** (2026-06-02) — **70/70 Done** (Wave W-ADAPT-0 through Wave W-ADAPT-7 complete)  
**Architecture spec:** [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) (AHIA) · runtime canon [§54](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md#54-adaptive-harness-intelligence-ahi--l4-runtime-addendum) · IDEAL [§25](IDEAL_HARNESS_AI_ARCHITECTURE.md#25-adaptive-harness-layer)  
**Prerequisites:** Phase **V** **Done** · Phase **V-REM** **Done** · Phase **W-OPS** **Done** · Phase **H-APP** **Done** · Phases **EVAL**, **COST**, **CG** closeouts **Done** (signal sources + governance envelopes exist)  
**Goal:** Close the gap between **L4 governance contracts** (`adaptive_governance.py`, `phase_v_closeout_gate.py --enforce-l4`) and **L4 adaptive runtime** — governed closed loop: **observe → propose → gate → shadow/canary → apply → verify → rollback**  
**Priority ladder:** **Band 2y** (§4.0) — **closed**; default queue = **§6.1 maintenance**  
**Execution order:** [§6.2ac](#62ac-phase-w-adapt-execution-order-band-2y--closed) · queue: [§6.1t](#61t-harness-implementation-queue--adaptive-harness-intelligence-closed)  
**Traceability:** [Appendix K](#appendix-k--adaptive-harness-intelligence-traceability-phase-w-adapt)

**Delivery rule:** One **W-ADAPT.\*** ID per PR → update master table + Appendix K + paydown log → `pytest -m gate` green → run `phase_w_adapt_report.py` when touching signal/proposal paths.

**Principle:** **evolve, not rewrite** · reuse Phase V contracts · **no classical RL** (ADR-ADAPT-001) · Tier-1 **domain-agnostic** · adaptation **async** (never block Nexus hot path) · **PolicyEngine** never bypassed.

**Out of scope:** K.1/K.2 business agents · deep RL / neural policy training · foundation model fine-tuning inside Nexus · autonomous prompt string mutation without Prompt Registry · second trace/eval/policy stacks · Mem0-like product memory (MEM-8 RFC only) · integration marketplace UI.

**L4 distinction (normative):**

| Gate | What it proves | Artifact |
|------|----------------|----------|
| **Governance L4** (Phase V — **Done**) | Adaptive loop **envelopes** + sample proposals pass rules | `phase_v_closeout_gate.py --enforce-l4` |
| **Runtime L4** (Phase W-ADAPT — **target**) | Closed loop **measurably improves** utility U on golden scenarios | `phase_w_adapt_closeout_gate.py --enforce-l4-runtime` |

```text
Wave W-ADAPT-0 (planning):       5 tasks  — RFC sync, ADR, package scaffold
Wave W-ADAPT-1 (observe L4-O):  12 tasks — SignalCollector, utility, report
Wave W-ADAPT-2 (recommend L4-R): 12 tasks — AdaptationEngine, proposals, scheduler
Wave W-ADAPT-3 (shadow L4-S):    7 tasks — ProfileVersionStore, shadow executor
Wave W-ADAPT-4 (apply L4-A):      10 tasks — canary, apply, rollback, runtime events
Wave W-ADAPT-5 (verify L4-V):     12 tasks — VerificationLoop, runtime L4 closeout, runbooks
Wave W-ADAPT-6 (patterns):         5 tasks — ProcessPatternMiner (after W-ADAPT-5 core)
Wave W-ADAPT-7 (Tier-3 + docs):    7 tasks — AdaptiveProfile wiring, author guide, acceptance
Total: 70 deliverables
```

### W-ADAPT — Traceability (AHIA section → task IDs)

| AHIA § | Topic | Task IDs |
|--------|--------|----------|
| §5–§6 | Audit gap / missing components | W-ADAPT-0.*, W-ADAPT-1.1–1.3 |
| §9.1 | SignalCollector | W-ADAPT-1.4–1.10 |
| §10 | HarnessOutcomeSignal + utility U | W-ADAPT-1.1, W-ADAPT-1.8 |
| §9.2 | AdaptationEngine sub-engines | W-ADAPT-2.1–2.7 |
| §9.3–9.4 | ProposalBuilder + governance gate | W-ADAPT-2.6–2.9 |
| §9.5 | AdaptationExecutor | W-ADAPT-3.3, W-ADAPT-4.2–4.5 |
| §9.6 | VerificationLoop | W-ADAPT-5.1–2.5 |
| §9.7 | ProcessPatternMiner | W-ADAPT-6.* |
| §9.8 | ProfileVersionStore | W-ADAPT-3.1–3.2 |
| §9.9 | AdaptationScheduler | W-ADAPT-2.12, W-ADAPT-6.5 |
| §11 | Four AdaptiveLoopKind loops | W-ADAPT-2.2–2.5 |
| §12 | Lifecycle modes L4-O→L4-V | W-ADAPT-1.* … W-ADAPT-5.* |
| §14 | Nexus / eval / capability graph integration | W-ADAPT-1.9, W-ADAPT-2.8, W-ADAPT-4.8 |
| §14.5 | AdaptiveProfile (Tier-3) | W-ADAPT-4.1, W-ADAPT-7.* |
| §16 | Security, HITL, audit events | W-ADAPT-4.6–4.7, W-ADAPT-5.5 |
| §20 | KPIs + L4 runtime evidence | W-ADAPT-5.6–5.12 |
| §21 | Runbooks | W-ADAPT-5.9 |
| Appendix A | Reuse existing modules | W-ADAPT-2.10, W-ADAPT-3.4, W-ADAPT-5.3–5.4 |

### W-ADAPT — Master deliverables register (70 tasks)

#### Wave W-ADAPT-0 — Planning and package scaffold

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| W-ADAPT-0.1 | **Plan + canon sync** — Phase W-ADAPT section, §4.0 Band 2y, §6.1t, §6.2ac, Appendix K; AHIA ↔ plan cross-links | **Done** | **Critical** | This section + AHIA Appendix B |
| W-ADAPT-0.2 | **`docs/project/technical/adr/entries/2026-06-05/ADR-ADAPT-001.md`** — Adaptive Harness Intelligence over classical RL (AHIA Appendix C) | **Done** | High | ADR accepted; linked from AHIA + canon §54 |
| W-ADAPT-0.3 | **Package scaffold** — `intergrax/runtime/adaptive/` with `contracts.py`, `__init__.py`, re-exports | **Done** | **Critical** | Importable; no runtime side effects |
| W-ADAPT-0.4 | **Extend `runtime/architecture/__init__.py`** — export adaptive contracts without duplicating `adaptive_governance.py` | **Done** | Medium | Unit smoke import |
| W-ADAPT-0.5 | **Gate test stub** — `tests/unit/runtime/adaptive/test_package_imports.py` | **Done** | Medium | `pytest -m gate` green |

#### Wave W-ADAPT-1 — Observe (L4-O)

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| W-ADAPT-1.1 | **`HarnessOutcomeSignal`** + **`UtilityWeights`** Pydantic contracts | **Done** | **Critical** | Schema validated; AHIA §10.1 |
| W-ADAPT-1.2 | **`ProfileVersionRecord`**, **`ProfileVersionDraft`**, **`ProcessPatternProposal`** contract stubs | **Done** | High | Validators; status enum |
| W-ADAPT-1.3 | **`SignalStore`** — SQLite persistence under `build/adaptive_harness/` | **Done** | **Critical** | CRUD + list by tenant/window |
| W-ADAPT-1.4 | **`SignalCollector`** — integrate `export_run_metrics()` / `RunMetricsExport` | **Done** | **Critical** | behavioral + cost fields populated |
| W-ADAPT-1.5 | **`SignalCollector`** — integrate `ExecutionGuard` + `HistoryAwareEvaluator` regression flags | **Done** | High | `regression_flags` on signal |
| W-ADAPT-1.6 | **`SignalCollector`** — integrate online/shadow eval (`OnlineEvaluationRegistry`) | **Done** | High | `quality_score`, `eval_mode` |
| W-ADAPT-1.7 | **`SignalCollector`** — integrate cost budget normalization (`cost_budget.py`) | **Done** | High | `cost_normalized` |
| W-ADAPT-1.8 | **`compute_utility()`** — AHIA §10.2 formula + default weights | **Done** | **Critical** | Unit tests for weight boundaries |
| W-ADAPT-1.9 | **`SignalCollector`** — HITL intervention counters from task/HITL runtime | **Done** | Medium | `hitl_interventions` |
| W-ADAPT-1.10 | **Nexus hook** — emit signal on task completion (`task_finisher` / lifecycle bridge) | **Done** | **Critical** | ≥1 signal per completed Nexus task in integration test |
| W-ADAPT-1.11 | **AgentEngine hook** — optional signal path for non-Nexus runs (parity with W-OPS shadow) | **Done** | Medium | Lab runtime records signal |
| W-ADAPT-1.12 | **`scripts/release/phase_w_adapt_report.py`** — signal trends + utility histograms | **Done** | High | JSON under `build/adaptive_harness/signal_trends.json` |

#### Wave W-ADAPT-2 — Recommend (L4-R)

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| W-ADAPT-2.1 | **`BanditStateStore`** — per `(tenant_id, task_class, arm_id)` Thompson state | **Done** | **Critical** | Unit tests; partitioned by tenant |
| W-ADAPT-2.2 | **`RoutingTuningEngine`** — `ROUTING_TUNING` proposals (LLM route + RAG tier arms) | **Done** | **Critical** | Respects `max_delta_percent`; cooldown |
| W-ADAPT-2.3 | **`ExecutionStrategyEngine`** — `EXECUTION_STRATEGY_TUNING` from step/retry/parallel metrics | **Done** | High | Uses `HistoryAwareEvaluator` patterns |
| W-ADAPT-2.4 | **`PolicyLearningEngine`** — `POLICY_LEARNING` proposals (tool deny deltas); **no apply** | **Done** | High | Requires `human_approver_id` in proposal |
| W-ADAPT-2.5 | **`EvaluationFeedbackEngine`** — `EVALUATION_FEEDBACK`; observe-only re-eval triggers | **Done** | Medium | Links to `evaluation_registry_trends` |
| W-ADAPT-2.6 | **`ProposalBuilder`** — wraps `AdaptiveLoopProposal` + attaches `ProfileVersionDraft` | **Done** | **Critical** | Passes `evaluate_bounded_adaptive_loop()` |
| W-ADAPT-2.7 | **`AdaptationEngine` facade** — ranks proposals from sub-engines | **Done** | **Critical** | Unit tests with synthetic signals |
| W-ADAPT-2.8 | **Governance gate stage 2** — `evaluate_capability_graph_compatibility()` pre-check | **Done** | High | Rejects incompatible skill/policy edges |
| W-ADAPT-2.9 | **Governance gate stage 4** — golden scenario smoke before shadow allocation | **Done** | High | Uses eval assets / NexusEvalRunner |
| W-ADAPT-2.10 | **Cost anomaly → proposal** — wire `cost_optimization.py` into `AdaptationEngine` | **Done** | Medium | Anomalies produce ROUTING/COST proposals |
| W-ADAPT-2.11 | **Extend `phase_w_adapt_report.py`** — proposal log + gate results | **Done** | High | `build/adaptive_harness/proposals.json` |
| W-ADAPT-2.12 | **`AdaptationScheduler` skeleton** — hourly `run_adaptation_engine` (recommend-only) | **Done** | High | No executor calls in this wave |

#### Wave W-ADAPT-3 — Shadow (L4-S)

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| W-ADAPT-3.1 | **`ProfileVersionStore`** — CRUD + immutable payloads | **Done** | **Critical** | SQLite; gitignored path |
| W-ADAPT-3.2 | **Profile version lifecycle** — `draft → shadow → canary → active → retired` | **Done** | **Critical** | State machine tests |
| W-ADAPT-3.3 | **`AdaptationExecutor.shadow()`** — allocate candidate version for shadow runs | **Done** | **Critical** | Trace tag `candidate_profile_version_id` |
| W-ADAPT-3.4 | **Extend `RuntimeArchitectureGovernanceBridge`** — candidate version in shadow observation | **Done** | High | Extends W-OPS.11 hook |
| W-ADAPT-3.5 | **`ProfilePromotionEvidence`** — mirror `agent_promotion.py` checklist for profiles | **Done** | Medium | evaluation + rollback plan refs |
| W-ADAPT-3.6 | **Integration test** — shadow run records observation with candidate version | **Done** | High | `tests/integration/runtime/adaptive/` |
| W-ADAPT-3.7 | **`QueryRouter` override** — load RAG tier weights from active/candidate profile | **Done** | Medium | Unit test per profile version |

#### Wave W-ADAPT-4 — Apply (L4-A)

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| W-ADAPT-4.1 | **`AdaptiveProfile`** on `ApplicationEnvironmentProfile` — mode, weights, loops, canary | **Done** | **Critical** | Default `enabled=False`, `mode=observe` |
| W-ADAPT-4.2 | **`wire_adaptive_profile()`** + **`nexus_factory`** — resolve active profile version pointers | **Done** | **Critical** | Lab app smoke |
| W-ADAPT-4.3 | **Canary traffic switch** — tenant allowlist + `canary_traffic_percent` | **Done** | High | Only allowlisted tenants see candidate |
| W-ADAPT-4.4 | **`AdaptationExecutor.apply()`** — atomic active pointer swap | **Done** | **Critical** | Rollback pointer preserved |
| W-ADAPT-4.5 | **`AdaptationExecutor.rollback()`** — restore previous pointer | **Done** | **Critical** | <5 min in drill test |
| W-ADAPT-4.6 | **HITL approval workflow** — `POLICY_LEARNING` blocked without approver confirmation | **Done** | **Critical** | Security test: 0 unapproved applies |
| W-ADAPT-4.7 | **`ADAPTIVE_*` RuntimeEvent types** — signal, proposal, apply, rollback | **Done** | High | Events in trace export |
| W-ADAPT-4.8 | **Extend governance bridge** — `submit_proposal()`, `apply_approved()` | **Done** | High | Typed; audit trail |
| W-ADAPT-4.9 | **Policy fragment versioning** — `RuntimePolicyBundle` slices via ProfileVersionStore | **Done** | High | PolicyEngine loads version id |
| W-ADAPT-4.10 | **Orchestration + RAG profile resolution** — versioned `OrchestrationProfile` / `RagProfile` | **Done** | High | `build_nexus_loop_from_environment` reads store |

#### Wave W-ADAPT-5 — Verify (L4-V)

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| W-ADAPT-5.1 | **`VerificationLoop`** — compare candidate vs baseline utility trends | **Done** | **Critical** | AHIA §9.6 checks 1–4 |
| W-ADAPT-5.2 | **Auto-rollback** on verification failure | **Done** | **Critical** | Invokes W-ADAPT-4.5; blocks loop kind |
| W-ADAPT-5.3 | **Eval registry integration** — `evaluation_registry_trends.py` in verify path | **Done** | High | Release comparison report |
| W-ADAPT-5.4 | **ExecutionGuard regression rate** — verify window threshold | **Done** | High | No spike vs baseline |
| W-ADAPT-5.5 | **Cost + adversarial checks** in verify — budget + prompt/tool/retrieval suites | **Done** | High | V-SEC suites still green |
| W-ADAPT-5.6 | **`scripts/release/phase_w_adapt_closeout_gate.py`** — `--enforce-l4-runtime` | **Done** | **Critical** | CI optional then required |
| W-ADAPT-5.7 | **`maturity_gate_evidence.py`** — `runtime_l4_closed_loop_passed` input | **Done** | High | Distinct from governance L4 |
| W-ADAPT-5.8 | **Extend `phase_v_closeout_gate.py`** — label governance-L4 vs runtime-L4 | **Done** | Medium | Docs in AHIA §20.2 |
| W-ADAPT-5.9 | **Runbooks** — `runbook/adaptive/rollback_profile.md`, `approve_policy_learning.md`, `shadow_failure_triage.md` | **Done** | Medium | Linked from HARNESS_ENVIRONMENT |
| W-ADAPT-5.10 | **Rollback drill acceptance test** | **Done** | High | Documented + automated smoke |
| W-ADAPT-5.11 | **`l4_runtime_evidence.json` generator** — 30-day utility improvement artifact | **Done** | **Critical** | AHIA §20.3 path |
| W-ADAPT-5.12 | **Scheduler: continuous verify** on active canaries | **Done** | High | W-ADAPT-2.12 extended |

#### Wave W-ADAPT-6 — Process pattern intelligence

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| W-ADAPT-6.1 | **`ProcessPatternMiner`** — sequence mining on trace events | **Done** | High | PrefixSpan or n-gram v1 |
| W-ADAPT-6.2 | **Trace reader** — load sequences from persisted runs / SQLite trace store | **Done** | High | Tenant-scoped |
| W-ADAPT-6.3 | **Pattern report + human review queue** in `phase_w_adapt_report.py` | **Done** | Medium | `ProcessPatternProposal` export |
| W-ADAPT-6.4 | **Optional skill stub generator** — scaffold manifest draft (no auto-register) | **Done** | Low | Output file only; human merges |
| W-ADAPT-6.5 | **Daily scheduler job** — `run_pattern_miner` | **Done** | Medium | Cron via AdaptationScheduler |

#### Wave W-ADAPT-7 — Tier-3 wiring, docs, acceptance

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| W-ADAPT-7.1 | **Default `AdaptiveProfile`** on `lab_application` + reference apps | **Done** | High | Lab: `enabled=True` observe (`LAB_ADAPTIVE_OBSERVE`); product refs: `enabled=False` |
| W-ADAPT-7.2 | **`BusinessOutcomeWebhook` contract** — optional Tier-3 signal for `business_outcome` | **Done** | Medium | Signed payload validation |
| W-ADAPT-7.3 | **`guides/AGENT_CREATION_GUIDE.md` Appendix V** — Adaptive Harness authoring | **Done** | High | Control plane map |
| W-ADAPT-7.4 | **`guides/HARNESS_ENVIRONMENT.md`** — adaptive ops section + env vars | **Done** | Medium | Lab enable observe mode docs |
| W-ADAPT-7.5 | **Lab debug routes** (optional) — list proposals / signals read-only | **Done** | Low | Behind lab profile flag |
| W-ADAPT-7.6 | **Acceptance test** — end-to-end observe → recommend (no apply) | **Done** | High | `tests/acceptance/adaptive/` |
| W-ADAPT-7.7 | **Docs sync** — README, docs/README, Appendix H row for IDEAL §25 runtime | **Done** | Medium | Zero stale "out of scope L4" |

### W-ADAPT — Execution matrix (dependencies)

```text
W-ADAPT-0 ──► W-ADAPT-1 ──► W-ADAPT-2 ──► W-ADAPT-3 ──► W-ADAPT-4 ──► W-ADAPT-5
                  │                                              │
                  └──────────────────► W-ADAPT-6 (after W-ADAPT-5.1)
W-ADAPT-4.1 ──► W-ADAPT-7 (parallel after W-ADAPT-4.1)
W-ADAPT-5 ──► W-ADAPT-7.6 (full E2E acceptance)
```

**Critical rules:**

- W-ADAPT-1 **must** complete before any `AdaptationExecutor.apply` code (W-ADAPT-4.4).
- W-ADAPT-2 **must** stay recommend-only until W-ADAPT-3 shadow path is green.
- W-ADAPT-6 **must not** start until W-ADAPT-5.1 verification core exists (AHIA R7).
- Every PR: `pytest -m gate` + existing Phase V scripts unchanged green.

### W-ADAPT — KPI thresholds (runtime L4)

| KPI | Target | Verified by |
|-----|--------|-------------|
| Signal coverage | ≥ **95%** completed runs emit signal | W-ADAPT-1.10 + report |
| Shadow beat baseline | ≥ **60%** candidates beat baseline U | W-ADAPT-5.1 |
| Apply rollback rate | < **10%** of applies | W-ADAPT-5.2 metrics |
| Golden utility improvement | ≥ **10%** vs static baseline | W-ADAPT-5.11 |
| Unapproved policy learning applies | **0** | W-ADAPT-4.6 audit |
| Mean rollback time | < **5 minutes** | W-ADAPT-5.10 drill |

**Runtime L4 sign-off requires:** W-ADAPT-5.6 `--enforce-l4-runtime` green + W-ADAPT-5.11 artifact showing 30-day window on ≥ **3** golden scenarios.

### W-ADAPT — Suggested PR order

```text
W-ADAPT-0.2 → 0.3 → 0.4 → 0.5
→ 1.1 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 → 1.8 → 1.9 → 1.10 → 1.12
→ 2.1 → 2.2 → 2.3 → 2.6 → 2.7 → 2.8 → 2.11 → 2.12
→ 3.1 → 3.2 → 3.3 → 3.4 → 3.6
→ 4.1 → 4.2 → 4.4 → 4.5 → 4.6 → 4.7 → 4.10
→ 5.1 → 5.2 → 5.6 → 5.7 → 5.11 → 5.9 → 5.10
→ 6.1 → 6.2 → 6.3 → 6.5
→ 7.1 → 7.3 → 7.4 → 7.6 → 7.7
(remaining IDs parallelize within wave constraints)
```

### W-ADAPT — Paydown log

| Date | W-ADAPT ID | Summary |
|------|------------|---------|
| 2026-06-05 | W-ADAPT-0.1 | AHIA RFC + canon §54 + README; Phase W-ADAPT register opened in plan |
| — | — | *(append row per merged PR)* |

---

## Phase AHI-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates W-ADAPT 70/70 + AUDIT-IDEAL-AHI.*; no open P0/P1  
**Goal:** Formal Full Harness LC closeout — gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-LC-S1 | **Re-audit** — W-ADAPT register + L4 verdict | **Done** | High | No P0/P1 |
| AHI-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| AHI-LC-S3 | **Gate verification** | **Done** | High | 75 adaptive tests · `phase_w_adapt_closeout_gate` |
| AHI-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** L4 adaptive thresholds product-gated · foundation model training out of scope

### 6.1av Harness implementation queue — Adaptive harness intelligence audit maintenance (planned)

**Source:** Layer 19 audit (2026-06-18) — `ADAPTIVE_HARNESS_INTELLIGENCE` L4 · [`../audit_results/2026-06-18/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../audit_results/2026-06-18/ADAPTIVE_HARNESS_INTELLIGENCE.md)  
**Priority ladder:** **Band 1** (§6.1) — product-gated L4 depth; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **AHI-MAINT-01** | Process | P4 | **Done** | L4 adaptive thresholds — product-gated promotion criteria + evidence bundle | Explicit product decision gate documented in architecture |
| 2 | **AHI-MAINT-02** | Cross-ref | P2 | **Done** | AUDIT-IDEAL-6.2 live routing — cross-ref [`LLM-MAINT-02`](LLM_ADAPTERS.md#61av-harness-implementation-queue--llm-adapters-audit-maintenance-planned) / M-LLM-X.5 | AHI canon documents LLM owner |
| 3 | **AHI-MAINT-03** | Test/Ops | P3 | **Done** | Production adaptive evidence — reference host runs populating `signal_trends.json` | `phase_w_adapt_report.py --fixture` non-zero CI signals |
| 4 | **AHI-MAINT-04** | Docs | P3 | **Done** | L4 **Frozen** items index — GAP-CTX-12, M-RAG.58, CVL L4 thresholds | Cross-domain table in architecture §L4 Frozen |
| 5 | **AHI-MAINT-05** | Cross-ref | P2 | **Done** | Bandit arm → `ModelRouter` policy hint on live routing (not hardcoded `balanced`) | `llm_routing_wiring.py` + `check_live_model_routing_wiring.py` |
| 6 | **AHI-MAINT-06** | Code | P2 | **Done** | `ProfileVersionStore` `artifact_type=llm_routing` + persistent `BanditStateStore` for production routing hints (replace in-memory-only path) | [`M-LLM-X.10`](LLM_ADAPTERS.md#wave-m-llm-x-10--llm-routing-enterprise-closeout--predefined-rule-catalog); author rules still win |

**Suggested PR order:** AHI-MAINT-04 → AHI-MAINT-02 → AHI-MAINT-03 → AHI-MAINT-01 → AHI-MAINT-05 → **AHI-MAINT-06** (after M-LLM-X.10.2).

**Explicitly out of scope:** foundation model training — canon constraint.

---

*End of Adaptive Harness Intelligence Implementation Plan.*
