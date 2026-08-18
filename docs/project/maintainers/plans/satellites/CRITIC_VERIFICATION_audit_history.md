# CRITIC_VERIFICATION — audit history + LC closeout

**Parent hub:** [`CRITIC_VERIFICATION.md`](../CRITIC_VERIFICATION.md)

## Audit §CVL-1 — Scope

What was audited:

- PEV Verify infrastructure: L0/L1/L2 stack, `CriticOrchestrator`, graph hooks, evaluator-loop, offline semantic runner.
- Tier-0 tools: `eval.judge`, `eval.trajectory`, registry observation bridge.
- Tier-3 wiring: `CriticProfile`, `wire_application_critic()`, assembly resolver, policy `critic_governance`.
- Cross-domain: NEXUS_EXECUTION_FLOW §18 hooks, FAUDIT-EVAL.1 baseline gate, ACP reflection CVL gateway.

Out of scope: FLOW-8 product reference app (§6.3), domain rubric packs, L4 adaptive thresholds (AHIA).

---

## Audit §CVL-2 — Pre-CRIT-V gaps (historical) — all closed

| GAP-ID | Description | Closed by | Evidence |
|--------|-------------|-----------|----------|
| GAP-CVL-01 | No universal `eval.judge` primitive | CRIT-V-2.1 | `tools/providers/eval/judge.py` |
| GAP-CVL-02 | No trajectory evaluation contract | CRIT-V-2.2 | `tools/providers/eval/trajectory.py` |
| GAP-CVL-03 | Evaluator-loop catalog only | CRIT-V-4.1/4.2 | `evaluator_loop_executor.py` + `graph_executor.py` |
| GAP-CVL-04 | NexusEvalRunner exact-match only | CRIT-V-5.1/5.2 | `nexus_eval_runner.py` + `test_nexus_eval_runner_semantic.py` |
| GAP-CVL-05 | L0→L1→L2 stack not explicit | CRIT-V-1/3 | `contracts.py`, `critic_orchestrator.py` |
| GAP-CVL-06 | Evaluation layer L2 depth | CRIT-V-0…7 | 24/24 register Done; gate green |
| GAP-CVL-07 | L1 client not wired from Tier-3 tools | CRIT-V-F.1/F.2 | `critic_tool_wiring.py`, `tool_registry_client.py` |
| GAP-CVL-08 | No UAEP step critic hook | CRIT-V-F.4 | `validate_uaep_step_with_critic_detail` |
| GAP-CVL-09 | No policy bridge for verdict actions | CRIT-V-F.5 | `policy_bridge.py` |
| GAP-CVL-10 | Architecture §2 stale gap list | **CVL-LC-1** (2026-06-13) | This audit + architecture sync |

**Coverage:** 10 gaps — **all closed**. No open P0/P1 items.

---

## Audit §CVL-3 — Maturity scores (2026-06-13, iteration II)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Architecture Completeness | **96%** | Trajectory dual-mode documented |
| Production Readiness | **92%** | Bootstrap idempotency + combined gate stability |
| Documentation Consistency | **97%** | §7.3 trajectory clarification (CVL-LC-4) |
| Implementation Consistency | **95%** | Graph critic tests pass in combined sessions |

**State:** **Architecturally Mature** — no P0/P1 blockers.

---

## Sprint CVL-LC-1 — Documentation sync (**Done** 2026-06-13)

| Field | Value |
|-------|-------|
| **Scope** | Architecture §2 historical gaps + status; plan audit register; audit prompt regeneration |
| **Goal** | Honest L3+ layer status — no false “open gap” list at doc open |
| **DoD** | Architecture/plan/audit prompt aligned; closes GAP-CVL-10, CVL-BACKLOG-04 |
| **Files** | `docs/project/architecture/CRITIC_VERIFICATION.md`, `docs/project/maintainers/plans/CRITIC_VERIFICATION.md`, `scripts/audit/generate_domain_audit_prompts.py`, `docs/audit_results/CRITIC_VERIFICATION.md` |

## Sprint CVL-LC-2 — NexusEvalRunner semantic wiring (**Done** 2026-06-13)

| Field | Value |
|-------|-------|
| **Scope** | `from_nexus_loop` extracts L1 client from critic hooks; fail-closed when semantic enabled without client |
| **Goal** | Offline harness eval uses wired critic path without manual client injection |
| **DoD** | Unit test for auto-wire + fail-closed; closes CVL-BACKLOG-03 |
| **Files** | `intergrax/eval/nexus_eval_runner.py`, `intergrax/runtime/critic/l1_gateway.py`, `intergrax/runtime/nexus/nexus_loop.py`, `tests/unit/eval/test_nexus_eval_runner_semantic.py` |

## Sprint CVL-LC-3 — Tool catalog bootstrap idempotency (**Done** 2026-06-13)

| Field | Value |
|-------|-------|
| **Scope** | `register_default_tools()` overrides already-registered bundles instead of raising |
| **Goal** | `RuntimeContext.build()` safe after partial catalog registration in test sessions |
| **DoD** | Critic graph integration tests pass in combined CVL+eval gate runs; closes CVL-BACKLOG-02 |
| **Files** | `intergrax/tools/registry/bootstrap.py`, `intergrax/tools/registry/catalog.py`, `tests/unit/tools/registry/test_bootstrap_idempotent.py` |

## Sprint CVL-LC-4 — Trajectory eval dual-mode docs (**Done** 2026-06-13)

| Field | Value |
|-------|-------|
| **Scope** | Architecture §7.3 + plan backlog clarify heuristic `eval.trajectory` vs `eval.trajectory_judge` skill |
| **Goal** | Honest L1 trajectory contract — no false expectation of LLM rubric on `eval.trajectory` tool |
| **DoD** | Architecture/plan aligned; closes CVL-BACKLOG-01 (documented) |
| **Files** | `docs/project/architecture/CRITIC_VERIFICATION.md`, `docs/project/maintainers/plans/CRITIC_VERIFICATION.md` |

---

## Phase CRITIC_VERIFICATION-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates CVL-LC-1…4 + CRIT-V-0…7; no open P0/P1  
**Prerequisites:** CRIT-V **Done** · CVL layer completion **Done** (2026-06-13)  
**Goal:** Formal Full Harness LC closeout — gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| CVL-LC-FH-S1 | **Re-audit** — CRIT-V register + CVL verdict | **Done** | High | No P0/P1 |
| CVL-LC-FH-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| CVL-LC-FH-S3 | **Gate verification** | **Done** | High | 33 critic tests · 2 CI scripts |
| CVL-LC-FH-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** L4 adaptive critic thresholds (AHI) · FLOW-8 product host · LLM trajectory judge optional

### 6.1av Harness implementation queue — Critic verification audit maintenance (planned)

**Source:** Layer 18 audit (2026-06-18) — `CRITIC_VERIFICATION` layer 25 · [`../audit_results/2026-06-18/CRITIC_VERIFICATION.md`](../audit_results/2026-06-18/CRITIC_VERIFICATION.md)  
**Priority ladder:** **Band 1** (§6.1) — optional eval depth + cross-refs; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **CVL-MAINT-01** | Code | P3 | **Done** | Optional wire `eval.trajectory_judge` skill into default eval path (beyond heuristic `eval.trajectory`) | `trajectory_judge_path.py` + unit test; host/profile opt-in |
| 2 | **CVL-MAINT-02** | Cross-ref | P4 | **Done** | L4 adaptive critic thresholds — cross-ref [`AHI-MAINT-*`](ADAPTIVE_HARNESS_INTELLIGENCE.md#61av-harness-implementation-queue--adaptive-harness-intelligence-audit-maintenance-planned) (**Frozen** until product gate) | AHI owns thresholds |
| 3 | **CVL-MAINT-03** | Cross-ref | P4 | **Done** | FLOW-8 product reference host — cross-ref [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) §6.3 | No CVL-owned product host |
| 4 | **CVL-MAINT-04** | Cross-ref | P2 | **Done** | Per-tool L1 critic output trace — cross-ref [`TOOL-MAINT-02`](TOOLS.md#61av-harness-implementation-queue--tools-audit-maintenance-planned) | Trace contract in TOOLS canon |

**Suggested PR order:** CVL-MAINT-04 (doc cross-ref) → CVL-MAINT-01 → CVL-MAINT-02 → CVL-MAINT-03.

**Note:** AUDIT-IDEAL-25.3 product release eval gate — **Done** (2026-06-18 revalidation).

---

**Default answer (infrastructure):** **[§6.1](.#61-harness-platform-maintenance-default--band-1)** gate green on every PR — CRIT-V and OBS-BUS platform closeouts **Done**.

**Maintenance-only mode:** If CRIT-V paused by explicit decision, revert to §6.1 gate-only maintenance.

**Not default:** K.1, K.2, Legal UAEP domain steps, new product Tier-3 apps — **[§6.3](.#63-end-of-plan--deferred-product-work-only)** · **[§6.3a](.#63a-business-backlog-register-consolidated)** · **[§4.0a](.#40a-implementation-scope-split-infrastructure-vs-business)**.

**Audit basis:** Governance audit (2026-06-05) → GOV-AUDIT **Done**; orchestration audit (2026-06-05) → Phase ORCH + §6.1b; tools/skills audit (2026-06-02) → Phase TS + §6.1c; integration/RAG audit (2026-06-02) → Phase INT + RAG + §6.1d/§6.1e; context engineering audit (2026-06-02) → Phase CTX + §6.1f; prior V-REM/MEM/DX/AA closeouts in [§6.1z](.#61z-harness-implementation-queue-consolidated) / [§6.1aa](.#61aa-harness-implementation-queue-memory-platform).

### 6.1i Harness implementation queue — prompt registry closeout (closed)

**Purpose:** Single ordered list for **Phase PE** (Band 2p). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **PE-DOC.1** | Docs | **Done** | Appendix M + cross-refs | Author map complete |
| 2 | **PE-1** | Code | **Done** | `prompt_runtime_bridge` + `PromptProfile` | `test_prompt_runtime_bridge.py` |
| 3 | **PE-2** | Code | **Done** | `prompt_wiring` + `PromptRegistryProtocol` | `test_prompt_wiring.py` |
| 4 | **PE-3** | Code | **Done** | environment + runtime context wire | gate green |
| 5 | **PE-4** | Code | **Done** | Nexus prompt registry injection | `test_tools_step_prompt_registry.py` |

### 6.1j Harness implementation queue — legacy module closeout (closed)

**Purpose:** Single ordered list for **Phase CLEAN** (post-2p closeout). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CLEAN-1** | Code | **Done** | Remove `chat_router.py`; YAML-only tests | `test_chat_agent_prompts_yaml.py` |
| 2 | **CLEAN-2** | Code | **Done** | Remove `tools_agent.py`; planner tests | `test_catalog_tool_planner.py` |
| 3 | **CLEAN-3** | CI | **Done** | `check_legacy_modules_removed.py` in CI | workflow green |
| 4 | **CLEAN-4** | Docs | **Done** | Plan + harness docs sync | no stale production refs |

**Suggested PR order (complete):** CLEAN-1 → CLEAN-2 → CLEAN-3 → CLEAN-4.

### 6.1k Harness implementation queue — agent assembly closeout (closed)

**Purpose:** Single ordered list for **Phase AS** (Band 2q). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **AS-DOC.1** | Docs | **Done** | Appendix N + cross-refs | Author map complete |
| 2 | **AS-1** | Code | **Done** | `agent_assembly_resolver` | `test_agent_assembly_resolver.py` |
| 3 | **AS-2** | Code | **Done** | Lifecycle metadata enforcement | resolver + routing tests |
| 4 | **AS-3** | CI | **Done** | `check_agent_skill_resolution.py` | CI green |

**Suggested PR order (complete):** AS-DOC.1 → AS-1 → AS-2 → AS-3.

**Explicitly excluded:** K.1, K.2, new product agents, domain-only contract packs — [§6.3a](.#63a-business-backlog-register-consolidated).

### 6.1l Harness implementation queue — registry architecture closeout (closed)

**Purpose:** Single ordered list for **Phase REG** (Band 2r). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **REG-DOC.1** | Docs | **Done** | Appendix O + cross-refs | Author map complete |
| 2 | **REG-1** | Code | **Done** | `HarnessRegistrySnapshot` + `registry_wiring` | `test_registry_wiring.py` |
| 3 | **REG-2** | Code | **Done** | `registry_assembly_resolver` wire | `test_registry_wiring.py` |
| 4 | **REG-3** | CI | **Done** | `check_harness_registry_resolution.py` | CI green |

**Suggested PR order (complete):** REG-DOC.1 → REG-1 → REG-2 → REG-3.

### 6.1m Harness implementation queue — capability graph closeout (closed)

**Purpose:** Single ordered list for **Phase CG** (Band 2s). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CG-DOC.1** | Docs | **Done** | Appendix P + cross-refs | Author map complete |
| 2 | **CG-1** | Code | **Done** | `capability_graph_wiring` | `test_capability_graph_wiring.py` |
| 3 | **CG-2** | Code | **Done** | `capability_graph_assembly_resolver` | wire-time validation tests |
| 4 | **CG-3** | CI | **Done** | `check_harness_capability_graph_wiring.py` | CI green |

**Suggested PR order (complete):** CG-DOC.1 → CG-1 → CG-2 → CG-3.

### 6.1n Harness implementation queue — observability closeout (closed)

**Purpose:** Single ordered list for **Phase OBS** (Band 2t). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **OBS-DOC.1** | Docs | **Done** | Appendix Q + cross-refs | Author map complete |
| 2 | **OBS-1** | Code | **Done** | `observability_runtime_bridge` + `observability_wiring` | `test_harness_observability_wiring.py` |
| 3 | **OBS-2** | Code | **Done** | `observability_assembly_resolver` | wire-time validation tests |
| 4 | **OBS-3** | CI | **Done** | `check_harness_observability_wiring.py` | CI green |

**Suggested PR order (complete):** OBS-DOC.1 → OBS-1 → OBS-2 → OBS-3.

### 6.1o Harness implementation queue — reliability closeout (closed)

**Purpose:** Single ordered list for **Phase REL** (Band 2u). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **REL-DOC.1** | Docs | **Done** | Appendix R + cross-refs | Author map complete |
| 2 | **REL-1** | Code | **Done** | `reliability_runtime_bridge` + `reliability_wiring` | `test_harness_reliability_wiring.py` |
| 3 | **REL-2** | Code | **Done** | `reliability_assembly_resolver` | wire-time validation tests |
| 4 | **REL-3** | CI | **Done** | `check_harness_reliability_wiring.py` | CI green |

**Suggested PR order (complete):** REL-DOC.1 → REL-1 → REL-2 → REL-3.

### 6.1q Harness implementation queue — security closeout (closed)

**Purpose:** Single ordered list for **Phase SEC** (Band 2v). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **SEC-DOC.1** | Docs | **Done** | Appendix S + cross-refs | Author map complete |
| 2 | **SEC-1** | Code | **Done** | `security_runtime_bridge` + `security_wiring` | `test_harness_security_wiring.py` |
| 3 | **SEC-2** | Code | **Done** | `security_assembly_resolver` | wire-time validation tests |
| 4 | **SEC-3** | CI | **Done** | `check_harness_security_wiring.py` | CI green |

**Suggested PR order (complete):** SEC-DOC.1 → SEC-1 → SEC-2 → SEC-3.

### 6.1r Harness implementation queue — cost governance closeout (closed)

**Purpose:** Single ordered list for **Phase COST** (Band 2w). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **COST-DOC.1** | Docs | **Done** | Appendix T + cross-refs | Author map complete |
| 2 | **COST-1** | Code | **Done** | `CostProfile` + `cost_runtime_bridge` + `cost_wiring` | `test_harness_cost_wiring.py` |
| 3 | **COST-2** | Code | **Done** | `cost_assembly_resolver` | wire-time validation tests |
| 4 | **COST-3** | CI | **Done** | `check_harness_cost_wiring.py` | CI green |

**Suggested PR order (complete):** COST-DOC.1 → COST-1 → COST-2 → COST-3.

### 6.1s Harness implementation queue — evaluation closeout (closed)

**Purpose:** Single ordered list for **Phase EVAL** (Band 2x). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **EVAL-DOC.1** | Docs | **Done** | Appendix U + cross-refs | Author map complete |
| 2 | **EVAL-1** | Code | **Done** | `EvaluationProfile` + `evaluation_runtime_bridge` + `evaluation_wiring` | `test_harness_evaluation_wiring.py` |
| 3 | **EVAL-2** | Code | **Done** | `evaluation_assembly_resolver` | wire-time validation tests |
| 4 | **EVAL-3** | CI | **Done** | `check_harness_evaluation_wiring.py` | CI green |

**Suggested PR order (complete):** EVAL-DOC.1 → EVAL-1 → EVAL-2 → EVAL-3.

### 6.1f Harness implementation queue — context engineering closeout (closed)

**Purpose:** Single ordered list for **Phase CTX** (Band 2n). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CTX-DOC.1–2** | Docs | **Done** | Appendix L + cross-refs | Author map complete |
| 2 | **CTX-1** | Code | **Done** | `context_runtime_bridge` | `test_context_runtime_bridge.py` |
| 3 | **CTX-2** | Code | **Done** | `context_wiring` + `nexus_factory` | `test_context_wiring.py` |

### 6.1e Harness implementation queue — RAG closeout (closed)

**Purpose:** Single ordered list for **Phase RAG** (Band 2m). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **RAG-DOC.1** | Docs | **Done** | Appendix K §K.5 + AUDIT_MAP §14 | Author map complete |
| 2 | **RAG-1** | Code | **Done** | `rag_runtime_bridge` + environment wire | `test_rag_runtime_bridge.py` |

### 6.1d Harness implementation queue — integration closeout (closed)

**Purpose:** Single ordered list for **Phase INT** (Band 2l). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **INT-DOC.1–2** | Docs | **Done** | Appendix K + cross-refs | Author map complete |
| 2 | **INT-1** | Code | **Done** | `integration_runtime_bridge` | `test_integration_runtime_bridge.py` |
| 3 | **INT-2** | Code | **Done** | `integration_health_wiring` | `test_integration_health_wiring.py` |

### 6.1c Harness implementation queue — tools/skills closeout (closed)

**Purpose:** Single ordered list for **Phase TS** (Band 2k). **Closed 2026-06-02** — all TS rows **Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **TS-DOC.1–2** | Docs | **Done** | Appendix J + cross-refs | Author map complete |
| 2 | **TS-1** | Code | **Done** | `catalog_runtime_bridge` + `RuntimeConfig.skill_profile` | `test_catalog_runtime_bridge.py` |
| 3 | **TS-2** | Code | **Done** | Harness host `resolve_llm_adapter` wiring | `test_harness_host_runtime_llm.py` |
| 4 | **TS-3** | Code | **Done** | `SkillResolverProtocol` | skill resolver tests green |

**Suggested PR order (complete):** TS-1 → TS-2 → TS-3 → TS-DOC.*.

**Explicitly excluded:** K.1, K.2, new product tools/skills, business agent packs — [§6.3a](.#63a-business-backlog-register-consolidated).

### 6.1aa Harness implementation queue — memory platform (closed)

**Purpose:** Phase MEM execution queue — **closed 2026-06-02** (48/48 Done). Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **MEM-1.1–MEM-1.4** | Code | **Done** | H-APP `MemoryProfile` + `ContextProfile.budget` + SQLite session → `RuntimeConfig` | MEM-1.5 gate test green |
| 2 | **MEM-2.1–MEM-2.3** | Code | **Done** | `SQLiteUserProfileStore` + bundle wiring + unit tests | LTM survives restart on sqlite profile |
| 3 | **MEM-1.6** | Docs/status | **Done** | H-APP.4.3 → **Done** | Bridge complete |
| 4 | **MEM-4.1–MEM-4.3** | Test | **Done** | Session + LTM + full-stack memory gates | acceptance/integration green |
| 5 | **MEM-5.1–MEM-5.2** | Test/Docs | **Done** | `engine_history_layer` tests + compression docs | unit + guide |
| 6 | **MEM-3.1–MEM-3.3** | Code | **Done** | Memory store plugin EP + reference fixture | bootstrap + gate |
| 7 | **MEM-0.3–MEM-DOC.*** | Docs | **Done** | Author cookbooks + Appendix G sync | guide updated |
| 8 | **MEM-6.*–MEM-7.*** | Code | **Done** | Retention enforcement + memory hooks | P2 after P0/P1 |
| 9 | **MEM-8.*–MEM-9.*** | RFC | **Done (RFC)** | Product memory layer + entity graph design | §6.3 gate for implementation |

**Suggested PR order:** See [Phase MEM — Suggested PR order](.#mem--paydown-log).

**Explicitly excluded:** K.1, K.2, Mem0 SaaS product, entity graph ship (RFC only), business agent memory.

### 6.1aj Harness implementation queue — Nexus execution depth (closed)

**Purpose:** Single ordered list for **Phase FLOW** (Band 2aj). **Closed 2026-06-09** — **18/18 harness Done**; product host **Deferred** §6.3. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **FLOW-2** | Code | **Done** | ADR-FLOW-001 — `DELEGATES_TO` → child node | Delegation integration tests |
| 2 | **FLOW-14** | Code | **Done** | `SubtaskContract` in delegation expansion | Scopes on child `DelegationSpec` |
| 3 | **FLOW-3** | Code | **Done** | `max_delegation_depth` enforcement | Depth limit test |
| 4 | **FLOW-15** | Code | **Done** | Subagent budget envelope | Child budget exceeded → fail |
| 5 | **FLOW-6** | Code | **Done** | Graph cycle detection | Cyclic graph fails fast |
| 6 | **FLOW-1** | Code | **Done** | Real `EngineBackedNexusPlanner` | LLM plan parse tests |
| 7 | **FLOW-4** | Code | **Done** | Run-level retry profile field | Graph retry integration |
| 8 | **FLOW-13** | Code | **Done** | `max_inflight_nodes` profile + wire | Backpressure event test |
| 9 | **FLOW-7** | Code | **Done** | `MergePolicy` / composer profile | Multi-agent merge tests |
| 10 | **FLOW-9** | Code | **Done** | Multi-agent eval hooks | Registry observation |
| 11 | **FLOW-11** | Code | **Done** | Pre-plan policy hooks | Planning boundary tests |
| 12 | **FLOW-5** | Code | **Done** | `AgentGraph.on_error` wire | Integration test |
| 13 | **FLOW-10** | Code/Docs | **Done** | Reserved lifecycle ADR ([ADR-FLOW-002](adr/entries/2026-06-07/ADR-FLOW-002.md)) | Lifecycle doc |
| 14 | **FLOW-12** | Code | **Done** | `DecisionRecord` regression gate | Gate test per step |
| 15 | **FLOW-16** | Docs | **Done** | `MODIFY_PLAN` ADR (ADR-FLOW-003) | ADR accepted |
| 16 | **FLOW-17** | Code | **Done** | `MULTI_AGENT` ordering policy | Stable order gate test |
| 17 | **FLOW-DOC.*** | Docs | **Done** | Flow reference + Appendix N paydown | Zero open FLOW-GAP |
| — | **FLOW-8** | Product | **Deferred** | §42.43 reference app | **§6.3** gate only |

**Suggested PR order:** See [Phase FLOW — Suggested PR order](.#flow--suggested-pr-order).

**Explicitly excluded:** K.1, K.2 (unless FLOW-8 activated), nested harness per child.

### 6.1ak Harness implementation queue — Critic & Verification Layer (closed)

**Purpose:** Single ordered list for **Phase CRIT-V** (Band 2ak). **Closed 2026-06-08** — CRIT-V-0…7 + **CRIT-V-FOLLOWUP** closeout **Done**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **CRIT-V-0.*** | Docs | **Done** | Architecture RFC + ADR + canon §55 + README | Cross-links resolve |
| 2 | **CRIT-V-1.*** | Code | **Done** | `CriticProfile` + CVL contracts | Unit tests |
| 3 | **CRIT-V-2.*** | Code | **Done** | `eval.judge` + `eval.trajectory` tools | Tool gate tests |
| 4 | **CRIT-V-3.1–3.3** | Code | **Done** | `CriticOrchestrator` + L0/L1 gateways | `test_critic_orchestrator.py` |
| 5 | **CRIT-V-3.4–3.5** | Code | **Done** | Graph partial + final hooks | Integration tests |
| 5 | **CRIT-V-4.*** | Code | **Done** | `EvaluatorLoopExecutor` | Loop budget tests |
| 6 | **CRIT-V-5.*** | Code | **Done** | Semantic `NexusEvalRunner` | Eval integration test |
| 7 | **CRIT-V-6.*** | Code/Docs | **Done** | Tier-3 wiring + Appendix W | CI assembly script |
| 8 | **CRIT-V-7.*** | Code/Docs | **Done** | FAUDIT-EVAL.1 + flow reference sync | Closeout gate green |
| 9 | **CRIT-V-FOLLOWUP** | Code | **Done** | L1 client, L2 HITL, UAEP hook, policy bridge | `test_critic_closeout.py`, gate green |

**Suggested PR order:** See [§6.2ak](.#62ak-phase-crit-v-execution-order-band-2ak--closed).

**Explicitly excluded:** FLOW-8 product app; domain rubric packs in Tier-0; mandatory universal LLM-judge.

### 6.1al Harness implementation queue — Unified Observability Spine (closed)

**Purpose:** Single ordered list for **Phase OBS-BUS** (Band 2al). **Closed 2026-06-08** — all OBS-BUS rows **Done**; audit map §21 → **L4**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **OBS-BUS-0** | Docs | **Done** | `architecture/OBSERVABILITY.md` + ADR-OBS-001 + canon/README | Links resolve |
| 2 | **OBS-BUS-1** | Code | **Done** | `RuntimeEventPayload` registry | Payload registry gate |
| 3 | **OBS-BUS-2** | Code | **Done** | `ObservabilityEmitter` + `TraceScope` | Causal tree tests |
| 4 | **OBS-BUS-3** | Code | **Done** | Emission coverage gaps | `check_observability_emission_coverage.py` |
| 5 | **OBS-BUS-4** | Code/Docs | **Done** | Extension SDK + scaffold | Agent tracing template |
| 6 | **OBS-BUS-5** | Code | **Done** | Persistence conformance | Integration tests |
| 7 | **OBS-BUS-6** | Code | **Done** | OTLP/journal dual-write | `test_journal_export.py`, `test_export_bridge.py` |
| 8 | **OBS-BUS-7** | CI | **Done** | L4 §21 gates | `check_observability_gates.py` in CI; audit map §21 → L4 |

**Suggested PR order:** See [Phase OBS-BUS — Execution order](.#obs-bus--execution-order-recommended).

**Explicitly excluded:** Product dashboards (§6.3a); vendor-only APM as sole store.

### 6.1am Harness implementation queue — Memory intelligence depth (closed)

**Purpose:** Single ordered list for **Phase MEM-DEPTH** (Band 2am). **Closed 2026-06-08** — **26/26 Done**. Canonical: [plan/MEMORY.md](plan/MEMORY.md).

**Suggested PR order:** See [§6.2ab](plan/MEMORY.md#62ab-phase-mem-depth-execution-order-band-2am--closed).

**Explicitly excluded:** K.1/K.2, Mem0 SaaS, Redis session default — [§6.3a](.#63a-business-backlog-register-consolidated).

### 6.1b Harness implementation queue — orchestration closeout (closed)

**Purpose:** Single ordered list for **Phase ORCH** (Band 2j). **Closed 2026-06-05** — all ORCH rows **Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **ORCH-DOC.1–2** | Docs | **Done** | Appendix I + cross-refs | Author map complete |
| 2 | **ORCH-1** | Code | **Done** | `planner_kind` / `classifier_kind` wiring | `test_orchestration_wiring.py` |
| 3 | **ORCH-2** | Code | **Done** | `ApplicationGraphSpec` → `NexusPlan` | `test_graph_spec_to_plan.py` |
| 4 | **ORCH-3** | Code | **Done** | `max_parallel_nodes` cap | `test_graph_executor_parallel_cap.py` |
| 5 | **ORCH-4** | Docs | **Done** | Closeout sync | Plan + Appendix I updated |

**Suggested PR order (complete):** ORCH-1 → ORCH-2 → ORCH-3 → ORCH-4.

**Explicitly excluded:** K.1, K.2, new graph node types, nested harness per child — [§6.3a](.#63a-business-backlog-register-consolidated).

### 6.1g Harness implementation queue — governance audit (closed)

**Purpose:** Phase GOV-AUDIT documentation closeout — **closed 2026-06-05**.

| Order | ID | Status | Deliverable |
|-------|-----|--------|-------------|
| 1 | GOV-DOC.1 | **Done** | Appendix H control plane |
| 2 | GOV-DOC.2 | **Done** | Cross-ref sync |
| 3 | GOV-DOC.3 | **Done** | EXTENSION_AUTHOR §10 |
| — | GOV-PROD.1 | **Deferred** | Product dashboard → §6.3 |

### 6.1z Harness implementation queue (consolidated — closed 2026-06-05)

**Purpose:** Single ordered list of **infrastructure** work. Excludes Band 3 / [§6.3a](.#63a-business-backlog-register-consolidated). **Closed 2026-06-05** — Phase V-REM complete. Prior DX/AA/MEM/W-OPS/H-APP rows remain **Done**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green; scripts in [§6.1](.#61-harness-platform-maintenance-default--band-1) |
| 1 | **V-REM-CG.1** | Code | **Done** | Fix per-application capability graph system edges | V-CG.2–4 closed |
| 2 | **V-REM-CG.2** | Test/CI | **Done** | Re-validate lineage/impact/compatibility on corrected graph | `phase_v_capability_graph_guard.py` green |
| 3 | **V-REM-ALG.1** | Code | **Done** | Runtime filter retired/deprecated agents | Unit tests green |
| 4 | **V-REM-ALG.2** | Code | **Done** | Production-eligible + owner gate at selection | Strict harness test green |
| 5 | **V-REM-PE.1** | Code | **Done** | PromptMeta owner/risk schema | Registry validation tests |
| 6 | **V-REM-PE.2** | Assets | **Done** | YAML prompt assets catalog seed | E2E governance validation |
| 7 | **V-REM-SEC.1** | Code | **Done** | Tool injection defense on execution path | Middleware unit tests |
| 8 | **V-REM-SEC.2** | Code | **Done** | Retrieval poisoning middleware per tenant/app | RagStep filter unit tests |
| 9 | **V-REM-SEC.3** | Code | **Done** | Tenant isolation + audit trail in main path | Intake middleware unit tests |
| 10 | **V-REM-A.1** | Test | **Done** | NexusEvalRunner integration + gate | A.4 → **Done** |
| — | **REG-*** | Regression | As needed | Fix gate/CI failures only | No feature scope |

**Closed (no implementation — do not reopen without regression):**

| ID | Resolution |
|----|------------|
| DX-0.3–DX-8.2 (except DX-5.7) | **Done** — 2026-06-02 DX residual closeout |
| AA-LABAG.1, AA-SIG.2, AA-LABAPP.6 | **Done** |
| AA-LABAG.2 | **Won't fix** — mocks remain in `agents/lab` until leadership requests move |
| W-OPS.1–15, H-APP.0–6.3, P-Ext, Q–V contracts, MEM 48/48 | **Done** |
| V-REM.0.1, V-REM.0.2 | **Done** — 2026-06-05 plan sync |
| V-REM-CG.1–A.1 | **Done** — 2026-06-05 runtime remediation |

**Explicitly excluded from this queue (business — implement only after §6.3 decision):** K.1, K.2, K.6, B.15, S-Ops.4, A.5, AA-LEG.2.2+, AA-LEGAPP.6–8, AA-RES.4–5, AA-RESAPP.6, AA-ORG.3–4, new Tier-3 product apps, domain skills — full list: [§6.3a](.#63a-business-backlog-register-consolidated).

**Suggested PR order:** V-REM-CG.1 → V-REM-CG.2 → V-REM-ALG.1 → V-REM-ALG.2 → V-REM-SEC.1 → V-REM-SEC.2 → V-REM-SEC.3 → V-REM-PE.1 → V-REM-PE.2 → V-REM-A.1. Regressions → **REG-*** under §6.1.

**Explicitly excluded:** K.1, K.2, new product eval modes requiring business datasets — [§6.3a](.#63a-business-backlog-register-consolidated).

### 6.1t Harness implementation queue — Adaptive Harness Intelligence (closed)

**Purpose:** Single ordered list for **Phase W-ADAPT** (Band 2y). **Closed 2026-06-02** — **70/70 Done** (Wave W-ADAPT-0 through Wave W-ADAPT-7 **Done**). Maintenance-only; see [§6.1](.#61-harness-platform-maintenance-default--band-1).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **W-ADAPT-0.2–0.5** | Docs/Code | **Done** | ADR-ADAPT-001 + `intergrax/runtime/adaptive` scaffold | Import + gate stub |
| 2 | **W-ADAPT-1.1–1.12** | Code | **Done** | Observe (L4-O): signals + utility + report | `phase_w_adapt_report.py` |
| 3 | **W-ADAPT-2.1–2.12** | Code | **Done** | Recommend (L4-R): engines + proposals (no apply) | Proposals in report |
| 4 | **W-ADAPT-3.1–3.7** | Code | **Done** | Shadow (L4-S): ProfileVersionStore + executor.shadow | Integration test green |
| 5 | **W-ADAPT-4.1–4.10** | Code | **Done** | Apply (L4-A): canary, apply, rollback, events | Policy learning HITL enforced |
| 6 | **W-ADAPT-5.1–5.12** | Code/Docs | **Done** | Verify (L4-V): VerificationLoop + runtime L4 closeout | `--enforce-l4-runtime` |
| 7 | **W-ADAPT-6.1–6.5** | Code | **Done** | ProcessPatternMiner + daily scheduler | pattern report |
| 8 | **W-ADAPT-7.1–7.7** | Code/Docs | **Done** | Tier-3 AdaptiveProfile + Appendix V + acceptance | E2E observe→recommend |

**Suggested PR order:** See [Phase W-ADAPT — Suggested PR order](.#w-adapt--suggested-pr-order).

**Explicitly excluded:** K.1, K.2, deep RL, foundation model training, autonomous prompt edits — [§6.3a](.#63a-business-backlog-register-consolidated).

### 6.1v Harness implementation queue — LLM completion response envelope (closed)

**Purpose:** Single ordered list for **Phase M-LLM-R** (Band 2z). **Closed 2026-06-06** — **39/39 Done**. Runs **in parallel** with W-ADAPT waves 5–7 (Tier-0 LLM contract; independent of L4 runtime loop).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **M-LLM-R.0.2–0.3** | Docs | **Done** | ADR-LLM-001 + canon §5.2.2 addendum | Linked from plan |
| 2 | **M-LLM-R.1.1–1.8** | Code | **Done** | Contract types + builders + public exports | Import smoke; no dict returns |
| 3 | **M-LLM-R.2.1–2.6** | Code | **Done** | `LLMAdapter` ABC typed signatures | ABC compiles; stubs updated |
| 4 | **M-LLM-R.3.1–3.7** | Code | **Done** | All provider adapters return envelope | Conformance per provider family |
| 5 | **M-LLM-R.4.1–4.6** | Code | **Done** | Nexus runtime consumers | `test_context_preflight + ACP agent tests` + tool planner |
| 6 | **M-LLM-R.5.1–5.3** | Code | **Done** | RAG + websearch + legacy | RAG unit tests green |
| 7 | **M-LLM-R.6.1–6.4** | Code | **Done** | Agents + scaffold + CI lint | `check_llm_adapter_typed_returns.py` + `check_agents_llm_adapter_response.py` |
| 8 | **M-LLM-R.7.1–7.5** | Code | **Done** | Usage alignment + replay/trace bridge | `test_replay_engine` + diagnostics |
| 9 | **M-LLM-R.8.1–8.4** | Docs/CI | **Done** | Docs + conformance + closeout | M-LLM.14 Done; Appendix L complete |

**Suggested PR order:** See [Phase M-LLM-R — Suggested PR order](.#phase-m-llm-r--llm-completion-response-envelope-audit-2026-06-06).

**Explicitly excluded:** K.1, K.2, product HTTP API DTOs, provider SDK rewrites — [§6.3a](.#63a-business-backlog-register-consolidated).

### 6.1w Harness implementation queue — Integration expansion (M.6 P4 closed)

**Purpose:** Ordered backlog for **Phase M.6 P4** (Band 2aa). **Status:** **Done** (2026-06-02) — **28/28 Done** · catalog **127**.  
**Register:** [M.6 P4 — Master register](.#m6-p4--master-register-28-slugs) · **Execution order:** [§6.2ae](.#62ae-phase-m6-p4-execution-order--done)
**Policy:** One slug per PR; runs **in parallel** with §6.1 maintenance — pull only when harness ops/adaptive/INT health needs the slug.

| Order | Wave | IDs | Slugs | Priority | Status |
|-------|------|-----|-------|----------|--------|
| 0 | CAT | M-P4-CAT.1, M-P4-CAT.2 | *(categories)* | **P0** | **Done** (beta) |
| 1 | H-INT-1 | M-P4.1–M-P4.4 | `pgvector`, `duckdb`, `influxdb`, `timescaledb` | P0/P1 | **Done** |
| 2 | H-INT-2 | M-P4.5–M-P4.7 | `grafana`, `loki`, `tempo` | **P0** | **Done** |
| 3 | H-INT-3 | M-P4.8–M-P4.11 | `aws_secrets_manager`, `azure_key_vault`, `gcp_secret_manager`, `doppler` | P0/P1 | **Done** |
| 4 | H-INT-4 | M-P4.12–M-P4.16 | `unleash`, `launchdarkly`, `github_actions`, `redpanda`, `cloudflare_r2` | P0/P1 | **Done** |
| 5 | H-INT-5 | M-P4.17–M-P4.28 | `memgraph`, `falkordb`, `incident_io`, `kubernetes`, `servicenow`, `bitbucket`, `asana`, `sendgrid`, `mailgun`, `mlflow`, `huggingface_hub`, `ollama` | P1/P2 | **Done** |

**Per-slug checklist (M.4):** contract → `providers/<category>/<slug>` → unit tests → `USAGE.md` → `layout.py` → `architecture/INTEGRATIONS.md` → canon §7.1.3 row → gate green → paydown log row.

**Explicitly excluded:** CRM, payments, blockchain, duplicate vector SaaS, LLM vendor APIs — see [M.6 P4 register](.#m6-p4--harness-platform-expansion-planned).

### 6.1x Harness implementation queue — Integration depth (M.6 P5 done)

**Purpose:** Closeout record for **Phase M.6 P5** (Band 2ab). **Status:** **Done** (2026-06-02) — **33/34**.  
**Register:** [M.6 P5 — Master register](.#m6-p5--master-register-34-slugs) · **Execution order:** [§6.2af](.#62af-phase-m6-p5-execution-order-band-2ab--planned)
**Policy:** One slug per PR (or one harden wave ≤4 slugs); runs **in parallel** with §6.1 maintenance — pull when W-OPS / W-ADAPT / EVAL / prod stack needs the slug.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 0 | CAT | M-P5-CAT.1–3 | `ci_cd` extend, `security_scanner`, category mapping | **P0** | **Done** (CAT.2 deferred: `trivy`) |
| 1 | H-INT-6 | M-P5.1–M-P5.10 | Ops/metrics/CI/local cloud: prometheus, clickhouse, vault, pagerduty, github, gitlab_ci, circleci, azure_pipelines, mailpit, localstack | **P0** | **Done** |
| 2 | H-INT-7 | M-P5.11–M-P5.20 | Eval/async/artifacts: langfuse, phoenix, braintrust, mlflow, influxdb, timescaledb, temporal, redpanda, minio, s3 | **P0/P1** | **Done** |
| 3 | H-INT-8 | M-P5.21–M-P5.28 | Data plane lab: neo4j, mongodb, elasticsearch, nats, chroma, weaviate, launchdarkly, signoz | **P1/P2** | **Done** |
| 4 | H-INT-9 | M-P5.29–M-P5.34 | P2 reserve: codecov, trivy, grafana_oncall, opentelemetry_collector, snowflake, supabase | **P2** | **Done** |
| 5 | PRE | M-P5-PRE.1 | Tier-3 presets: `harness_metrics_stack`, `harness_eval_stack`, `harness_async_stack`, `harness_ci_stack` | **P0** | **Done** |

**Explicitly excluded:** Band 3 product agents; see [M.6 P5 register](.#m6-p5--harness-integration-depth-done--3334).

### 6.1y Harness implementation queue — Integration expansion (M.6 P6 Done)

**Purpose:** Ordered backlog for **Phase M.6 P6** (Band 2ac). **Status:** **Done** (2026-06-02) — **32/32**.  
**Register:** [M.6 P6 — Master register](.#m6-p6--master-register-32-slugs) · **Execution order:** [§6.2ag](.#62ag-phase-m6-p6-execution-order-band-2ac--done)
**Policy:** One slug per PR (or one CAT wave before first slug in a new category); runs **in parallel** with §6.1 maintenance — pull when security/sandbox/identity/GitOps/speech harness gaps block ops.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 0 | CAT | M-P6-CAT.1–9 | New categories: `security_scanner`, `sandbox_host`, `identity_provider`, `speech_provider`, `workflow_orchestrator`, `vision_serving`, `ml_inference_host`, `billing_meter`, `crm` | **P0** | **Done** |
| 1 | H-INT-10 | M-P6.1–M-P6.4 | Security + secrets: `trivy`, `snyk`, `semgrep`, `infisical` | **P0** | **Done** |
| 2 | H-INT-11 | M-P6.5–M-P6.7 | Cloud sandbox: `e2b`, `modal`, `daytona` | **P0/P1** | **Done** |
| 3 | H-INT-12 | M-P6.8–M-P6.10 | Identity: `auth0`, `keycloak`, `workos` | **P0/P1** | **Done** |
| 4 | H-INT-13 | M-P6.11–M-P6.13 | GitOps CI: `argocd`, `buildkite`, `jenkins` | **P0/P1** | **Done** |
| 5 | H-INT-14 | M-P6.14–M-P6.15 | Speech catalog: `elevenlabs`, `deepgram` | **P0** | **Done** |
| 6 | H-INT-15 | M-P6.16–M-P6.19 | Enterprise ops: `newrelic`, `splunk`, `zendesk`, `statsig` | **P1** | **Done** |
| 7 | H-INT-16 | M-P6.20–M-P6.24 | Data/workflow: `prefect`, `airflow`, `typesense`, `neon`, `pulsar` | **P1** | **Done** |
| 8 | H-INT-17 | M-P6.25–M-P6.32 | Reserve: `algolia`, `confluent`, `backblaze_b2`, `triton`, `replicate`, `stripe`, `salesforce`, `hubspot` | **P2** | **Done** |
| 9 | PRE | M-P6-PRE.1 | Tier-3 presets: `harness_security_stack`, `harness_sandbox_stack`, `harness_identity_stack`, `harness_gitops_stack` | **P0** | **Done** |
| 10 | WIRE | M-P6-WIRE.1–7 | Tool surface + sandbox/speech/identity bridges + promote gate + infra `p6` | **P0** | **Done** |

**Per-slug checklist:** see [M.6 P6 register](.#m6-p6--harness-integration-expansion-planned).

**Closeout target:** catalog **167** slugs; optional `HARNESS_M6_P6_PROBE_SLUGS`; four Tier-3 presets; gate green.

### 6.1 Harness platform maintenance (default — Band 1)

§4.1 backlog is **closed**. Ongoing work = keep the harness green; **Band 2y W-ADAPT**, **Band 2z M-LLM-R**, **Band 2aa M.6 P4**, and **Band 2ab M.6 P5** are **closed**. **Band 2ac M.6 P6** = **Done** (32/32) — see **[§6.1y](.#61y-harness-implementation-queue--integration-expansion-m6-p6-done)**. **Next product work** = [§6.3](.#63-end-of-plan--deferred-product-work-only) (product prioritization only).

```text
Verify (every harness PR):
  uv run pytest -m gate -q
  python scripts/maintenance/check_harness_no_getattr.py
  python scripts/maintenance/check_legacy_modules_removed.py
  python scripts/maintenance/check_agent_skill_resolution.py
  python scripts/maintenance/check_harness_registry_resolution.py
  python scripts/maintenance/check_harness_capability_graph_wiring.py
  python scripts/maintenance/check_legacy_tool_plan_booleans.py
  python scripts/maintenance/check_trace_bridge_event_catalog.py
  python scripts/maintenance/check_plugin_catalog.py
  python scripts/maintenance/check_llm_adapter_typed_returns.py
  python scripts/maintenance/check_agents_llm_adapter_response.py
  uv run python scripts/release/phase_w_ops_evidence.py
  # Per release (ops):
  uv run python scripts/release/export_harness_shadow_eval_trend.py --release-id <release-id>
  uv run python scripts/release/record_harness_release_cycle.py --cycle-id <release-id> --verify-gate
  python scripts/maintenance/check_scaffold_harness_alignment.py
  python scripts/maintenance/check_agents_no_tier3_imports.py
  python scripts/maintenance/check_intergrax_no_applications_imports.py
  uv run python scripts/maintenance/check_harness_prompt_golden_catalog.py
  uv run python scripts/maintenance/check_agents_lifecycle_metadata.py
  uv run intergrax doctor --ci
  uv run python scripts/release/phase_v_closeout_gate.py --enforce --enforce-l4
  uv run python scripts/release/phase_w_adapt_closeout_gate.py --enforce-l4-runtime
  uv run python scripts/release/phase_v_capability_graph_guard.py --enforce
```

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3.

**Maintenance depth (2026-06-07):** **OBS-DEPTH.1 Done** — unified run journal. **T10-DEPTH.1 Done** — broker task index + PagerDuty acknowledge adapter. **T-EXPAND T11 Done** — 160 tools. **LEG-DEPTH.1–3 + O.5 depth Done** — planner schema uses `tool_ids`; legacy booleans accepted with deprecation trace; `from_legacy()` gated by `check_legacy_tool_plan_booleans.py`. **OBS-DEPTH.2 Done** — `check_trace_bridge_event_catalog.py` + gate test. **OBS live emit Done** — `RuntimeState.trace_event` → `runtime_event_bus`. **Celery purge_completed Done** — optional KV task index. **notify.dispatch_due Done** — Tier-0 dispatcher tool. **T-EXPAND T12 Done** — 170 tools (health slot probes + notify dispatcher). **T-EXPAND T13 Done** — 172 tools (`eval.judge`, `eval.trajectory` / CRIT-V). **L2→L3 §21 Done** — `test_observability_layer_depth_gate.py` regression gate.

### 6.1ah Harness implementation queue — FAUDIT-32 remediation (closed)

**Status:** **Done** (2026-06-06) — **23/23 Done**  
**Source:** [Phase FAUDIT-32](.#phase-faudit-32--full-architecture-audit-closeout) · **Appendix M**
**Priority ladder:** **Band 2ad** (§4.0) — runs **after** FAUDIT-TIER.1 on every harness PR that touches `intergrax/runtime/architecture`

**Execution order (recommended):**

```text
Wave P0 (architecture integrity):
  FAUDIT-TIER.1 → FAUDIT-TIER.2

Wave P1 (identity + intake + observability):
  FAUDIT-INTAKE.1 → FAUDIT-ID.1 → FAUDIT-OBS.1 → FAUDIT-EVAL.1

Wave P2 (control-plane depth):
  FAUDIT-PE.1 → FAUDIT-REG.1 → FAUDIT-CG.1 → FAUDIT-CG.2
  → FAUDIT-SEC.1 → FAUDIT-REL.1 → FAUDIT-COST.1

Wave P3 (orchestration + cognition + memory):
  FAUDIT-ORCH.1 → FAUDIT-SUB.1 → FAUDIT-COG.1 → FAUDIT-LLM.1
  → FAUDIT-POL.1 → FAUDIT-MEM.1 → FAUDIT-ALG.1 → FAUDIT-OPS.1
  → FAUDIT-INTAKE.2 → FAUDIT-ID.2
```

| ID | Status | Priority | Blocks |
|----|--------|----------|--------|
| FAUDIT-TIER.1 | **Done** | **Critical** | `intergrax/applications/reference/harness_manifest_catalog.py` |
| FAUDIT-TIER.2 | **Done** | High | `scripts/maintenance/check_intergrax_no_applications_imports.py` |
| FAUDIT-INTAKE.1 | **Done** | High | `intergrax/contracts/task_envelope.py` |
| FAUDIT-INTAKE.2 | **Done** | Medium | `tests/unit/runtime/architecture/test_faudit_remediation.py` |
| FAUDIT-ID.1 | **Done** | High | `intergrax/contracts/actor_identity.py` |
| FAUDIT-ID.2 | **Done** | Medium | `DelegationSpec.permission_scopes` |
| FAUDIT-POL.1 | **Done** | High | `PolicyEngine.evaluate_pre_llm/pre_output` |
| FAUDIT-LLM.1 | **Done** | High | `intergrax/llm_adapters/registry/model_router.py` |
| FAUDIT-COG.1 | **Done** | High | `intergrax/contracts/decision_record.py` + UAEP emit |
| FAUDIT-ORCH.1 | **Done** | Medium | `GraphExecutor` inflight backpressure |
| FAUDIT-SUB.1 | **Done** | High | `SubtaskContract` + safer defaults |
| FAUDIT-MEM.1 | **Done** | High | `retention_enforcement.py` + `PolicyScopedMemoryView` STM purge |
| FAUDIT-PE.1 | **Done** | High | `prompt_golden_catalog.py` + `tests/fixtures/prompt_golden` + CI script |
| FAUDIT-ALG.1 | **Done** | High | lifecycle states + reference agent `owner_team` adoption + CI script |
| FAUDIT-REG.1 | **Done** | High | `HarnessRegistrySnapshot` agent/eval fields |
| FAUDIT-CG.1 | **Done** | High | prompt seeds in `capability_graph_wiring.py` |
| FAUDIT-CG.2 | **Done** | Medium | `phase_v_capability_graph_guard.py` impact log |
| FAUDIT-OBS.1 | **Done** | High | `RuntimeEventType.LLM_CALL/POLICY_DECISION` |
| FAUDIT-REL.1 | **Done** | High | expanded `RuntimeErrorCode` + classifier |
| FAUDIT-SEC.1 | **Done** | High | `intergrax/contracts/data_classification.py` |
| FAUDIT-COST.1 | **Done** | High | `run_budget` wired in `nexus_factory` |
| FAUDIT-EVAL.1 | **Done** | High | `phase_v_closeout_gate.py` eval baseline |
| FAUDIT-OPS.1 | **Done** | Medium | `build/architecture_hardening/release_cycles.json` |

**DoD (§6.1ah queue closure):** All **Planned** rows **Done**; Appendix M scorecard shows **0 Critical**, **≤5 High** (documented deferrals only); tier gate green.

### 6.1ai Harness implementation queue — FAUDIT-32 follow-up (closed)

**Status:** **Done** (2026-06-06) — post-remediation depth for PE/ALG/MEM adoption  
**Priority ladder:** **Band 2ad** (§4.0) — runs after §6.1ah closure

| ID | Status | Deliverable |
|----|--------|-------------|
| FAUDIT-PE.1+ | **Done** | Real `prompts` golden hashes in `tests/fixtures/prompt_golden/expectations.json`; `scripts/maintenance/check_harness_prompt_golden_catalog.py`; gate test |
| FAUDIT-ALG.1+ | **Done** | `lifecycle_state` + `owner_team` on reference Tier-2 agents; `scripts/maintenance/check_agents_lifecycle_metadata.py` |
| FAUDIT-MEM.1+ | **Done** | `should_forget_stm_record` wired in `PolicyScopedMemoryView.read` |

**Explicitly deferred (Band 3 / product):** MEM-9 entity graph memory implementation (RFC only); K.1/K.2 business agents.

### 6.2bo Phase EVAL execution order (Band 2x — closed 2026-06-02)

**Status:** **Done** · register: [Phase EVAL](.#phase-eval--evaluation-control-plane-closeout) · queue: [§6.1s](.#61s-harness-implementation-queue--evaluation-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | EVAL-DOC.1 | Appendix U + plan sync | High |
| 2 | EVAL-1 | `EvaluationProfile` + `evaluation_runtime_bridge` + `evaluation_wiring` | Critical |
| 3 | EVAL-2 | `evaluation_assembly_resolver` | High |
| 4 | EVAL-3 | `check_harness_evaluation_wiring.py` | Medium |

### 6.2bn Phase COST execution order (Band 2w — closed 2026-06-02)

**Status:** **Done** · register: [Phase COST](.#phase-cost--cost-governance-control-plane-closeout) · queue: [§6.1r](.#61r-harness-implementation-queue--cost-governance-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | COST-DOC.1 | Appendix T + plan sync | High |
| 2 | COST-1 | `CostProfile` + `cost_runtime_bridge` + `cost_wiring` | Critical |
| 3 | COST-2 | `cost_assembly_resolver` | High |
| 4 | COST-3 | `check_harness_cost_wiring.py` | Medium |

### 6.2bm Phase SEC execution order (Band 2v — closed 2026-06-02)

**Status:** **Done** · register: [Phase SEC](.#phase-sec--security-control-plane-closeout) · queue: [§6.1q](.#61q-harness-implementation-queue--security-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | SEC-DOC.1 | Appendix S + plan sync | High |
| 2 | SEC-1 | `security_runtime_bridge` + `security_wiring` | Critical |
| 3 | SEC-2 | `security_assembly_resolver` | High |
| 4 | SEC-3 | `check_harness_security_wiring.py` | Medium |

### 6.2bl Phase REL execution order (Band 2u — closed 2026-06-02)

**Status:** **Done** · register: [Phase REL](.#phase-rel--reliability-control-plane-closeout) · queue: [§6.1o](.#61o-harness-implementation-queue--reliability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REL-DOC.1 | Appendix R + plan sync | High |
| 2 | REL-1 | `reliability_runtime_bridge` + `reliability_wiring` | Critical |
| 3 | REL-2 | `reliability_assembly_resolver` | High |
| 4 | REL-3 | `check_harness_reliability_wiring.py` | Medium |

### 6.2bk Phase OBS execution order (Band 2t — closed 2026-06-02)

**Status:** **Done** · register: [Phase OBS](.#phase-obs--observability-control-plane-closeout) · queue: [§6.1n](.#61n-harness-implementation-queue--observability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | OBS-DOC.1 | Appendix Q + plan sync | High |
| 2 | OBS-1 | `observability_runtime_bridge` + `observability_wiring` | Critical |
| 3 | OBS-2 | `observability_assembly_resolver` | High |
| 4 | OBS-3 | `check_harness_observability_wiring.py` | Medium |

### 6.2bj Phase CG execution order (Band 2s — closed 2026-06-02)

**Status:** **Done** · register: [Phase CG](.#phase-cg--capability-graph-control-plane-closeout) · queue: [§6.1m](.#61m-harness-implementation-queue--capability-graph-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CG-DOC.1 | Appendix P + plan sync | High |
| 2 | CG-1 | `capability_graph_wiring` | Critical |
| 3 | CG-2 | `capability_graph_assembly_resolver` | High |
| 4 | CG-3 | `check_harness_capability_graph_wiring.py` | Medium |

### 6.2bi Phase REG execution order (Band 2r — closed 2026-06-02)

**Status:** **Done** · register: [Phase REG](.#phase-reg--registry-architecture-control-plane-closeout) · queue: [§6.1l](.#61l-harness-implementation-queue--registry-architecture-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REG-DOC.1 | Appendix O + plan sync | High |
| 2 | REG-1 | `HarnessRegistrySnapshot` + `registry_wiring` | Critical |
| 3 | REG-2 | `registry_assembly_resolver` | High |
| 4 | REG-3 | `check_harness_registry_resolution.py` | Medium |

### 6.2bg Phase AS execution order (Band 2q — closed 2026-06-02)

**Status:** **Done** · register: [Phase AS](.#phase-as--agent-assembly-control-plane-closeout) · queue: [§6.1k](.#61k-harness-implementation-queue--agent-assembly-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | AS-DOC.1 | Appendix N + plan sync | High |
| 2 | AS-1 | `agent_assembly_resolver` | Critical |
| 3 | AS-2 | Lifecycle state on `AgentContract` | High |
| 4 | AS-3 | `skill_ids` resolution audit script | Medium |

### 6.2bh Phase CLEAN execution order (closed 2026-06-02)

**Status:** **Done** · register: [Phase CLEAN](.#phase-clean--legacy-module-closeout) · queue: [§6.1j](.#61j-harness-implementation-queue--legacy-module-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CLEAN-1 | Remove `chat_router.py` | Critical |
| 2 | CLEAN-2 | Remove `tools_agent.py` | Critical |
| 3 | CLEAN-3 | `check_legacy_modules_removed.py` in CI | High |
| 4 | CLEAN-4 | Docs sync | Low |

### 6.2bf Phase CTX execution order (Band 2n — closed 2026-06-02)

**Status:** **Done** · register: [Phase CTX](.#phase-ctx--context-engineering-control-plane-closeout) · queue: [§6.1f](.#61f-harness-implementation-queue--context-engineering-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CTX-1 | `context_runtime_bridge` | Critical |
| 2 | CTX-2 | `context_wiring` + Nexus factory wire | High |
| 3 | CTX-DOC.1–2 | Appendix L + plan sync | Low |

### 6.2be Phase RAG execution order (Band 2m — closed 2026-06-02)

**Status:** **Done** · register: [Phase RAG](.#phase-rag--rag-retrieval-control-plane-closeout) · queue: [§6.1e](.#61e-harness-implementation-queue--rag-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | RAG-1 | `rag_runtime_bridge` + environment wire | Critical |
| 2 | RAG-DOC.1 | Appendix K §K.5 + plan sync | Low |

### 6.2bd Phase INT execution order (Band 2l — closed 2026-06-02)

**Status:** **Done** · register: [Phase INT](.#phase-int--integration-control-plane-closeout) · queue: [§6.1d](.#61d-harness-implementation-queue--integration-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | INT-1 | `integration_runtime_bridge` | Critical |
| 2 | INT-2 | `integration_health_wiring` | High |
| 3 | INT-DOC.1–2 | Appendix K + plan sync | Low |

### 6.2bc Phase TS execution order (Band 2k — closed 2026-06-02)

**Status:** **Done** · register: [Phase TS](.#phase-ts--tools--skills-control-plane-closeout) · queue: [§6.1c](.#61c-harness-implementation-queue--toolsskills-closeout-closed)

Work **one TS ID per PR**; after each step update the TS master table + §6.1c + paydown log; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | TS-1 | `catalog_runtime_bridge` + `materialize_runtime_config` | Critical | TS-DOC.* (parallel OK) |
| 2 | TS-2 | Harness host LLM adapter wiring | High | — |
| 3 | TS-3 | `SkillResolverProtocol` | Medium | — |
| 4 | TS-DOC.1–2 | Appendix J + plan sync | Low | TS-1–3 |

### 6.2aj Phase FLOW execution order (Band 2aj — closed 2026-06-07)

**Status:** **Done** · register: [Phase FLOW](.#phase-flow--nexus-execution-depth) · queue: [§6.1aj](.#61aj-harness-implementation-queue--nexus-execution-depth-closed)

Work **one FLOW ID per PR**; after each step update FLOW master table + §6.1aj + Appendix N; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | FLOW-2 | Delegation graph expansion (ADR-FLOW-001) | **Critical** | — |
| 2 | FLOW-14 | `SubtaskContract` on expanded child node | High | FLOW-2 |
| 3 | FLOW-3 | `max_delegation_depth` enforcement | High | FLOW-2 |
| 4 | FLOW-15 | Subagent budget envelope | Medium | FLOW-14 |
| 5 | FLOW-6 | Strict graph cycle detection | High | — |
| 6 | FLOW-1 | LLM-backed Nexus planner | High | — (parallel with 5–8 after step 1) |
| 7 | FLOW-4 | Run-level retry profile | Medium | FLOW-2 |
| 8 | FLOW-13 | `max_inflight_nodes` profile wire | Medium | — |
| 9 | FLOW-7 | Merge policy / composer profile | Medium | — |
| 10 | FLOW-9 | Multi-agent evaluation hooks | Medium | FLOW-7 optional |
| 11 | FLOW-11 | Pre-plan policy hooks | Medium | — |
| 12 | FLOW-5 | `AgentGraph.on_error` wire | Low | FLOW-4 optional |
| 13 | FLOW-10 | Reserved lifecycle states ADR | Low | — |
| 14 | FLOW-12 | `DecisionRecord` regression gate | Medium | — |
| 15 | FLOW-16 | `MODIFY_PLAN` ADR (ADR-FLOW-003) | Low | — |
| 16 | FLOW-17 | `MULTI_AGENT` ordering policy | Low | — |
| 17 | FLOW-DOC.* | Docs closeout | Low | FLOW-1–17 (except deferred FLOW-8) |

### 6.2ak Phase CRIT-V execution order (Band 2ak — closed)

**Status:** **Done** (2026-06-08) · register: [Phase CRIT-V](.#phase-crit-v--critic--verification-layer) · queue: [§6.1ak](.#61ak-harness-implementation-queue--critic-verification-layer-closed)

Work **one CRIT-V ID per PR**; after each step update CRIT-V master table + §6.1ak; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | CRIT-V-0.* | Architecture + ADR + canon + README | High | — |
| 2 | CRIT-V-1.1 | `CriticProfile` on environment profile | **Critical** | CRIT-V-0 |
| 3 | CRIT-V-1.2 | CVL contracts (`CriticRequest`, `CriticVerdict`, …) | **Critical** | CRIT-V-1.1 |
| 4 | CRIT-V-1.3 | `EvaluatorLoopSpec` | High | CRIT-V-1.2 |
| 5 | CRIT-V-2.1 | `eval.judge` tool | **Critical** | CRIT-V-1.2, M-LLM-R |
| 6 | CRIT-V-2.2 | `eval.trajectory` tool | High | CRIT-V-2.1 |
| 7 | CRIT-V-2.3 | Registry observation hook for judge/trajectory | Medium | CRIT-V-2.1 |
| 8 | CRIT-V-3.1 | `CriticOrchestrator` | **Critical** | CRIT-V-2.1 |
| 9 | CRIT-V-3.2–3.3 | L0/L1 gateways | High | CRIT-V-3.1 |
| 10 | CRIT-V-3.4–3.5 | Graph partial + final hooks | High | CRIT-V-3.1 |
| 11 | CRIT-V-3.6 | Critic trace events | Medium | CRIT-V-3.4 |
| 12 | CRIT-V-4.1–4.2 | Evaluator-loop executor + graph wire | High | CRIT-V-3.4 |
| 13 | CRIT-V-5.1–5.2 | Semantic offline eval runner | Medium | CRIT-V-2.1 |
| 14 | CRIT-V-6.1–6.3 | Tier-3 critic wiring + policy + CI | High | CRIT-V-3.1 |
| 15 | CRIT-V-6.4 | Appendix W author map | Medium | CRIT-V-6.1 |
| 16 | CRIT-V-7.1 | FAUDIT-EVAL.1 baseline CI gate | High | CRIT-V-6.3 |
| 17 | CRIT-V-7.2–7.3 | Flow reference sync + lab demo | Medium | CRIT-V-3.6 |

### 6.2bb Phase ORCH execution order (Band 2j — closed 2026-06-05)

**Status:** **Done** · register: [Phase ORCH](.#phase-orch--orchestration-control-plane-closeout) · queue: [§6.1b](.#61b-harness-implementation-queue--orchestration-closeout-closed)

Work **one ORCH ID per PR**; after each step update the ORCH master table + §6.1b + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | Depends on |
|-------|-----|-------------|----------|------------|
| 1 | ORCH-1 | Planner/classifier kind registry + `nexus_factory` wiring | **Critical** | ORCH-DOC.* |
| 2 | ORCH-2 | `graph_spec_to_plan` + planning runner integration | High | ORCH-1 (shared factory path) |
| 3 | ORCH-3 | `max_parallel_nodes` on `OrchestrationProfile` + `GraphExecutor` | Medium | — (parallel OK after ORCH-1) |
| 4 | ORCH-4 | Docs closeout — Appendix I + plan §0.5 | Low | ORCH-1–3 |

### 6.2v Phase V-REM execution order (Band 2i — closed 2026-06-05)

**Status:** **Done** · register: [Phase V-REM](.#phase-v-rem--phase-v-runtime-remediation-audit-closeout) · queue: [§6.1z](.#61z-harness-implementation-queue-consolidated) (closed)

Work **one V-REM ID per PR**; after each step update the V-REM master table + Appendix J + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | Closes |
|-------|-----|-------------|----------|--------|
| 1 | V-REM-CG.1 | Fix per-application capability graph system edge mapping | **Critical** | V-CG.2–4 |
| 2 | V-REM-CG.2 | Re-validate lineage/impact/compatibility on corrected graph | High | V-CG.2–4 |
| 3 | V-REM-ALG.1 | Runtime filter for retired/deprecated agents | High | V-ALG.3 |
| 4 | V-REM-ALG.2 | Production-eligible + owner gate at agent selection | High | V-ALG.4 |
| 5 | V-REM-SEC.1 | Tool injection defense on main execution path | High | V-SEC.2 |
| 6 | V-REM-SEC.2 | Retrieval poisoning middleware per tenant/app | High | V-SEC.3 |
| 7 | V-REM-SEC.3 | Tenant isolation + audit trail in UnifiedTaskRunner/NexusLoop | High | V-SEC.4 |
| 8 | V-REM-PE.1 | PromptMeta owner/risk schema + validation | High | V-PE.1 |
| 9 | V-REM-PE.2 | YAML prompt assets catalog seed | Medium | V-PE.1 |
| 10 | V-REM-A.1 | NexusEvalRunner integration tests + gate | Medium | A.4, A.4.1 |

**Phase V-REM closeout:** **Done** (2026-06-05). Verified via `phase_v_closeout_gate.py --enforce --enforce-l4`.

### 6.2w Phase W-OPS execution order (Band 2d — complete 2026-06-06)

**Status:** **Done** · register: [Phase W-OPS](.#phase-w-ops--operational-harness-maturity-ideal-l3-ops)

Work **one W-OPS ID per PR**; after each step update the W-OPS table + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | IDEAL gap |
|-------|-----|-------------|----------|-----------|
| 1 | W-OPS.1 | Side-effect tool idempotency keys + dedup | **Critical** | Reliability §8.3 |
| 2 | W-OPS.2 | Integration circuit breaker (`_shared`) | **Critical** | Reliability §8.2 |
| 3 | W-OPS.3 | Long-running / checkpoint / retry gate tests | High | Reliability §8.3 |
| 4 | W-OPS.6 | `tenant_id` on TaskEnvelope → trace/events | High | Identity §3.2 |
| 5 | W-OPS.7 | Mandatory harness API key (staging profile) | High | Identity §3.2 |
| 6 | W-OPS.4 | SLO catalog + incident budget + runbooks | **Critical** | Observability §11 |
| 7 | W-OPS.5 | L3-ops evidence (2 release cycles) | **Critical** | §12.3 vs V-V6 CI |
| 8 | W-OPS.8 | `harness.*` platform skill packs | Medium | Capability §3.6 |
| 9 | W-OPS.9 | `requires_skills` shipped demo | Medium | Registries §19 |
| 10 | W-OPS.10 | Harness lab stable stack health (catalog slugs) | Medium | Capability §3.6 |
| 11 | W-OPS.11 | Online/shadow evaluation registry writes | Medium | Evaluation §18 |
| 12 | W-OPS.12 | W-ML Celery Tier-3 scale-out (optional) | Low | Modality §3.5.1 |
| 13 | W-OPS.13 | ToolsAgent removal roadmap | Low | Cognition hygiene |
| 14 | W-OPS.14 | Typed wiring (no `load_callable`) | Low | DX §22 |
| 15 | W-OPS.15 | Architecture metrics threshold enforcement | Low | §21.6 |

**Wave P0 (orders 1–7)** must be **Done** before declaring **operational IDEAL L3**. **Wave P1/P2** run in parallel with P0 when owners differ.

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new product applications, Problem Radar wave 2+.

### 6.2x Phase H-APP execution order (Band 2e — complete 2026-06-03)

**Status:** **Done** · canonical register: [Phase H-APP — Master deliverables register](.#h-app--master-deliverables-register-all-43-tasks) · audit narrative: [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) §7.

Work **one H-APP ID per PR**; after each step update the H-APP master table + paydown log; keep §6.1 scripts green.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| H0 | H-APP.0.1–H-APP.0.5 | 5 | Terminology, CI guards, `poc_template` getattr fix, manifest conformance |
| H1 | H-APP.1.1–H-APP.1.8 | 8 | `ApplicationEnvironmentProfile`, unified wiring, runtime bridge, LLM resolver |
| H2 | H-APP.2.1–H-APP.2.8 | 8 | Identity, policy DSL, execution modes, V-SEC per application |
| H3 | H-APP.3.1–H-APP.3.6 | 6 | Orchestration profile, graph spec, Nexus factory, shadow/sandbox |
| H4 | H-APP.4.1–H-APP.4.8 | 8 | Context, memory, reliability, observability profiles |
| H5 | H-APP.5.1–H-APP.5.5 | 5 | Migrate lab/legal/research/poc/docker_verify + scaffold |
| H6 | H-APP.6.1–H-APP.6.3 | 3 | Operational L3 sign-off (release cycles + CI + audit §4) |
| **Total** | | **43** | |

**Suggested PR order (same as Phase H-APP paydown):** H-APP.0.3 → H-APP.1.1–H-APP.1.4 → H-APP.1.5–H-APP.1.8 → H-APP.3.4–H-APP.3.5 → H-APP.2.1–H-APP.2.8 → H-APP.4.1–H-APP.4.8 → H-APP.3.1–H-APP.3.3 → H-APP.5.1–H-APP.5.5 → H-APP.0.1–H-APP.0.5 → H-APP.6.1–H-APP.6.3.

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new **product** Tier-3 apps, Problem Radar wave 2+, marketplace UI, catalog hot-reload.

### 6.2y Phase DX execution order (Band 2f — mostly done)

**Status:** **Done** (2026-06-02) · **47/47 Done** · canonical register: [Phase DX — Master deliverables register](.#dx--master-deliverables-register-all-47-tasks).

Work **one DX ID per PR**; after each step update the DX master table + paydown log; keep §6.1 scripts green. **Start with DX1 (scaffold/H-APP alignment)** before DX2 facades — otherwise new authors copy broken `factory.py` patterns.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| DX0 | DX-0.1–DX-0.4 | 4 | LangGraph mapping, responsibility matrix, progressive disclosure |
| DX1 | DX-1.1–DX-1.6 | 6 | **P0** — scaffold + poc/legal/research factories on H-APP path only |
| DX2 | DX-2.1–DX-2.6 | 6 | `HarnessApplication`, `AgentGraph`, `IntergraxAgent` + `@step` |
| DX3 | DX-3.1–DX-3.6 | 6 | `--minimal` stack, `intergrax run`, `doctor`, TTFRun acceptance |
| DX4 | DX-4.1–DX-4.4 | 4 | Integration presets + picker + gate tests |
| DX5 | DX-5.1–DX-5.8 | 8 | Host hooks, YAML loader, logging, event catalog, policy rule plugins |
| DX6 | DX-6.1–DX-6.5 | 5 | Tier-2 hygiene, external `intergrax init` template |
| DX7 | DX-7.1–DX-7.5 | 5 | JSON Schema + spec versioning + UI feed (Phase 2 prep) |
| DX8 | DX-8.1–DX-8.3 | 3 | `doctor --ci`, DX metrics artifact, scaffold alignment script |
| **Total** | | **47** | |

**Suggested PR order:** DX-1.1 → DX-1.2 → DX-1.3 → DX-1.6 → DX-8.3 → DX-2.1 → DX-2.2 → DX-2.3 → DX-2.5 → DX-3.1 → DX-3.2 → DX-3.5 → DX-3.6 → DX-4.1 → DX-4.4 → DX-1.4–DX-1.5 → DX-2.4 → DX-2.6 → DX-3.3–DX-3.4 → DX-5.1–DX-5.2 → DX-6.1–DX-6.2 → DX-4.2–DX-4.3 → DX-5.3–DX-5.8 → DX-6.3–DX-6.5 → DX-7.1–DX-7.5 → DX-8.1–DX-8.2 → DX-0.1–DX-0.4.

**Success gate for Phase DX full closeout:** All rows **Done** or **Won't fix**; DX-3.5 + DX-8.1 green in CI; DX-3.6 quickstart validated; DX-7.1 schemas under `build/harness_specs`. **Core path (DX1–DX2, DX3.2–3.3, DX8.3) already meets harness authoring needs.**

**Explicitly out of NOW:** K.1, K.2, visual environment builder UI, new product Tier-3 apps, Problem Radar wave 2+.

### 6.2z Phase AA execution order (Band 2g — mostly done)

**Status:** **Mostly Done** (2026-06-02) · platform **Done** · domain **Deferred** · canonical register: [Phase AA — Master deliverables register](.#aa--master-deliverables-register-all-tasks).

Work **one AA ID per PR/session**; after each step update the AA master table + paydown log + conformance matrix; keep §6.1 scripts green. **Legal:** follow **hard reset** policy (AA-LEG.0.1) — no incremental preservation of legacy pipeline code.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| AA0 | AA-0.1, AA-0.2, AA-S0.1–AA-S0.6, AA-LG.1, AA-APP.0.1–AA-APP.0.3 | 12 | Scaffold checklist, tier guards, deploy triad standard |
| AA1 | AA-D0.1–AA-D0.7 | 7 | README, guides, TIER3_READINESS, USAGE |
| AA2 | AA-LEG.0.2–AA-LEG.3.1 | 12 | **Legal agent hard reset** |
| AA3 | AA-LEGAPP.1–AA-LEGAPP.8 | 8 | `legal_application` + deploy triad |
| AA4 | AA-ECHO.1–AA-ECHO.5 | 5 | Reference echo agent |
| AA5 | AA-SIG.1–AA-SIG.3 | 3 | Signoff probe |
| AA6 | AA-PR.1–AA-PR.5 | 5 | Problem radar (docs/hygiene; frozen feature) |
| AA7 | AA-ORG.1–AA-ORG.5 | 5 | Organization worker |
| AA8 | AA-RES.1–AA-RES.6 | 6 | Research agents |
| AA9 | AA-LABAG.1–AA-LABAG.2 | 2 | Lab mocks |
| AA10 | AA-LABAPP.1–AA-LABAPP.7 | 7 | Lab application host |
| AA11 | AA-POC.1–AA-POC.5 | 5 | POC template (canonical shell) |
| AA12 | AA-RESAPP.1–AA-RESAPP.6 | 6 | Research application host |
| **Total** | | **83** | |

**Suggested PR order:** AA-S0.2 → AA-S0.5 → AA-APP.0.1 → AA-APP.0.3 → AA-POC.1 → AA-POC.2 → AA-LABAPP.2 → AA-ECHO.2 → AA-LEG.0.3 → AA-LEG.1.1 → AA-LEG.1.2 → AA-LEG.1.3 → AA-LEG.2.1 → AA-LEG.2.2 → … → AA-LEGAPP.1–AA-LEGAPP.6 → AA-D0.1 → AA-D0.3–AA-D0.5 → AA-RESAPP.* → AA-LABAPP.1 → AA-APP.0.2 → remaining ARCHITECTURE.md rows.

**Per-application deploy triad gate (AA-APP.0.2):** for each of `lab_application`, `legal_application`, `local_workspace_application`, `poc_template_application`, `research_application` assert:

1. `docker/Dockerfile` + `docker-compose.yml` + `build-docker.sh` / `.bat`
2. `BUILD_AND_DEPLOY.md` present and matches scaffold generator output (or documented drift)
3. `ARCHITECTURE.md` § **Dependencies** lists required `pyproject.toml` extras (e.g. `harness-author`, provider-specific `llm-*`, `dev-ci` for tests)

**Doc pair gate (AA-D0.6):** for each listed Tier-2 agent and Tier-3 application assert `ARCHITECTURE.md` and `IMPLEMENTATION_PLAN.md` exist and cross-link. Gate: `tests/unit/applications/test_agent_app_doc_pair.py`.

**Success gate for Phase AA platform closeout:** **Met** (2026-06-02) — conformance matrix **OK**; legal tree = scaffold; `lab_application` on `build_harness_host_runtime`; AA-APP.0.2 green; gate **533**. **Full AA register closeout** additionally requires Band 3 domain rows **Done** or explicitly **Deferred** (current policy: **Deferred**).

**Explicitly out of NOW:** K.1/K.2 implementation, Legal **live LLM** E2E (Band 3), new product hosts beyond the four listed, Legal UAEP step port (AA-LEG.2.2+) unless product reprioritizes §6.3.

### 6.2aa Phase MEM execution order (Band 2h — closed)

**Status:** **Done** (2026-06-02) · **48/48 Done** · canonical register: [Phase MEM — Master deliverables register](.#mem--master-deliverables-register-all-48-tasks).

Work **one MEM ID per PR**; after each step update the MEM master table + paydown log; keep §6.1 scripts green. **Start with MEM-1.*** before MEM-3/MEM-7 — bridge must exist before plugins/hooks.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| MEM0 | MEM-0.1–MEM-0.4, MEM-PAR.1, MEM-CHk.1, MEM-PERS.1, MEM-ST.1, MEM-OBS.2 | 9 | Register, audit baseline, parity tables (MEM-OBS.2 Done) |
| MEM1 | MEM-1.1–MEM-1.6, MEM-2.1–MEM-2.3 | 9 | **P0** — H-APP bridge + SQLite user LTM |
| MEM2 | MEM-3.*, MEM-4.*, MEM-5.*, MEM-CTX.1, MEM-DOC.1–6, MEM-GRAPH.1, MEM-TASK.* | 18 | **P1** — gates, plugins, context docs |
| MEM3 | MEM-6.*, MEM-7.*, MEM-OBS.1, MEM-DOC.5, MEM-CTX.2, MEM-PERS.2, MEM-ST.4 | 9 | **P2** — retention, hooks, metrics |
| MEM4 | MEM-8.*, MEM-9.1, MEM-PERS.3 | 4 | **P3** — product RFCs |
| **Total** | | **48** | |

**Success gate:** P0 + P1 **Done**; H-APP.4.3 **Done**; user LTM durable on sqlite lab profile; `MemoryProfile` drives all reference hosts.

**Explicitly out of NOW:** K.1/K.2, Mem0 auto-ingest ship (MEM-8.2), entity graph implementation (MEM-9.1 beyond RFC).

### 6.2ab Phase MEM-DEPTH execution order (Band 2am — closed)

**Status:** **Done** (2026-06-08) · **26/26 Done** · canonical register: [Phase MEM-DEPTH — Master deliverables register](.#mem-depth--master-deliverables-register-all-26-tasks).

Work **one MEM-DEPTH ID per PR**; after each step update the MEM-DEPTH master table + paydown log; keep §6.1 scripts green. **Start with MEM-DEPTH-0.3 (ADR) then MEM-DEPTH-1.*** — architecture decision before Context Compiler code.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| MEMD0 | MEM-DEPTH-0.1–0.4 | 4 | Canon doc, plan register, ADR, cross-links |
| MEMD1 | MEM-DEPTH-1.1–1.6 | 6 | **P0** — Context Compiler + never-overflow |
| MEMD2 | MEM-DEPTH-2.1–2.2 | 2 | **P1** — Mongo session persistence parity |
| MEMD3 | MEM-DEPTH-3.1–3.5 | 5 | **P1** — Lifecycle automation |
| MEMD4 | MEM-DEPTH-4.1–4.3 | 3 | **P1** — Explore / discovery |
| MEMD5 | MEM-DEPTH-5.1–5.6 | 6 | **P2/P3** — Entity graph, temporal validity, quality gates |
| **Total** | | **26** | |

**Success gate:** P0 + P1 **Done**; never-overflow acceptance green; Memory Layer **L3+** on FAUDIT re-run.

**Explicitly out of NOW:** K.1/K.2, Mem0 SaaS, entity graph ship without §6.3 decision (MEM-DEPTH-5.1).

### 6.1p Phase P-Ext paydown (Band 2c — optional parallel with §6.1)

**Status:** **Done** (2026-06-02) · closure complete; extend catalogs via Appendix I + author guide.

| Order | ID | Deliverable | Priority |
|-------|-----|-------------|----------|
| 1 | P-Ext.0.5 | Fixture pip package (`tests/fixtures/plugin_packages`) | P0 |
| 2 | P-Ext.0.6 | EP discovery tests (all three groups) | P0 |
| 3 | P-Ext.1.6 | Integration EP test via fixture | P0 |
| 4 | P-Ext.1.10 | Tier-3 `integration_wiring` → `bootstrap_catalogs()` | P0 |
| 5 | P-Ext.2.9–2.11 | External tool example + unit + EP tests | P0 |
| 6 | P-Ext.3.6–3.8 | External skill example + unit + EP tests | P0 |
| 7 | P-Ext.0.7 | `INTERGRAX_DISCOVER_PLUGINS` + lab wiring | P1 |
| 8 | P-Ext.4.3, 4.5, 1.8 | Conflict policy + CI smoke (incl. integration counts) | P1 |
| 9 | P-Ext.1.5, 1.7, 5.5–5.6 | Slug/docs cleanup + author guide matrix | P2 |
| 10 | P-Ext.2.12, 3.9–3.11 | Tool/skill lazy bootstrap, scaffold plugin template, importer docs | P2 |
| 11 | P-Ext.1.3a, 1.4, 1.9, 1.11–1.12 | Typed resolve expansion, health API, integration wiring helper | P3 |
| 12 | P-Ext.5.1, 3.10, 3.12 | Scaffold CLI (all three catalogs) + harness `requires_skills` demo | P3 |

Full task register: [Appendix I](.#appendix-i--plugin-catalog-traceability-phase-p-ext).

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3. **Feature queues:** Phase W-ADAPT — §6.1t; Phase M-LLM-R — §6.1v; Phase M.6 P4 — §6.1w (closed); Phase M.6 P5 — §6.1x (closed); Phase M.6 P6 — §6.1y (closed).

### 6.2ag Phase M.6 P6 execution order (Band 2ac — Done)

**Status:** **Done** (2026-06-02) · register: [M.6 P6](.#m6-p6--harness-integration-expansion-planned) · queue: [§6.1y](.#61y-harness-implementation-queue--integration-expansion-m6-p6-done)

```text
Wave H-INT-0 (categories):  M-P6-CAT.1 → M-P6-CAT.2 → M-P6-CAT.3 → M-P6-CAT.4 → M-P6-CAT.5 → M-P6-CAT.6 → M-P6-CAT.7 → M-P6-CAT.8 → M-P6-CAT.9
Wave H-INT-10 (security):   M-P6.1 → M-P6.2 → M-P6.3 → M-P6.4
Wave H-INT-11 (sandbox):    M-P6.5 → M-P6.6 → M-P6.7
Wave H-INT-12 (identity):   M-P6.8 → M-P6.9 → M-P6.10
Wave H-INT-13 (gitops CI):  M-P6.11 → M-P6.12 → M-P6.13
Wave H-INT-14 (speech):     M-P6.14 → M-P6.15
Wave H-INT-15 (enterprise): M-P6.16 → M-P6.17 → M-P6.18 → M-P6.19
Wave H-INT-16 (data/wf):    M-P6.20 → M-P6.21 → M-P6.22 → M-P6.23 → M-P6.24
Wave H-INT-17 (reserve):    M-P6.25 → M-P6.26 → M-P6.27 → M-P6.28 → M-P6.29 → M-P6.30 → M-P6.31 → M-P6.32
Wave PRE (presets):         M-P6-PRE.1  (after H-INT-10 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P5 **Done**; M-P5.FU wiring **Done**; Phase SEC closeout **Done** (V-SEC patterns for `security_scanner`).  
**Parallelism:** H-INT-10 unblocks STABLE promote gate; H-INT-11 unblocks cloud `sandbox.exec`; H-INT-12 unblocks multi-tenant hosts; H-INT-14 unifies speech catalog.  
**Closeout target:** catalog **167** slugs; optional `HARNESS_M6_P6_PROBE_SLUGS` + four Tier-3 presets; gate green.

### 6.2af Phase M.6 P5 execution order (Band 2ab — Planned)

**Status:** **Done** (2026-06-02) · register: [M.6 P5](.#m6-p5--harness-integration-depth-done--3334) · queue: [§6.1x](.#61x-harness-implementation-queue--integration-depth-m6-p5-done)

```text
Wave H-INT-0 (categories):  M-P5-CAT.1 → M-P5-CAT.2 → M-P5-CAT.3
Wave H-INT-6 (ops/CI):      M-P5.1 → M-P5.2 → M-P5.3 → M-P5.4 → M-P5.5 → M-P5.6 → M-P5.7 → M-P5.8 → M-P5.9 → M-P5.10
Wave H-INT-7 (eval/async):  M-P5.11 → M-P5.12 → M-P5.13 → M-P5.14 → M-P5.15 → M-P5.16 → M-P5.17 → M-P5.18 → M-P5.19 → M-P5.20
Wave H-INT-8 (data lab):    M-P5.21 → M-P5.22 → M-P5.23 → M-P5.24 → M-P5.25 → M-P5.26 → M-P5.27 → M-P5.28
Wave H-INT-9 (P2 reserve):  M-P5.29 → M-P5.30 → M-P5.31 → M-P5.32 → M-P5.33 → M-P5.34
Wave PRE (presets):         M-P5-PRE.1  (after H-INT-6 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P4 **Done**; M-P4.FU wiring **Done**; Phase INT closeout **Done** (health probe patterns).  
**Parallelism:** H-INT-6 unblocks W-OPS metrics + multi-CI; H-INT-7 unblocks EVAL/W-ADAPT; H-INT-8 is lab-only.  
**Closeout target:** catalog **136** slugs; `HARNESS_M6_P5_PROBE_SLUGS` + four Tier-3 presets; gate green.

### 6.2ae Phase M.6 P4 execution order (Band 2aa — Done)

**Status:** **Done** (2026-06-02) · register: [M.6 P4](.#m6-p4--harness-platform-expansion-done) · queue: [§6.1w](.#61w-harness-implementation-queue--integration-expansion-m6-p4-closed)

```text
Wave H-INT-0 (categories):  M-P4-CAT.1 → M-P4-CAT.2  (before first slug in new category)
Wave H-INT-1 (storage):     M-P4.1 → M-P4.2 → M-P4.3 → M-P4.4
Wave H-INT-2 (obs stack):   M-P4.5 → M-P4.6 → M-P4.7
Wave H-INT-3 (secrets):     M-P4.8 → M-P4.9 → M-P4.10 → M-P4.11
Wave H-INT-4 (control):     M-P4.12 → M-P4.13 → M-P4.14 → M-P4.15 → M-P4.16
Wave H-INT-5 (enterprise):  M-P4.17 → M-P4.18 → M-P4.19 → M-P4.20 → M-P4.21 → M-P4.22 → M-P4.23 → M-P4.24 → M-P4.25 → M-P4.26 → M-P4.27 → M-P4.28
```

**Prerequisites:** Phase M core + M.6 P1/P2/P3 **Done**; Phase INT closeout **Done** (health probe patterns).  
**Parallelism:** Any wave after H-INT-0 may start when a slug is needed — prefer H-INT-1 → H-INT-2 → H-INT-3 order for W-OPS/adaptive unblock.  
**Closeout:** **Done** — catalog **127** in `layout.py`; `tests/unit/integrations/providers/test_p5_m6_p4_providers.py` (42 tests).

### 6.2ad Phase M-LLM-R execution order (Band 2z — closed 2026-06-06)

**Status:** **Done** · register: [Phase M-LLM-R](.#phase-m-llm-r--llm-completion-response-envelope-audit-2026-06-06) · queue: [§6.1v](.#61v-harness-implementation-queue--llm-completion-response-envelope-closed)

```text
Wave M-LLM-R-0 (planning):     M-LLM-R.0.2 → 0.3  (0.1 **Done**)
Wave M-LLM-R-1 (contracts):    M-LLM-R.1.1 → 1.8
Wave M-LLM-R-2 (ABC):          M-LLM-R.2.6 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5
Wave M-LLM-R-3 (providers):    M-LLM-R.3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7
Wave M-LLM-R-4 (Nexus):        M-LLM-R.4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6
Wave M-LLM-R-5 (RAG/web):      M-LLM-R.5.1 → 5.2 → 5.3
Wave M-LLM-R-6 (agents):       M-LLM-R.6.1 → 6.2 → 6.3 → 6.4
Wave M-LLM-R-7 (obs/replay):   M-LLM-R.7.1 → 7.2 → 7.3 → 7.4 → 7.5
Wave M-LLM-R-8 (closeout):     M-LLM-R.8.1 → 8.2 → 8.3 → 8.4
```

**Prerequisites:** Phase M-LLM **Done** (M-LLM.1–13); no dependency on W-ADAPT runtime L4 gate.

**Parallelism:** May run alongside W-ADAPT-5+; coordinate M-LLM-R.7.5 with W-ADAPT signal work if both touch `signal_collector.py`.

**Closeout gate:** `scripts/maintenance/check_llm_adapter_typed_returns.py` + `scripts/maintenance/check_agents_llm_adapter_response.py` + full `tests/unit/llm_adapters` gate green (M-LLM-R.8.3, M-LLM-R.6.4).

### 6.2ac Phase W-ADAPT execution order (Band 2y — closed)

**Status:** **Done** (2026-06-02) · register: [Phase W-ADAPT](.#phase-w-adapt--adaptive-harness-intelligence-l4-runtime) · queue: [§6.1t](.#61t-harness-implementation-queue--adaptive-harness-intelligence-closed)

```text
Wave W-ADAPT-0 (planning):        W-ADAPT-0.2 → 0.3 → 0.4 → 0.5  (**Done**)
Wave W-ADAPT-1 (observe L4-O):    W-ADAPT-1.1 → 1.12  (**Done**)
Wave W-ADAPT-2 (recommend L4-R):  W-ADAPT-2.1 → 2.12  (**Done**)
Wave W-ADAPT-3 (shadow L4-S):      W-ADAPT-3.1 → 3.2 → 3.3 → 3.4 → 3.6 → 3.7 → 3.5  (**Done**)
Wave W-ADAPT-4 (apply L4-A):       W-ADAPT-4.1 → 4.10  (**Done**)
Wave W-ADAPT-5 (verify L4-V):      W-ADAPT-5.1 → 5.3 → 5.4 → 5.5 → 5.2 → 5.11 → 5.6 → 5.7 → 5.8 → 5.9 → 5.10 → 5.12  (**Done**)
Wave W-ADAPT-6 (patterns):         W-ADAPT-6.2 → 6.1 → 6.3 → 6.5 → 6.4  (**Done**)
Wave W-ADAPT-7 (Tier-3 + docs):    W-ADAPT-7.1 → 7.2 → 7.3 → 7.4 → 7.5 → 7.6 → 7.7  (**Done**)
```

**Prerequisites:** Phase V + V-REM + W-OPS + EVAL + COST + CG closeouts **Done**.

**Runtime L4 gate:** `uv run python scripts/release/phase_w_adapt_closeout_gate.py --enforce-l4-runtime` (added in W-ADAPT-5.6).

### 6.2 Harness architecture hardening (Band 2 — Phase V) — Done

**Status:** **Done** (2026-06-05) — Phase V contracts + V-REM runtime enforcement complete. Closeout: `phase_v_closeout_gate.py --enforce --enforce-l4`.

| Item | Status | Notes |
|------|--------|-------|
| V-CG … V-V6 | **Done** | Governance + CI + runtime enforcement |
| V-REM | **Done** | 10/10 closed — §6.1z queue closed |
| W-ML | **Done** | [architecture/MODALITY.md](architecture/MODALITY.md) |
| P-Ext | **Done** | Appendix I |
| M.6 P5 / M.6 P6 / R-Skill expansion | **On demand** | W-OPS.10, W-OPS.8, §6.1x, §6.1y |

**Forbidden in Band 2/2d:** K.1, K.2, product-specific skills, new product application hosts.

### 6.3 End of plan — deferred product work only (Band 3)

**This section is the last band in the implementation plan.** Nothing here is the default “next step” after harness work.

| ID | Deliverable | Status | Gate to start |
|----|-------------|--------|----------------|
| K.1 | Problem Radar prototype | **Deferred** | Explicit product decision + [Appendix A](.#appendix-a--agent-operating-system-certification-checklist) |
| K.2 | Vendor Discovery prototype | **Deferred** | Same as K.1 |
| K.6 / B.15 / S-Ops.4 | Legal live LLM E2E | **Deferred** | Product/CI budget decision |
| `agents/legal` UAEP domain steps | Scaffold shell **Done** (Band 2g); step port **Deferred** | **Business** | [§6.3a](.#63a-business-backlog-register-consolidated) AA-LEG.2.2+ |
| Tier-3 product apps | New `applications/<product>` beyond lab + reference hosts | **Deferred** | Product decision; scaffold exists (Phase N **Done**) |
| Domain skills | Product agent skill packs (non-`harness.*`) | **Deferred** | With K.1 or K.2 |
| `agents/problem_radar` | Wave 1 scaffold frozen | **Deferred** | Do not extend until K.1 reprioritized |

**When Band 3 may start:** Record the decision in this plan (date + chosen K.1 vs K.2), then follow [guides/AGENT_CREATION_GUIDE.md](guides/AGENT_CREATION_GUIDE.md). Tier-3 scaffold reference (Phase N) applies **only after** that decision — not as ongoing harness work.

**Tier-3 scaffold (for when Band 3 is approved):**

```bash
python -m intergrax.scaffold new-stack <slug> --profile lab --capability <slug>.basic
```

See [`applications/TIER3_READINESS.md`](../../../../../applications/TIER3_READINESS.md). Existing hosts (`lab_application`, `legal_application`, `research_application`, `poc_template_application`) are sufficient for **all harness** work. **Product:** [`local_workspace_application`](../../../../../applications/local_workspace_application) — Local Knowledge Workspace (LKW) — first business environment after harness GA; see [ARCHITECTURE.md](../../../../../applications/local_workspace_application/docs/ARCHITECTURE.md).

### 6.3a Business backlog register (consolidated)

**Single register for Band 3 and AA domain-deferred rows.** Do not duplicate in harness session summaries.

| ID | Deliverable | Module | Priority | Depends on |
|----|-------------|--------|----------|------------|
| **LKW.0** | Local Knowledge Workspace — scaffold + architecture baseline | `agents/local_{indexer,search,synthesizer}`, `applications/local_workspace_application` | **High** | Product reprioritization (2026-06-07) — **Done** |
| **LKW.1** | Wave 1 — ingest + search smoke on explicit paths | `agents/local_*/steps` | **High** | LKW.0 |
| **LKW.2** | Multi-agent pipeline (`local.workspace.pipeline` graph) | `local_workspace_application` + Nexus graph | High | LKW.1 |
| **LKW.3** | Tier-0 `filesystem.*` read tools + allowlist policy | `intergrax/tools/providers/filesystem` | Medium | LKW.1 |
| **LKW.4** | Background ingest queue + incremental index | Tier-0 queue + Tier-3 worker | Medium | LKW.2 |
| **LKW.5** | `LKW_DATA_HOME` + Chroma persistent local index | `local_workspace_application/host/settings.py` | High | LKW.1 |
| **LKW.6** | Local OS daemon (Win/Linux/macOS) + interaction intake on host | `local_workspace_application` | High | LKW.1 |
| **LKW.6b** | Slack Socket Mode + slash command → Nexus (interaction surface) | Tier-3 + `slack` integration | Medium | LKW.6 |
| **LKW.7** | Background file watcher + incremental index + optional Slack notify | Tier-0 queue + Tier-3 worker | Medium | LKW.3 |
| **LKW.8** | Tray / file-picker UI (localhost HTTP/MCP client) | Product (out of harness) | Low | LKW.6 |
| **DSW.0** | Dispute Simulation Workspace — scaffold + architecture baseline | `agents/dispute_{intake,analyst,strategist,scenario}`, `applications/dispute_sim_application` | **High** | Product reprioritization (2026-06-07) — **Done** |
| **DSW.1** | Wave 1 — case intake + RAG ingest + timeline artifact | `agents/dispute_intake/steps` | **High** | DSW.0 |
| **DSW.2** | Multi-agent pipeline (`dispute.pipeline` graph) | `dispute_sim_application` + Nexus graph | High | DSW.1 |
| **DSW.3** | Analyst matrix + strategist brief domain steps | `agents/dispute_analyst`, `agents/dispute_strategist` | High | DSW.1 |
| **DSW.4** | Scenario variants + correspondence review + HITL | `agents/dispute_scenario` | High | DSW.3 |
| **DSW.5** | Optional subgraph to `legal.review` for clause drill-down | Nexus graph | Medium | DSW.3 |
| **DSW.6** | Case persistence + retention policy | `dispute_sim_application/host/settings.py` | Medium | DSW.1 |
| **DSW.7** | Polish dispute eval fixtures + regression | `tests` / agent eval | Medium | DSW.4 |
| **K.1** | Problem Radar prototype (wave 2+) | `agents/problem_radar` | Product | Explicit reprioritization |
| **K.2** | Vendor Discovery prototype | (greenfield) | Product | K.1 decision or parallel product call |
| **AA-LEG.2.2** | Legal UAEP steps (one step per PR from `SPEC_FROM_LEGACY.md`) | `agents/legal/steps` | High | Product/legal owner |
| **AA-LEG.2.3** | Remove any parallel legal runtime (Nexus gateway only) | `agents/legal` | High | AA-LEG.2.2 |
| **AA-LEG.2.4** | Legal agent tests per ported step | `agents/legal/tests` | High | AA-LEG.2.2 |
| **AA-LEGAPP.6** | `legal_application` host smoke on real steps | `legal_tests` | High | AA-LEG.2.2 |
| **AA-LEGAPP.8** | Consolidate duplicate legal test trees | `legal_tests` vs agent tests | Low | AA-LEG.2.4 |
| **AA-RES.4** | Research skill ids on contracts | `agents/research` | Medium | Product |
| **AA-RES.5** | Research UAEP + graph delegation tests | `agents/research/tests` | High | Product |
| **AA-RESAPP.6** | Research application smoke + manifest wiring | `research_application_tests` | High | AA-RES.5 |
| **AA-ORG.3** | Organization worker scaffold-align (`contract`, `steps`) | `agents/organization_worker` | Medium | Harness demo |
| **AA-ORG.4** | Lab manifest flag + integration test | `lab_application/manifest.py` | Medium | AA-ORG.3 |
| ~~AA-LABAPP.6~~ | ~~Extra lab host smoke~~ | — | — | **Done** (2026-06-02) — not in business queue |
| **K.6 / B.15 / S-Ops.4** | Legal full E2E with live LLM | CI / acceptance | Low | CI budget approval |
| **Tier-3 product** | New `applications/<product>` beyond four reference hosts | `applications` | Product | Phase N scaffold + §6.3 decision |
| **Domain skills** | Non-`harness.*` skill packs for product agents | `intergrax/skills/providers` | Product | With K.1 or K.2 |
| **A.5** | Full Legal regression (all steps, live model) | Phase A row | Low | K.6 / B.15 |
| **Phase E** | Legal agent refactoring (parallel track) | `agents/legal` | On demand | Product architecture |

**Not business (infrastructure — closed; see [§6.1z](.#61z-harness-implementation-queue-consolidated)):** DX-5.7, AA-LEG.0.2, OPS-L3.1 **Done**; ongoing **§6.1** maintenance only.

### 6.1u Archived — Phase U cadence (complete 2026-06-01)

Security, policy, contracts, typing (U-Sec through U-CI). See Phase U definition of done. Residual U-Leg.* moved to §4.1 — not reopened as a new phase.

### 6.1s Archived — Phase S cadence (complete 2026-06-01)

See Phase S definition of done and Appendix F. Do not reopen S.* unless regression (fix under T.* or U.*).

### 6.1a Archived — Phase Q cadence (complete 2026-06-01)

Phase Q used **one Q.* deliverable per PR** → update Appendix C + paydown log. See Appendix C for Waves 1–9 and gate **417** at close. Do not reopen Q.* unless regression found (residual hardening → Appendix D).

### 6.1b Phase N (complete)

Tier-3 scaffold cadence remains the reference for new applications (`new-stack`); lab defaults include RAG/websearch tools and legal + research skill bundles.

### 6.4 Historical gate milestones (archived)

Phases F–L, J, Q, Q+, R, S, T, U, and §4.1 are **Done**. Gate milestones: **417** (Phase Q), **481** (harness completion, 2026-06-02). Phase tables: §2–§3; paydown: Appendices C–G.

> **Note:** Older phase closers said “next: Phase K (K.1/K.2).” That meant harness prerequisites were met, **not** that product work becomes the default implementation queue. **Current rule:** §4.0 Band 3 / §6.3 only after explicit product prioritization.

### D.2 Debug API (Done)

Standalone laboratory server:

```bash
uv run uvicorn intergrax.debug.app:create_debug_app --factory --host 127.0.0.1 --port 8099
```

Endpoints (mirror CLI):

```text
GET /debug/tasks?tenant=t1&limit=20
GET /debug/tasks/{run_id}?tenant=t1
GET /debug/tasks/{run_id}/trace?tenant=t1&include_runtime=true
```

Mount on an existing app:

```python
from intergrax.debug.router import create_debug_router

app.include_router(create_debug_router(db_path=Path("build/intergrax_trace.db")))
```

Environment: `INTERGRAX_TRACE_DB` (same as CLI).

### D.3 Experiment registry (Done)

SQLite registry at `build/intergrax_experiments.db` (`INTERGRAX_EXPERIMENTS_DB`).

```bash
python -m intergrax.debug experiments register --hypothesis "..." --capability echo.basic
python -m intergrax.debug experiments link-run EXPERIMENT_ID RUN_ID
python -m intergrax.debug experiments decide EXPERIMENT_ID --decision keep
python -m intergrax.debug experiments list --decision pending
```

HTTP: `GET/POST /debug/experiments`, `POST /debug/experiments/{id}/decision`, `POST /debug/experiments/{id}/runs/{run_id}`.

### D.4 Experiment workflow API (Done)

Platform-level `notebooks` was **removed** (2026-06-12). §35 workflow: `intergrax.experiments.workflow.ExperimentSession`; tests in `tests/unit/experiments`.

```python
from pathlib import Path
from intergrax.experiments.workflow import ExperimentSession, ensure_repo_root_on_path

ensure_repo_root_on_path()
session = ExperimentSession(trace_db=Path("build/experiments/trace.db"))
```

### D.5 Cost in trace (Done)

`AgentExecutionResult.cost` and `duration_seconds` are derived from LLM usage (`intergrax/contracts/runtime_cost.py`):

- Mapping: `runtime_answer_to_agent_result()` reads `llm_usage_report` or `stats.extra.cost`
- NexusLoop: aggregates multi-agent cost into task metadata (`execution_cost`) and `RunStats.llm_usage` on finalize
- Debug API/CLI: `stats.cost` on run detail; CLI `tasks show` prints cost line

Cost proxy: **1 cost unit = 1 LLM token** (laboratory default, matches EvalRunner).

### F.1 Shadow workspace (Done)

Isolated temporary filesystem for experiments (§20). Enable on a Nexus task:

```python
task = Task(
    tenant_id="t1",
    user_id="u1",
    message="analyze vendor",
    context=TaskContext(capability="research.web_search"),
    metadata={"shadow_workspace": True},  # optional: "shadow_workspace_cleanup": True
)
```

UAEP agents receive `ctx.metadata["shadow_workspace"]` in `run_step`. Result metadata includes `shadow_workspace_id`.

Root directory: `INTERGRAX_SHADOW_ROOT` (default `build/shadow_workspaces`).

### F.2 Sandbox runtime (Done)

Controlled session for risky tool use (§21). Enable on a Nexus task:

```python
task = Task(..., metadata={"sandbox": True})
```

Agents invoke allowlisted operations through the tool gateway:

```python
await ctx.invoke_tool(ToolRequest(
    tool_name="sandbox.exec",
    agent_id=ctx.agent_id,
    input={"operation": "write_file", "payload": {"path": "out.txt", "content": "..."}},
))
```

Operations: `echo`, `write_file`, `read_file`, `list_files`. Root: `INTERGRAX_SANDBOX_ROOT` (default `build/sandbox_sessions`).

### F.3 Advanced HITL (Done)

Human responses beyond approve:

```python
# Re-submit paused task with verdict
task = Task(..., task_id=original_task_id, metadata={"human_response": "reject"})
# or "approve" / "escalate"
```

- **reject** → task `FAILED`, decision persisted
- **escalate** → `INTERRUPT_ESCALATED` event, escalation chain in metadata, stays `WAITING_FOR_HUMAN`
- Store: `INTERGRAX_HUMAN_DECISIONS_DB` (default `build/intergrax_human_decisions.db`)

Optional on `NexusLoop`: `human_decision_store=SQLiteHumanDecisionStore(...)`.

### F.4 Long-running tasks (Done)

Enable durable pause/resume on Nexus tasks (§26):

```python
from intergrax.runtime.task import Task, TaskExecutionOptions, TaskLongRunningOptions

task = Task(
    tenant_id="t1",
    user_id="u1",
    message="monitor vendors for 30 days",
    context=TaskContext(capability="hitl.basic"),
    options=TaskExecutionOptions(
        long_running=TaskLongRunningOptions(
            enabled=True,
            notify_channel="slack",  # or "teams" / "log"
        ),
    ),
)
```

On pause (`WAITING_FOR_HUMAN`), NexusLoop persists a checkpoint with `resume_token` in result metadata.

Resume with the same `task_id` and token:

```python
Task(
    ...,
    task_id=original_task_id,
    options=TaskExecutionOptions(
        long_running=TaskLongRunningOptions(enabled=True, resume_token=token),
    ),
    metadata={"human_approved": True, "resume_token": token},
)
```

Optional on `NexusLoop`: `checkpoint_store=SQLiteTaskCheckpointStore(...)`, `notification_adapter=LoggingNotificationAdapter()`.

Env:

- `INTERGRAX_TASK_CHECKPOINTS_DB` (default `build/intergrax_task_checkpoints.db`)
- `INTERGRAX_RUNTIME_EVENTS_DB` (optional; enables SQLite runtime events in NexusLoop / debug API)
- `INTERGRAX_TASK_MEMORY_DB` (optional; TaskMemory SQLite path for lab / debug)
- `INTERGRAX_SLACK_WEBHOOK_URL` / `INTERGRAX_TEAMS_WEBHOOK_URL` (stub adapters; no network unless configured)

### H.6 Organization Worker lab runbook (Done)

Reference flow for §38 — virtual worker via Slack / Teams without orchestration in adapters.

**Agent:** `agents/organization_worker` — capability `org.vendor_report`.

**Lab app factory:**

```python
from intergrax.lab import create_organization_worker_lab_app

app = create_organization_worker_lab_app()  # pre-wired registry + HITL intake enricher
```

**HTTP (debug API):**

```bash
uv run uvicorn intergrax.lab.organization_worker:create_organization_worker_lab_app --factory --host 127.0.0.1 --port 8099
```

1. **Intake + execute** (Slack-shaped slash command):

```bash
curl -s -X POST "http://127.0.0.1:8099/debug/interactions/intake?execute=true&tenant=T1" \
  -H "Content-Type: application/json" \
  -d '{"command":"/intergrax","text":"org.vendor_report Acme Corp Q1","user_id":"U1","team_id":"T1"}'
```

Response includes `state: waiting_for_human`, `resume_token`, HITL notification on configured channel (`slack` / `teams` / `log`).

2. **Resume after approval:**

```bash
curl -s -X POST "http://127.0.0.1:8099/debug/tasks/{task_id}/human-response?tenant=T1" \
  -H "Content-Type: application/json" \
  -d '{"response":"approve","resume_token":"<token from intake>"}'
```

Teams intake uses the same endpoints with Bot Framework activity JSON (`channelId: msteams`).

**Registry helper:** `build_organization_worker_registry()` in `intergrax.runtime.registry`.

**Tests:** `tests/integration/debug/test_organization_worker_demo.py` (gate).

### D.1 Debug CLI (Done)



```bash

python -m intergrax.debug tasks list --tenant t1 --limit 20

python -m intergrax.debug tasks show RUN_ID --tenant t1

python -m intergrax.debug tasks trace RUN_ID --tenant t1

python -m intergrax.debug tasks trace RUN_ID --tenant t1 --format json --runtime

python -m intergrax.debug --db path/to/trace.db tasks list

```



Reuse:



- `SQLiteRunTraceStore` / `RunTraceReader` — `intergrax/runtime/nexus/tracing`

- `trace_bridge` — `intergrax/runtime/events/trace_bridge.py`

- `NexusLoop.event_bus` — in-process runs (not persisted; CLI uses SQLite trace)



---

# Audit Result: Critic & Verification Layer (CVL)

**Audit date:** 2026-06-13  
**Method:** Layer Completion Mode vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` §18 · `INTEGRAX_HARNESS_AUDIT_MAP.md` §25 · code `runtime/critic` · `tools/providers/eval` · Tier-3 critic wiring
**Verdict:** **CRIT-V-0…7 + FOLLOWUP Done** — domain **CRITIC_VERIFICATION** closed at **L3+** (Architecturally Mature).

---
