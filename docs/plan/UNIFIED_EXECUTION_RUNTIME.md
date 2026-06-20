# Unified Execution Runtime — Implementation Plan

**Architecture (1:1):** [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/plan/` satellites on demand).

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/UNIFIED_EXECUTION_RUNTIME_appendices.md`](plan/UNIFIED_EXECUTION_RUNTIME_appendices.md) | appendices |

> **Cursor context budget:** read this hub + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.2–3.3, §23–§24 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-4.1 | §4 Identity | Cryptographic signing / audit-protect for critical actions | P2 | **Done** |
| AUDIT-IDEAL-4.2 | §4 Identity | Hard tenant storage isolation (Postgres multi-tenant RFC → ship) | P1 | **Done** |
| AUDIT-IDEAL-5.1 | §5 Policy | Pre-output policy hooks on all LLM response paths | P1 | **Done** |
| AUDIT-IDEAL-5.2 | §5 Policy | Compliance profile templates per regulated domain class | P2 | **Done** |
| AUDIT-IDEAL-23.1 | §23 Security | Immutable multi-region security audit trail | P2 | **Done** |
| AUDIT-IDEAL-23.2 | §23 Security | Retrieval poisoning + tool injection live on product hosts | P1 | **Done** |
| AUDIT-IDEAL-24.1 | §24 Cost | Cost forecasting from historical run patterns | P2 | **Done** |
| AUDIT-IDEAL-24.2 | §24 Cost | Automated cost optimization recommendations (AHI) | P2 | **Done** |
| AUDIT-IDEAL-24.3 | §24 Cost | CPU/memory/concurrency quotas with tenant fairness | P2 | **Done** |
| UAEP-AUDIT-01 | §8 Runtime | Populate `tenant_id` on all `RuntimeEvent` emitters (UAEP + trace middleware) | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

### 6.1av Harness implementation queue — UAEP audit maintenance

**Source:** Interactive layer audit (2026-06-19) — `UNIFIED_EXECUTION_RUNTIME` layers 4, 5, 8, 23–24 · [`../audit_results/2026-06-19/UNIFIED_EXECUTION_RUNTIME.md`](../audit_results/2026-06-19/UNIFIED_EXECUTION_RUNTIME.md) · prior: [`../audit_results/2026-06-18/UNIFIED_EXECUTION_RUNTIME.md`](../audit_results/2026-06-18/UNIFIED_EXECUTION_RUNTIME.md)  
**Priority ladder:** **Band 1** (§6.1) — incremental after gate maintenance; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **UAEP-AUDIT-01** | Code | P2 | **Done** | `tenant_id` on `RuntimeEvent` in `UAEPExecutor._emit`, `TraceEmittingMiddleware`, and any orphan emitters | §42.44.2; regression gate on event tenant propagation |
| 2 | **UAEP-MAINT-02** | Code | P3 | **Done** | Dedup `STEP_COMPLETED` — canonical emitter in `HarnessKernel`; adjust `TraceEmittingMiddleware` to avoid duplicate journal entries | Single `STEP_COMPLETED` per step boundary in unified run journal |
| 3 | **UAEP-MAINT-03** | Docs | P3 | **Done** | Security middleware layout diagram in `AGENT_CREATION_GUIDE.md` Appendix H (`runtime/architecture/` + Tier-3 `*_wiring.py` map) | No new mechanisms; author onboarding clarity |
| 4 | **UAEP-MAINT-04** | Test | P3 | **Done** | Regression gate: at most one `STEP_COMPLETED` per step boundary (`HarnessKernel` canonical; middleware must not duplicate) | `test_kernel_emits_single_step_completed_per_step` + `test_trace_middleware_does_not_emit_step_completed_on_after_step`; gate green |

**Suggested PR order:** none — §6.1av queue closed (2026-06-19).

**Explicitly excluded:** `EscalationRouter` SUPERVISOR_AGENT target (§42.38 lab-minimal — deferred); FLOW-8 product host; GOV-PROD.1 — [§6.3](../plan/PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).

---

### 6.1aw Harness implementation queue — Security & Trust Planes (SEC-PLANES) — **Closed**

**Source:** Idea audit (2026-06-19) — modular security layer without duplicate tier · canon [§42.45.3](../architecture/UNIFIED_EXECUTION_RUNTIME.md#42453-security-and-trust-planes-canonical)  
**Priority ladder:** **Band 2bb** (§4.0) — incremental after §6.1 gate maintenance; **one ID per PR**  
**Prerequisites:** Phase SEC **Done** (SEC-1–3) · Phase M.12 **Done** (llm_guardrail) · GOV-DOC.3 **Done** (`policy_rules` EP)  
**Status:** **Done** (2026-06-19) — **17/17** · **Follow-on:** [Phase SEC-PLANES-EVOL](#phase-sec-planes-evol--enterprise-hardening-active) (Band **2bc**)

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **SEC-PLANES-DOC.1** | Docs | P1 | **Done** | Security & Trust Planes canon §42.45.2–§42.45.8 | Architecture + plan register linked |
| 2 | **SEC-PLANES-DOC.2** | Docs | P2 | **Done** | `AGENT_CREATION_GUIDE.md` Appendix H — Security Planes operator index + preset matrix | Cross-ref §42.45; no new runtime code |
| 3 | **SEC-PLANES-ADR-1** | ADR | P2 | **Done** | ADR: SecurityDefensePlugin EP + S1/S2/S3 plane discipline | Linked from §42.45 + hub; `check_harness_adr.py` green |
| 4 | **SEC-EXT-1** | Code | P1 | **Done** | `SecurityDefensePlugin` protocol + `SecurityInspectionResult` typed contract | Unit tests on protocol + schema |
| 5 | **SEC-EXT-2** | Code | P1 | **Done** | Entry point group `intergrax.security_defenses` + `register_security_defense_plugins()` | EP discovery gate; lab fixture package |
| 6 | **SEC-EXT-3** | Code | P1 | **Done** | Wire defense plugins via `security_runtime_bridge` → `MiddlewarePipeline` | Runs after native V-SEC middleware; before `ToolRuntime` |
| 7 | **SEC-EXT-4** | Code | P2 | **Done** | `ApplicationSecurityProfile.defense_plugin_ids` + `security_assembly_resolver` validation | Unknown plugin id fails wire-time on strict hosts |
| 8 | **SEC-EXT-5** | Test | P2 | **Done** | Lab reference plugin + gate tests (`tests/unit/runtime/security/`) | Plugin on `BEFORE_TOOL_CALL` blocks + traces |
| 9 | **SEC-BUNDLE-1** | Code | P2 | **Done** | Shipped defense bundle manifest pattern (native rule packs) | At least one bundle: `harness.strict_injection` |
| 10 | **SEC-BUNDLE-2** | Code | P2 | **Done** | `harness_defense_stack()` preset + `SecurityEnvelope.production()` factory | Preset composes S1+S2+S3 toggles; doc example |
| 11 | **SEC-BUNDLE-3** | Code | P3 | **Done** | `bootstrap_security_providers()` helper | Shipped bundles at import; EP via explicit call — **follow-on** SEC-EVOL-1 for `catalog_bootstrap` |
| 12 | **ENC-1** | Code | P1 | **Done** | `EncryptionEnforcementPolicy` — `DataClassification.RESTRICTED` requires resolved `secrets_store` | Fail-closed when backend missing on strict profile |
| 13 | **ENC-2** | Code | P2 | **Done** | Hook enforcement at memory write + sensitive tool output paths | RESTRICTED payload denied or encrypted via integration adapter |
| 14 | **ENC-3** | Test | P2 | **Done** | Gate tests + `check_harness_encryption_policy.py` CI script | Strict host without secrets backend fails assembly |
| 15 | **ENC-DOC.1** | Docs | P3 | **Done** | Encryption posture matrix (transit TLS vs at-rest integration) in §42.45 | No duplicate KMS SDK in Tier-0 |
| 16 | **SEC-PLANES-DOC.3** | Docs | P2 | **Done** | `EXTENSION_AUTHOR_GUIDE.md` §12 — `intergrax.security_defenses` author surface | Depends on SEC-EXT-2; cross-ref §42.21 item 7 |
| 17 | **SEC-EXT-6** | CI | P3 | **Done** | `check_harness_security_defense_plugins.py` — EP + assembly smoke | CI workflow step; strict profile lab |

**Suggested PR order:** SEC-PLANES-DOC.1 → SEC-PLANES-DOC.2 → SEC-PLANES-ADR-1 → SEC-EXT-1 → SEC-EXT-2 → SEC-EXT-3 → SEC-EXT-4 → SEC-EXT-5 → SEC-BUNDLE-1 → SEC-BUNDLE-2 → SEC-BUNDLE-3 → ENC-1 → ENC-2 → ENC-3 → ENC-DOC.1 → SEC-PLANES-DOC.3 → SEC-EXT-6.

**Phase complete when:** all **Planned** rows **Done**; §42.45.8 maturity table shows zero **Planned** for SEC-PLANES scope; gate green.

**Explicitly excluded:** standalone `SecurityEngine` tier or package; harness-native blockchain integration (M.6 exclusion); Tier-3 attestation/receipt products (product wiring only); new business agents — [§6.3a](#63a-business-backlog-register-consolidated).

---

### 6.1bc Harness implementation queue — SEC-PLANES-EVOL (enterprise hardening) — **Closed**

**Source:** Post-SEC-PLANES enterprise audit (2026-06-19) · canon [§42.45.10](../architecture/UNIFIED_EXECUTION_RUNTIME.md#424510-enterprise-hardening--maturity-model-and-backlog)  
**Priority ladder:** **Band 2bc** (§4.0) — incremental after SEC-PLANES closeout; **one ID per PR**  
**Prerequisites:** Phase SEC-PLANES **Done** (17/17) · Phase OBS spine **Done** (ADR-OBS-003)  
**Status:** **Done** (2026-06-19) — **7/7**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **SEC-EVOL-1** | Code | P1 | **Done** | Wire `bootstrap_security_providers()` into `catalog_bootstrap.py` | `bootstrap_catalogs()` invokes security bootstrap; gate test |
| 2 | **SEC-EVOL-2** | Code | P2 | **Done** | Lab EP fixture package + discovery gate | `[project.entry-points."intergrax.security_defenses"]` in repo lab fixture; CI loads plugin |
| 3 | **SEC-EVOL-3** | Code | P1 | **Done** | Security domain spine signals | `platform.security.defense_blocked` + `platform.security.encryption_denied` from middleware; typed payloads |
| 4 | **SEC-EVOL-4** | Code | P2 | **Done** | Encrypt-via-adapter path for RESTRICTED payloads | When `secrets_store` configured, middleware encrypts (not only deny) on memory write + tool output |
| 5 | **SEC-EVOL-5** | Code | P3 | **Done** | Defense plugin inspection budget / timeout guard | Per-plugin wall-clock budget on hook path; fail-closed on overrun |
| 6 | **SEC-EVOL-DOC-1** | Docs | P2 | **Done** | Enterprise maturity checklist in guides + §42.45.10 sync | Appendix H tenant-scope note; EXTENSION §12 author checklist |
| 7 | **SEC-EVOL-6** | CI | P3 | **Done** | Extend CI smoke for catalog bootstrap + EP discovery | `check_harness_security_defense_plugins.py` covers post-catalog-bootstrap path |

**Suggested PR order:** SEC-EVOL-1 → SEC-EVOL-3 → SEC-EVOL-2 → SEC-EVOL-4 → SEC-EVOL-DOC-1 → SEC-EVOL-5 → SEC-EVOL-6.

**Phase complete when:** all **Planned** rows **Done**; §42.45.8 follow-on table has zero **Planned**; gate green.

**Explicitly excluded:** harness-native blockchain; Tier-0 KMS SDK; SOC2/ISO certification artifacts; new business agents — [§6.3a](#63a-business-backlog-register-consolidated).

---

### 6.1j Harness implementation queue — legacy module closeout (closed)

**Purpose:** Single ordered list for **Phase CLEAN** (post-2p closeout). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CLEAN-1** | Code | **Done** | Remove `chat_router.py`; YAML-only tests | `test_chat_agent_prompts_yaml.py` |
| 2 | **CLEAN-2** | Code | **Done** | Remove `tools_agent.py`; planner tests | `test_catalog_tool_planner.py` |
| 3 | **CLEAN-3** | CI | **Done** | `check_legacy_modules_removed.py` in CI | workflow green |
| 4 | **CLEAN-4** | Docs | **Done** | Plan + harness docs sync | no stale production refs |

**Suggested PR order (complete):** CLEAN-1 → CLEAN-2 → CLEAN-3 → CLEAN-4.### 6.1k Harness implementation queue — agent assembly closeout (closed)

**Purpose:** Single ordered list for **Phase AS** (Band 2q). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **AS-DOC.1** | Docs | **Done** | Appendix N + cross-refs | Author map complete |
| 2 | **AS-1** | Code | **Done** | `agent_assembly_resolver` | `test_agent_assembly_resolver.py` |
| 3 | **AS-2** | Code | **Done** | Lifecycle metadata enforcement | resolver + routing tests |
| 4 | **AS-3** | CI | **Done** | `check_agent_skill_resolution.py` | CI green |

**Suggested PR order (complete):** AS-DOC.1 → AS-1 → AS-2 → AS-3.

**Explicitly excluded:** K.1, K.2, new product agents, domain-only contract packs — [§6.3a](#63a-business-backlog-register-consolidated).

---

### 6.1q Harness implementation queue — security closeout (closed)

**Purpose:** Single ordered list for **Phase SEC** (Band 2v). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **SEC-DOC.1** | Docs | **Done** | Appendix S + cross-refs | Author map complete |
| 2 | **SEC-1** | Code | **Done** | `security_runtime_bridge` + `security_wiring` | `test_harness_security_wiring.py` |
| 3 | **SEC-2** | Code | **Done** | `security_assembly_resolver` | wire-time validation tests |
| 4 | **SEC-3** | CI | **Done** | `check_harness_security_wiring.py` | CI green |

**Suggested PR order (complete):** SEC-DOC.1 → SEC-1 → SEC-2 → SEC-3.### 6.1r Harness implementation queue — cost governance closeout (closed)

**Purpose:** Single ordered list for **Phase COST** (Band 2w). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **COST-DOC.1** | Docs | **Done** | Appendix T + cross-refs | Author map complete |
| 2 | **COST-1** | Code | **Done** | `CostProfile` + `cost_runtime_bridge` + `cost_wiring` | `test_harness_cost_wiring.py` |
| 3 | **COST-2** | Code | **Done** | `cost_assembly_resolver` | wire-time validation tests |
| 4 | **COST-3** | CI | **Done** | `check_harness_cost_wiring.py` | CI green |

**Suggested PR order (complete):** COST-DOC.1 → COST-1 → COST-2 → COST-3.

---

### 6.2bn Phase COST execution order (Band 2w — closed 2026-06-02)

**Status:** **Done** · register: [Phase COST](plan/UNIFIED_EXECUTION_RUNTIME.md) · queue: [§6.1r](#61r-harness-implementation-queue--cost-governance-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | COST-DOC.1 | Appendix T + plan sync | High |
| 2 | COST-1 | `CostProfile` + `cost_runtime_bridge` + `cost_wiring` | Critical |
| 3 | COST-2 | `cost_assembly_resolver` | High |
| 4 | COST-3 | `check_harness_cost_wiring.py` | Medium |### 6.2bm Phase SEC execution order (Band 2v — closed 2026-06-02)

**Status:** **Done** · register: [Phase SEC](plan/UNIFIED_EXECUTION_RUNTIME.md) · queue: [§6.1q](#61q-harness-implementation-queue--security-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | SEC-DOC.1 | Appendix S + plan sync | High |
| 2 | SEC-1 | `security_runtime_bridge` + `security_wiring` | Critical |
| 3 | SEC-2 | `security_assembly_resolver` | High |
| 4 | SEC-3 | `check_harness_security_wiring.py` | Medium |

---

(Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)
7. **Architecture governance** — for Phase V streams, update compatibility/evaluation evidence (graph impact + score deltas)
8. **Security/cost controls** — hardening changes include policy-enforced tests for deny/degrade paths
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts



---



**Status:** **Done** (2026-06-05) — runtime governance via V-REM, H-APP, DX-5.8; documentation via GOV-DOC.*  
**Prerequisites:** Phase V-REM **Done**, H-APP.2.4–2.8 **Done**, DX-5.8 **Done**  
**Goal:** Close governance/policy/observability audit (AUDIT_MAP §5, §21) with a single authoring map and traceability — **no** new OS features.  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md` Appendix H](guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane)

**Delivery rule:** GOV-DOC.* = docs-only PRs; no code unless regression found → route to **REG-*** under §6.1.

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| GOV-DOC.1 | **Appendix H** — control plane map (profiles, bundles, hooks, EP groups, mandatory vs optional observability) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + §H.1–H.8 present |
| GOV-DOC.2 | **Cross-ref sync** — plan Documentation model, README, `guides/HARNESS_ENVIRONMENT.md`, [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.11.5, AUDIT_MAP §5/§21, audit prompt ref #5 | **Done** | Medium | `docs/*` | Links resolve; no orphan audit layer |
| GOV-DOC.3 | **`guides/EXTENSION_AUTHOR_GUIDE.md` §10** — `intergrax.policy_rules` author surface | **Done** | Medium | `guides/EXTENSION_AUTHOR_GUIDE.md` | DX-5.8 traceability |
| GOV-PROD.1 | Unified product observability dashboard (beyond lab debug APIs) | **Deferred** | — | — | **§6.3** product decision; optional `observability_backend` remains harness path |

**Explicitly out of scope:** K.1/K.2 policy; product-specific legal/org policy fragments beyond lab reference YAML.

---



**Status:** **Done** (2026-06-06) — 32-layer audit (`scope: C`) + **23/23 FAUDIT remediation** implemented → [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed)  
**Source:** [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8  
**Traceability:** **Appendix M** (layer scorecard + gap → FAUDIT ID matrix)

**Audit verdict (2026-06-06, pre-remediation snapshot):** Harness **control-plane wiring closeouts** (ORCH, TS, INT, RAG, CTX, PE, AS, REG, CG, OBS, REL, SEC, COST, EVAL, W-ADAPT, M-LLM-R) are **Done** as documented — but **closeout ≠ full layer maturity**. Per-layer inspection at audit time showed **12/32 layers at L3+**, **19/32 at L2**, **1 Critical** tier-boundary violation, **~20 High** residuals — all routed to **FAUDIT.\*** and **closed** via [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed) + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed).

**Post-remediation (2026-06-06):** **0 Critical** open; tier CI gate green; **23/23 FAUDIT** + follow-up Done. **IDEAL-L3 W1 (2026-06-09):** identity, security, cost, reliability depth — see [Phase IDEAL-L3](IDEAL_HARNESS_L3.md).

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



**Status:** **Done** (2026-06-05) — **6/6** deliverables Done (ORCH-DOC.* + ORCH-1–4); gate **581 passed**  
**Prerequisites:** R-Delegate **Done**, Q+-N.* runners **Done**, H-APP.3.1–3.2 **Done**, V-MA.* **Done**  
**Goal:** Close orchestration audit residuals (AUDIT_MAP §7–§10) — wire declared Tier-3 profile fields to runtime; bridge declarative graph spec to execution plan; cap graph batch concurrency.  
**Priority ladder:** **Band 2j** (§4.0) — **default implementation queue** after §6.1 gate on each PR.  
**Execution order:** [§6.2bb](#62bb-phase-orch-execution-order-band-2j--active) · queue: [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-active)  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md` Appendix I](guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane)

**Delivery rule:** One **ORCH-*** ID per PR → update master table + §6.1b + paydown log below → `pytest -m gate` + §6.1 scripts green.

**Audit verdict (baseline — preserve as acceptance context):**

| Area | Maturity (L0–L4) | Residual before ORCH | Close via |
|------|------------------|----------------------|-----------|
| Nexus stack (§8) | **L3–L4** | — | ORCH-DOC.* (documented) |
| Planning strategies (§7) | **L3–L4** | — | ORCH-1 **Done** |
| Declarative graph (§9) | **L3–L4** | — | ORCH-2 **Done** |
| Graph concurrency (§9) | **L3** | — | ORCH-3 **Done** |
| Subagent delegation (§10) | **L3–L4** | — | R-Delegate (Done) |

---

## Cross-domain ORCH/flow registers removed

See [`ORCHESTRATION.md`](ORCHESTRATION.md) · [`plan/plan/PLATFORM_FOUNDATION_master_registers.md`](plan/plan/PLATFORM_FOUNDATION_master_registers.md).

---
## Phase SEC-PLANES — Security & Trust Planes (**Closed**)

**Status:** **Done** (2026-06-19) — **17/17 Done** (SEC-PLANES)  
**Source:** Idea audit (2026-06-19) — modular security without duplicate tier  
**Architecture:** [§42.45.2–§42.45.9](../architecture/UNIFIED_EXECUTION_RUNTIME.md#4245-security-and-data-governance)  
**Prerequisites:** Phase SEC **Done** · M.12 **Done** · GOV-DOC.3 **Done**  
**Priority ladder:** **Band 2bb** (§4.0) — **closed**  
**Queue:** [§6.1aw](#61aw-harness-implementation-queue--security--trust-planes-sec-planes--closed)  
**Execution order:** [§6.2bo](#62bo-phase-sec-planes-execution-order-band-2bb--closed)  
**ADR:** [ADR-SEC-001](../adr/entries/2026-06-19/ADR-SEC-001.md)

**Follow-on:** [Phase SEC-PLANES-EVOL](#phase-sec-planes-evol--enterprise-hardening-active) (Band **2bc**, queue [§6.1bc](#61bc-harness-implementation-queue--sec-planes-evol-enterprise-hardening--active)).

**Goal:** Deliver a **fully modular, provider-backed security surface** inside UAEP — Security & Trust Planes (S1/S2/S3), `SecurityEnvelope` composition, shipped presets, `intergrax.security_defenses` EP, and encryption enforcement bridge — **without** a standalone Security tier or engine.

**Delivery rule:** One **SEC-PLANES-*** / **SEC-EXT-*** / **SEC-BUNDLE-*** / **ENC-*** ID per PR → update §6.1aw + §42.45.8 maturity → gate green.

### SEC-PLANES — Master register

| ID | Area | Deliverable | Priority | Status | Modules / docs | Acceptance |
|----|------|-------------|----------|--------|----------------|------------|
| SEC-PLANES-DOC.1 | DOC | Security & Trust Planes canon §42.45.2–§42.45.8 | P1 | **Done** | `architecture/UNIFIED_EXECUTION_RUNTIME.md` | Plan cross-link; plane model documented |
| SEC-PLANES-DOC.2 | DOC | Appendix H — Security Planes operator index | P2 | **Done** | `guides/AGENT_CREATION_GUIDE.md` | Preset matrix S1/S2/S3; cross-ref §42.45 |
| SEC-PLANES-ADR-1 | ADR | SecurityDefensePlugin + plane discipline ADR | P2 | **Done** | `docs/adr/entries/2026-06-19/ADR-SEC-001.md` | Hub + §42.45 link; no SecurityEngine tier |
| SEC-EXT-1 | EXT | `SecurityDefensePlugin` + `SecurityInspectionResult` | P1 | **Done** | `intergrax/runtime/security/defense_plugin.py` | Protocol unit tests |
| SEC-EXT-2 | EXT | `intergrax.security_defenses` EP + loader | P1 | **Done** | `runtime/security/defense_plugin_loader.py` | EP discovery; lab fixture |
| SEC-EXT-3 | EXT | Wire defense plugins in `security_runtime_bridge` | P1 | **Done** | `security_runtime_bridge.py`, `application_security_wiring.py` | After native V-SEC; before ToolRuntime |
| SEC-EXT-4 | EXT | `defense_plugin_ids` on profile + assembly resolver | P2 | **Done** | `environment_profile.py`, `security_assembly_resolver.py` | Wire-time fail on unknown id (strict) |
| SEC-EXT-5 | EXT | Lab reference plugin + gate tests | P2 | **Done** | `tests/unit/runtime/security/` | BEFORE_TOOL_CALL block + trace |
| SEC-EXT-6 | CI | `check_harness_security_defense_plugins.py` | P3 | **Done** | `scripts/`, CI workflow | Smoke on strict lab profile |
| SEC-BUNDLE-1 | BUNDLE | Shipped defense bundle manifest | P2 | **Done** | `intergrax/runtime/security/defense_registry.py` | `harness.strict_injection` bundle registered |
| SEC-BUNDLE-2 | BUNDLE | `harness_defense_stack()` + `SecurityEnvelope.production()` | P2 | **Done** | `integrations/registry/presets.py`, `bundles.py` | Composes S1+S2+S3; doc example |
| SEC-BUNDLE-3 | BUNDLE | `bootstrap_security_providers()` | P3 | **Done** | `intergrax/core/security_bootstrap.py` | Shipped bundles + optional EP; **SEC-EVOL-1** wires `catalog_bootstrap` |
| ENC-1 | ENC | `EncryptionEnforcementPolicy` + secrets_store gate | P1 | **Done** | `runtime/security/encryption_policy.py` | RESTRICTED requires backend on strict |
| ENC-2 | ENC | Hook enforcement memory write + tool output | P2 | **Done** | `runtime/security/encryption_middleware.py` | RESTRICTED deny when no secrets backend |
| ENC-3 | ENC | Gate tests + `check_harness_encryption_policy.py` | P2 | **Done** | `scripts/`, tests | Assembly fails without secrets backend |
| ENC-DOC.1 | DOC | Encryption posture matrix in §42.45 | P3 | **Done** | architecture canon §42.45.9 | Transit vs at-rest; no Tier-0 KMS SDK |
| SEC-PLANES-DOC.3 | DOC | `EXTENSION_AUTHOR_GUIDE.md` §12 | P2 | **Done** | `guides/EXTENSION_AUTHOR_GUIDE.md` | Depends SEC-EXT-2 |

### SEC-PLANES — Workstreams

| Workstream | IDs | Outcome |
|------------|-----|---------|
| **A — Canon & author maps** | SEC-PLANES-DOC.* | Operators and extension authors share one plane model |
| **B — Defense plugin EP** | SEC-PLANES-ADR-1, SEC-EXT-1–6 | Third-party S2 inspections as first-class plugins |
| **C — Shipped presets** | SEC-BUNDLE-* | Production-ready bundles without custom code |
| **D — Encryption bridge** | ENC-* | Close `DataClassification.requires_encryption()` gap |

**Phase complete when:** 17/17 **Done**; §42.45.8 has no **Planned** rows for SEC-PLANES scope.

**Explicitly excluded:** standalone Security tier; harness blockchain; Tier-3 receipt/attestation products; duplicate PolicyEngine — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.2bo Phase SEC-PLANES execution order (Band 2bb — **Closed**)

```text
1. SEC-PLANES-DOC.1   (Done) — canon §42.45 planes
2. SEC-PLANES-DOC.2   (Done) — Appendix H operator index
3. SEC-PLANES-ADR-1   (Done) — ADR before defense EP wiring
4. SEC-EXT-1 → SEC-EXT-2 → SEC-EXT-3 → SEC-EXT-4 → SEC-EXT-5  (Done)
5. SEC-BUNDLE-1 → SEC-BUNDLE-2 → SEC-BUNDLE-3  (Done)
6. ENC-1 → ENC-2 → ENC-3 → ENC-DOC.1  (Done)
7. SEC-PLANES-DOC.3 → SEC-EXT-6  (Done)
```

---

## Phase SEC-PLANES-EVOL — Enterprise hardening (**Closed**)

**Status:** **Done** (2026-06-19) — **7/7 Done** (SEC-PLANES-EVOL)  
**Source:** Post-SEC-PLANES enterprise audit (2026-06-19)  
**Architecture:** [§42.45.10](../architecture/UNIFIED_EXECUTION_RUNTIME.md#424510-enterprise-hardening--maturity-model-and-backlog)  
**Prerequisites:** Phase SEC-PLANES **Done** (17/17) · OBS spine domain signals (ADR-OBS-003)  
**Priority ladder:** **Band 2bc** (§4.0) — incremental after SEC-PLANES; **one ID per PR**  
**Queue:** [§6.1bc](#61bc-harness-implementation-queue--sec-planes-evol-enterprise-hardening--active)  
**Execution order:** [§6.2bp](#62bp-phase-sec-planes-evol-execution-order-band-2bc--active)

**Goal:** Close enterprise-grade gaps without a new Security tier — catalog bootstrap wiring, author EP fixture, observability spine, encrypt-via-adapter, and defense inspection resilience.

**Delivery rule:** One **SEC-EVOL-*** ID per PR → update §6.1bc + §42.45.8 follow-on maturity → gate green.

### SEC-PLANES-EVOL — Master register

| ID | Area | Deliverable | Priority | Status | Modules / docs | Acceptance |
|----|------|-------------|----------|--------|----------------|------------|
| SEC-EVOL-1 | BOOT | `bootstrap_security_providers()` in `catalog_bootstrap` | P1 | **Done** | `intergrax/core/catalog_bootstrap.py`, `security_bootstrap.py` | Host catalog bootstrap loads EP plugins by default |
| SEC-EVOL-2 | EXT | Lab EP fixture + discovery gate | P2 | **Done** | lab fixture package, tests | CI discovers fixture plugin via EP group |
| SEC-EVOL-3 | OBS | Security domain spine signals | P1 | **Done** | `security_events.py`, middleware | `platform.security.defense_blocked`, `platform.security.encryption_denied` |
| SEC-EVOL-4 | ENC | Encrypt-via-adapter for RESTRICTED | P2 | **Done** | `encryption_transform.py`, middleware | Transform path when backend configured; deny when not |
| SEC-EVOL-5 | RES | Defense plugin inspection budget | P3 | **Done** | `PluginSecurityDefenseMiddleware` | Timeout/budget guard; fail-closed on overrun |
| SEC-EVOL-DOC-1 | DOC | Enterprise maturity author checklist | P2 | **Done** | §42.45.10, Appendix H, EXTENSION §12 | Tenant scope + plugin author checklist documented |
| SEC-EVOL-6 | CI | Catalog bootstrap + EP discovery smoke | P3 | **Done** | `scripts/check_harness_security_defense_plugins.py` | Smoke after `bootstrap_catalogs()` |

### SEC-PLANES-EVOL — Workstreams

| Workstream | IDs | Outcome |
|------------|-----|---------|
| **A — Bootstrap & author DX** | SEC-EVOL-1, SEC-EVOL-2, SEC-EVOL-6 | Default EP discovery; lab fixture for third-party authors |
| **B — Observability** | SEC-EVOL-3 | SOC/SIEM-friendly security deny signals on spine |
| **C — Encryption depth** | SEC-EVOL-4 | RESTRICTED transform path beyond deny-only gate |
| **D — Resilience & docs** | SEC-EVOL-5, SEC-EVOL-DOC-1 | Hot-path guardrails + operator author checklist |

**Phase complete when:** 7/7 **Done**; §42.45.8 follow-on table has no **Planned** rows.

**Explicitly excluded:** harness blockchain; Tier-0 KMS; certification artifacts — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.2bp Phase SEC-PLANES-EVOL execution order (Band 2bc — **Closed**)

```text
1. SEC-EVOL-1   (Done) — catalog_bootstrap wiring
2. SEC-EVOL-3   (Done) — security spine signals
3. SEC-EVOL-2   (Done) — lab EP fixture + discovery gate
4. SEC-EVOL-4   (Done) — encrypt-via-adapter path
5. SEC-EVOL-DOC-1 (Done) — enterprise maturity checklist
6. SEC-EVOL-5   (Done) — defense inspection budget
7. SEC-EVOL-6   (Done) — CI smoke extension
```

---

### 6.1bd Harness implementation queue — SEC-ENT (enterprise production) — **Closed**

**Source:** Post-EVOL enterprise audit (2026-06-19) · canon [§42.45.11](../architecture/UNIFIED_EXECUTION_RUNTIME.md#424511-enterprise-production-readiness)  
**Priority ladder:** **Band 2bd** (§4.0) — after SEC-PLANES-EVOL; **one ID per PR**  
**Prerequisites:** Phase SEC-PLANES-EVOL **Done** (7/7)  
**Status:** **Done** (2026-06-19) — **6/6**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **SEC-ENT-1** | Code | P1 | **Done** | Live `SecretsStore` encryptor resolution at host wiring | `resolve_restricted_payload_encryptor(env)` prefers integration adapter |
| 2 | **SEC-ENT-2** | Code | P2 | **Done** | Typed `platform.security.*` payload schemas + registry | `register_security_payload_schemas()` at bootstrap |
| 3 | **SEC-ENT-3** | CI | P2 | **Done** | `check_harness_security_spine_signals.py` | Platform kind + event_kind registry smoke |
| 4 | **SEC-ENT-4** | Code | P1 | **Done** | Defense plugin tenant-scope guard | Cross-tenant `resource_tenant_id` blocks before inspect |
| 5 | **SEC-ENT-5** | Code | P2 | **Done** | Security spine subscriber / ops counters | `wire_security_spine_subscriber()` on host wiring |
| 6 | **SEC-ENT-DOC-1** | Docs | P2 | **Done** | §42.45.11 production matrix + ops runbook index | Appendix H SIEM subscribe note |

**Suggested PR order:** SEC-ENT-1 → SEC-ENT-4 → SEC-ENT-3 → SEC-ENT-2 → SEC-ENT-5 → SEC-ENT-DOC-1.

**Phase complete when:** 6/6 **Done**; §42.45.11 maturity rows **Done**; gate green.

**Explicitly excluded:** SOC2/ISO certification; harness-native KMS SDK; Tier-3 product SIEM dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---

## Phase SEC-ENT — Enterprise production (**Closed**)

**Status:** **Done** (2026-06-19) — **6/6 Done** (SEC-ENT)  
**Architecture:** [§42.45.11](../architecture/UNIFIED_EXECUTION_RUNTIME.md#424511-enterprise-production-readiness)  
**Queue:** [§6.1bd](#61bd-harness-implementation-queue--sec-ent-enterprise-production--closed)  
**Execution order:** [§6.2bq](#62bq-phase-sec-ent-execution-order-band-2bd--closed)

**Goal:** Close remaining harness-scope enterprise gaps — live secrets-store encryptor wiring, typed spine payloads, tenant-scope defense guard, ops counters, CI spine audit.

### SEC-ENT — Master register

| ID | Area | Deliverable | Priority | Status | Modules / docs | Acceptance |
|----|------|-------------|----------|--------|----------------|------------|
| SEC-ENT-1 | ENC | Live `SecretsStore` encryptor at host wiring | P1 | **Done** | `security_runtime_bridge.py`, `application_security_wiring.py` | Production hosts use adapter when resolvable |
| SEC-ENT-2 | OBS | Typed security spine payloads | P2 | **Done** | `runtime/security/payloads.py`, `security_bootstrap.py` | event_kind registry bound to schema_id |
| SEC-ENT-3 | CI | `check_harness_security_spine_signals.py` | P2 | **Done** | `scripts/`, CI workflow | Platform catalog + registry green |
| SEC-ENT-4 | DEF | Tenant-scope guard on defense plugins | P1 | **Done** | `defense_plugin.py` | Cross-tenant block + spine signal |
| SEC-ENT-5 | OBS | Security spine ops counters | P2 | **Done** | `security_observability.py` | Subscriber on `platform.security.*` |
| SEC-ENT-DOC-1 | DOC | §42.45.11 + Appendix H SIEM index | P2 | **Done** | architecture + guides | Ops subscribe path documented |

### 6.2bq Phase SEC-ENT execution order (Band 2bd — **Closed**)

```text
1. SEC-ENT-1   (Done) — live SecretsStore encryptor
2. SEC-ENT-4   (Done) — tenant-scope defense guard
3. SEC-ENT-3   (Done) — spine signals CI gate
4. SEC-ENT-2   (Done) — typed payloads
5. SEC-ENT-5   (Done) — ops counters subscriber
6. SEC-ENT-DOC-1 (Done) — production runbook index
```

---

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (COST-DOC.1 + COST-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §24; V-COST **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix T**.

**Priority ladder:** **Band 2w** (§4.0) — closed; default queue = **§6.1** maintenance.

### COST — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| COST-DOC.1 | COST0 | **Appendix T** — cost governance control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| COST-1 | COST1 | **`CostProfile`** + **`cost_runtime_bridge`** + **`cost_wiring`** | **Done** | `environment_profile.py`, `cost_runtime_bridge.py`, `cost_wiring.py`, `policy_wiring.py` | `test_harness_cost_wiring.py` |
| COST-2 | COST2 | **`cost_assembly_resolver`** — profile ↔ budget conformance | **Done** | `cost_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py` | assembly validation tests |
| COST-3 | COST3 | **Host cost CI** — `check_harness_cost_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product FinOps dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (EVAL-DOC.1 + EVAL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25; V-EVAL **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix U**.

**Priority ladder:** **Band 2x** (§4.0) — closed; default queue = **§6.1** maintenance.

### EVAL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| EVAL-DOC.1 | EVAL0 | **Appendix U** — evaluation control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| EVAL-1 | EVAL1 | **`EvaluationProfile`** + **`evaluation_runtime_bridge`** + **`evaluation_wiring`** | **Done** | `environment_profile.py`, `evaluation_runtime_bridge.py`, `evaluation_wiring.py`, `policy_wiring.py` | `test_harness_evaluation_wiring.py` |
| EVAL-2 | EVAL2 | **`evaluation_assembly_resolver`** — profile ↔ registry conformance | **Done** | `evaluation_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py`, `runtime.py` | assembly validation tests |
| EVAL-3 | EVAL3 | **Host evaluation CI** — `check_harness_evaluation_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product quality dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-08) — **24/24** deliverables Done (CRIT-V-0 through CRIT-V-7)  
**Prerequisites:** Phase EVAL **Done** (registry wiring), Phase FLOW **Done** (graph hooks), Phase M-LLM-R **Done** (typed LLM envelope)  
**Goal:** Deliver production-grade PEV **Verify** infrastructure — L0/L1/L2 critic stack with tier-separated competencies; uplift Evaluation audit layer L2→L3.  
**Priority ladder:** **Band 2ak** (§4.0) — **Done** (2026-06-08). Default queue reverts to §6.1 gate maintenance.  
**Architecture:** [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · canon [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · [ADR-CRITIC-001](adr/entries/2026-06-07/ADR-CRITIC-001.md)  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25 (Evaluation), §7 (Reasoning), §10 (Multi-agent); closes **FAUDIT-EVAL.1** residual  
**Execution order:** [§6.2ak](#62ak-phase-crit-v-execution-order-band-2ak--closed) · queue: [§6.1ak](#61ak-harness-implementation-queue--critic-verification-layer-closed)

**Delivery rule:** One **CRIT-V-*** ID per PR → update master table + §6.1ak + gate green.

### CRIT-V — Master register

| ID | Wave | Deliverable | Status | Modules / docs | Acceptance |
|----|------|-------------|--------|----------------|------------|
| CRIT-V-0.1 | 0 | **Architecture RFC** — CVL full spec | **Done** | `architecture/CRITIC_VERIFICATION.md` | Linked from canon §55, README |
| CRIT-V-0.2 | 0 | **ADR-CRITIC-001** — tier-separated PEV verify | **Done** | `docs/adr/entries/2026-06-07/ADR-CRITIC-001.md` | Status Accepted; adr index |
| CRIT-V-0.3 | 0 | **Canon §55** addendum | **Done** | `intergrax_runtime_architecture.md` §55 | Cross-links resolve |
| CRIT-V-0.4 | 0 | **README** sections (root + docs) | **Done** | `README.md`, `docs/README.md` | Navigation table |
| CRIT-V-1.1 | 1 | **`CriticProfile`** on `ApplicationEnvironmentProfile` | **Done** | `contracts/environment_profile.py`, `critic_runtime_bridge.py`, `RuntimeConfig` | Unit: `test_harness_critic_wiring.py` |
| CRIT-V-1.2 | 1 | **CVL contracts** — `CriticRequest`, `CriticVerdict`, `LayerVerdict`, `RubricSpec` | **Done** | `runtime/critic/contracts.py` | Unit: `test_critic_contracts.py` |
| CRIT-V-1.3 | 1 | **`EvaluatorLoopSpec`** — max iterations, revise routing | **Done** | `runtime/critic/evaluator_loop_spec.py` | Unit: `test_evaluator_loop_spec.py` |
| CRIT-V-2.1 | 2 | **`eval.judge` tool** — semantic scoring via separate LLM profile | **Done** | `tools/providers/eval/judge.py`, bundle | `test_eval_critic_tools.py` |
| CRIT-V-2.2 | 2 | **`eval.trajectory` tool** — process scoring from replay slice | **Done** | `tools/providers/eval/trajectory.py` | Uses `trace_reader` |
| CRIT-V-2.3 | 2 | **Registry hook** — judge/trajectory → `OnlineEvaluationObservation` | **Done** | `service.py` `_append_critic_observation` | Observation appended when registry bound |
| CRIT-V-3.1 | 3 | **`CriticOrchestrator`** — L0→L1→L2 pipeline | **Done** | `runtime/critic/critic_orchestrator.py` | Unit: short-circuit, layer order |
| CRIT-V-3.2 | 3 | **`L0Gateway`** — wraps `NexusValidationEngine` + schema | **Done** | `runtime/critic/l0_gateway.py` | Reuses existing validators |
| CRIT-V-3.3 | 3 | **`L1Gateway`** — invokes eval tools via `CriticEvalToolClient` | **Done** | `runtime/critic/l1_gateway.py` | No direct LLM in Tier-1 |
| CRIT-V-3.4 | 3 | **Graph partial hook** — `GraphExecutor` → `verify_partial` | **Done** | `graph_executor.py`, `critic_wiring.py` | Integration test: L0 fail → retry |
| CRIT-V-3.5 | 3 | **Graph final hook** — `GraphRunner` → `verify_final` | **Done** | `graph_runner.py` | Terminal state respects verdict |
| CRIT-V-3.6 | 3 | **Critic trace events** — `critic.*` trace steps | **Done** | `runtime/critic/trace.py`, `trace_bridge.py` | Visible in lab trace API |
| CRIT-V-4.1 | 4 | **`EvaluatorLoopExecutor`** — critique→revise routing | **Done** | `runtime/critic/evaluator_loop_executor.py` | Unit: budget exhaustion → FAIL/HITL |
| CRIT-V-4.2 | 4 | **Graph integration** — `EVALUATOR_LOOP` pattern wired | **Done** | `graph_executor.py`, `evaluator_loop_metadata.py` | Acceptance: 2-iteration loop |
| CRIT-V-5.1 | 5 | **`NexusEvalRunner` semantic mode** — optional L1 via `eval.judge` | **Done** | `eval/nexus_eval_runner.py` | Integration: non-exact pass |
| CRIT-V-5.2 | 5 | **`EvalCase` rubric field** — rubric_ref + semantic_threshold | **Done** | `eval/eval_case.py` | Backward compatible |
| CRIT-V-6.1 | 6 | **`wire_application_critic()`** — Tier-3 wiring | **Done** | `applications/_shared/critic_wiring.py` | Mirror EVAL pattern |
| CRIT-V-6.2 | 6 | **`critic_assembly_resolver`** — wire-time validation | **Done** | `critic_assembly_resolver.py`, `check_harness_critic_wiring.py` | CI script |
| CRIT-V-6.3 | 6 | **Policy bundle** — `critic_governance` fragment | **Done** | `policy_wiring.py` | Merged at host build |
| CRIT-V-6.4 | 6 | **Appendix W** — critic control plane author map | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CRIT-V-7.1 | 7 | **FAUDIT-EVAL.1** — `require_baseline_for_release` CI gate | **Done** | `phase_v_closeout_gate.py`, `check_harness_critic_wiring.py` | Closeout gate green |
| CRIT-V-7.2 | 7 | **Flow reference §18 sync** — CVL hook table | **Done** | `architecture/NEXUS_EXECUTION_FLOW.md` | Hooks documented |
| CRIT-V-7.3 | 7 | **Lab harness demo** — L0+L1 on sample agent (not FLOW-8) | **Done** | `test_harness_critic_wiring.py`, lab host | Trace shows critic steps |
| CRIT-V-F.1 | F | **`ToolRegistryCriticEvalClient`** — L1 bridge to Tier-0 eval tools | **Done** | `runtime/critic/tool_registry_client.py`, `critic_tool_wiring.py` | `test_critic_closeout.py` |
| CRIT-V-F.2 | F | **`critic_llm_profile`** — separate judge LLM adapter | **Done** | `critic_llm_resolver.py`, `environment_profile.py` | Assembly + wiring tests |
| CRIT-V-F.3 | F | **L2 `L2Gateway`** + HITL escalation path | **Done** | `l2_gateway.py`, `critic_orchestrator.py` | `test_critic_l2_gateway.py` |
| CRIT-V-F.4 | F | **UAEP step hook** | **Done** | `uaep.py`, `validate_uaep_step_with_critic_detail` | UAEP critic path |
| CRIT-V-F.5 | F | **`CriticPolicyBridge`** + policy engine | **Done** | `policy_bridge.py`, `runtime_policy_engine.py` | `test_critic_closeout.py` |
| CRIT-V-F.6 | F | **Assembly gate** — require L1 client when semantic/trajectory enabled | **Done** | `critic_assembly_resolver.py` | `test_critic_assembly_resolver.py` |

**Explicitly excluded:** FLOW-8 §42.43 product reference app ([§6.3](#63-end-of-plan--deferred-product-work-only)); domain rubric packs in Tier-0; mandatory universal LLM-judge on all runs.

**Phase CRIT-V complete when:** CRIT-V-1 through CRIT-V-7 **Done**; Evaluation audit layer ≥ **L3**; gate green; FAUDIT-EVAL.1 closed.

---

---

## Phase CLEAN — Legacy module closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CLEAN-1–4)

**Audit basis:** Phase U-Leg residual; `scripts/check_legacy_modules_removed.py`; prior `check_tools_agent_*` audits merged.

**Priority ladder:** closeout between Band 2p and 2q; default queue = **Band 2q** [Phase AS](#phase-as--agent-assembly-control-plane-closeout).

### CLEAN — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CLEAN-1 | CLEAN1 | **Remove `legacy/chat_router.py`** — YAML assets tested without runtime module | **Done** | `tests/unit/chat_agent/` | prompt YAML tests green |
| CLEAN-2 | CLEAN2 | **Remove `tools/tools_agent.py`** — `CatalogToolPlanner` + `ToolPlanningService` canonical | **Done** | `catalog_tool_planner.py`, `tool_planning_service.py` | `test_catalog_tool_planner.py` |
| CLEAN-3 | CLEAN3 | **Unified CI audit** — `check_legacy_modules_removed.py` replaces `check_tools_agent_*` | **Done** | `scripts/`, `.github/workflows/unit-tests.yml` | audit script green in CI |
| CLEAN-4 | CLEAN4 | **Docs sync** — plan, HARNESS_ENVIRONMENT, AGENT_CREATION_GUIDE, README, TOOLS | **Done** | `docs/*` | no stale `ToolsAgent` production paths |

**Retained (not CLEAN scope):** `ToolInvocationPlan.from_legacy()` + deprecation tests; `EnginePlan.use_rag`/`use_websearch` on LLM schema; `intergrax/legacy/rag_answers/` archive with import guard; diagnostic type names (`CoreLLMUsedToolsAgentAnswerDiagV1`).

---

---

## Phase GOV-AUDIT — Governance control plane (audit closeout)

**Status:** **Done** (2026-06-05) — runtime governance via V-REM, H-APP, DX-5.8; documentation via GOV-DOC.*  
**Prerequisites:** Phase V-REM **Done**, H-APP.2.4–2.8 **Done**, DX-5.8 **Done**  
**Goal:** Close governance/policy/observability audit (AUDIT_MAP §5, §21) with a single authoring map and traceability — **no** new OS features.  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md` Appendix H](guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane)

**Delivery rule:** GOV-DOC.* = docs-only PRs; no code unless regression found → route to **REG-*** under §6.1.

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| GOV-DOC.1 | **Appendix H** — control plane map (profiles, bundles, hooks, EP groups, mandatory vs optional observability) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + §H.1–H.8 present |
| GOV-DOC.2 | **Cross-ref sync** — plan Documentation model, README, `guides/HARNESS_ENVIRONMENT.md`, canon §42.11.5, AUDIT_MAP §5/§21, audit prompt ref #5 | **Done** | Medium | `docs/*` | Links resolve; no orphan audit layer |
| GOV-DOC.3 | **`guides/EXTENSION_AUTHOR_GUIDE.md` §10** — `intergrax.policy_rules` author surface | **Done** | Medium | `guides/EXTENSION_AUTHOR_GUIDE.md` | DX-5.8 traceability |
| GOV-PROD.1 | Unified product observability dashboard (beyond lab debug APIs) | **Deferred** | — | — | **§6.3** product decision; optional `observability_backend` remains harness path |

**Explicitly out of scope:** K.1/K.2 policy; product-specific legal/org policy fragments beyond lab reference YAML.

---

---

## Phase SEC — Security control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (SEC-DOC.1 + SEC-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §23; V-SEC / V-REM-SEC **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix S**.

**Priority ladder:** **Band 2v** (§4.0) — closed; default queue = **§6.1** maintenance.

### SEC — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| SEC-DOC.1 | SEC0 | **Appendix S** — security control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| SEC-1 | SEC1 | **`security_runtime_bridge`** + **`security_wiring`** | **Done** | `security_runtime_bridge.py`, `security_wiring.py`, `runtime_config_bridge.py` | `test_harness_security_wiring.py` |
| SEC-2 | SEC2 | **`security_assembly_resolver`** — profile ↔ middleware conformance | **Done** | `security_assembly_resolver.py`, `harness_host_runtime.py`, `nexus_factory.py` | assembly validation tests |
| SEC-3 | SEC3 | **Host security CI** — `check_harness_security_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only security dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---

---

## Phase COST — Cost governance control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (COST-DOC.1 + COST-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §24; V-COST **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix T**.

**Priority ladder:** **Band 2w** (§4.0) — closed; default queue = **§6.1** maintenance.

### COST — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| COST-DOC.1 | COST0 | **Appendix T** — cost governance control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| COST-1 | COST1 | **`CostProfile`** + **`cost_runtime_bridge`** + **`cost_wiring`** | **Done** | `environment_profile.py`, `cost_runtime_bridge.py`, `cost_wiring.py`, `policy_wiring.py` | `test_harness_cost_wiring.py` |
| COST-2 | COST2 | **`cost_assembly_resolver`** — profile ↔ budget conformance | **Done** | `cost_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py` | assembly validation tests |
| COST-3 | COST3 | **Host cost CI** — `check_harness_cost_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product FinOps dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---

---

### Phase P4 — §42 Unified Execution Runtime



| Step | Deliverable | Status |

|------|-------------|--------|

| P4.1 | Event bus + trace bridge | **Done** |

| P4.2 | UAEP in AgentEngine | **Done** |

| P4.3 | Governance (interrupt, HITL) | **Done** |

| P4.4 | Tool gateway unification | **Done** |

| P4.5 | Agent migration (Echo, Research, Legal) | **Done** |



**P4.5 delivered (2026-05-27):** `uaep_pipeline.py`; Research, Summary, Legal agents on UAEP (`get_steps` / `run_step` / `decide_after_step`); integration tests + NexusLoop research. Gate: 31 tests.



**P4.4 delivered (2026-05-27):** `RuntimeToolGateway`, `ToolRuntime.invoke_request`, Legal bridge via `ToolRequest`; UAEP `BoundToolGateway`. Gate: 25 tests.



**P4.3 delivered (2026-05-27):** `runtime/interrupts/`, `runtime/human/`, policy in UAEP + NexusLoop.



---

---

### Phase G — §42 Runtime Convergence

**Goal:** Close largest gaps vs §42.9, §42.10, §42.24, §42.40 (evolve, not rewrite).

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| G.1 | `RuntimeCheckpoint` contract | **Done** | §42.9.2 | Plan + graph node states + UAEP step index |
| G.2 | UAEP mid-execution resume | **Done** | §42.9.3 | Skip re-run paused step on resume |
| G.3 | HITL middleware hooks | **Done** | §42.10 | `BEFORE/AFTER_HUMAN_APPROVAL` in NexusLoop |
| G.4 | `HumanRequest` v2 fields | **Done** | §42.10.1 | Typed urgency, deadline propagation, timeout stub |
| G.5 | RuntimeEvent-first observability | **Done** | §42.24 | `RuntimeEventPersistence` + `store.py` (`open_runtime_event_store`, env `INTERGRAX_RUNTIME_EVENTS_DB` only) |
| G.6 | Debug API: HITL + checkpoints | **Done** | §19 | Pluggable stores; events/checkpoints/HITL resume |
| G.7 | Graph failure recovery | **Done** | §42.40, §30 | Skip completed nodes; checkpoint on graph fail |
| G.8 | Cooperative cancellation | **Done** | §42.26 | Cancel propagation through graph / UAEP |

---

---

### Phase J — Unified Execution Entry (§41)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| J.1 | NexusLoop default in apps | **Done** | §41 | Legal + Research: `UnifiedTaskRunner` only (legacy `AgentEngine` removed, B.14) |
| J.2 | RunService → UnifiedTaskRunner | **Done** | §41 | `NexusTaskExecutionAdapter` + `CreateRunRequest.payload` → Task |
| J.3 | Worker queue Task v2 | **Done** | §41 | `QueuedNexusExecutionAdapter`, `nexus.task.v2` Celery handler, checkpoint resume |
| J.4 | Long-running scheduler | **Done** | §26 | `LongRunningScheduler`, delayed resume + HITL timeout enforcement |
| J.5 | Partial results API | **Done** | §26 | `GET /debug/tasks/{id}/progress`, `TASK_PROGRESS` events, notification template |

---

---

#### V-COST — Cost & Resource Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-COST.1 | Budget envelopes (tenant/app/agent/model/tool) | **Done** | High | Budget policy enforcement tests |
| V-COST.2 | Token/tool/resource quotas with deny/degrade behavior | **Done** | High | Quota exceedance behavior deterministic |
| V-COST.3 | Forecast + anomaly detection for spend and token drift | **Done** | Medium | Forecast/anomaly report available |
| V-COST.4 | Optimization recommendations with policy guardrails | **Done** | Medium | Recommendations recorded in ops reports |#### V-MA — Multi-Agent Coordination Model Catalog

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-MA.1 | Coordination patterns catalog (hierarchical/orchestrator-worker/supervisor-worker/peer/swarm/evaluator-loop) | **Done** | High | Canon section + selection table |
| V-MA.2 | Pattern selection matrix (risk/latency/cost/complexity) | **Done** | High | Matrix used in planning docs |
| V-MA.3 | Pattern-specific acceptance tests | **Done** | Medium | Test suite covers selected patterns |

---

## Phase GR-DOC — Guardrail catalog documentation

**Status:** **Done** (2026-06-09) — documentation (GR-DOC.*) + implementation (GR-INT.*) via [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) Phase **M.12**.

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §5 (Policy), §23 (Security); IDEAL §3.3 guardrails vector.

**ADR:** **No ADR needed** for documentation-only catalog. First shipped `llm_guardrail` slug (M.12) requires harness ADR (`docs/adr/`) for contract + middleware bridge.

### GR-DOC — Master register

| ID | Type | Deliverable | Status | Evidence |
|----|------|-------------|--------|----------|
| GR-DOC.1 | Docs | UAEP **§42.11.6** guardrail catalog (types → hooks → artifacts) | **Done** | `architecture/UNIFIED_EXECUTION_RUNTIME.md` |
| GR-DOC.2 | Docs | Hub cross-index row + §42.37 governance layer 7 | **Done** | `intergrax_runtime_architecture.md` |
| GR-DOC.3 | Docs | Harness term **Guardrails** in PLATFORM §5.3.1 | **Done** | `architecture/PLATFORM_FOUNDATION.md` |
| GR-DOC.4 | Docs | Integration canon **§47** + vendor library matrix | **Done** | `architecture/INTEGRATIONS.md` |
| GR-INT.1 | Code | `LlmGuardrailBackend` contract + M.12 slugs | **Done** | [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) M.12 |
| GR-INT.2 | Code | `guardrail_runtime_bridge` → §42.42 middleware | **Done** | M.12-WIRE.1 |
| GR-INT.3 | CI | `check_harness_guardrail_wiring.py` | **Done** | M.12-WIRE.3 |
| GR-INT.4 | Code | CVL L0 ↔ guardrail scan composition | **Done** | `runtime/critic/guardrail_l0.py` |

### GR-DOC — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-09 | GR-DOC.1–4 | Guardrail catalog canon + hub index + PLATFORM term + INTEGRATIONS §47 vendor matrix |
| 2026-06-09 | GR-INT.1–3 | M.12 code: adapters, middleware, assembly, tests, CI |
| 2026-06-09 | GR-INT.FU | Output hooks (AFTER_LLM_OUTPUT/FINALIZATION), chained backends, runtime bridge tests |
| 2026-06-09 | GR-INT.4 | NeMo opens.py, HTTP smoke tests, CVL L0 merge, lab/legal guardrail toggles, USAGE.md |
| 2026-06-09 | GR-INT.HARD | `GUARDRAIL_BLOCKED` runtime event, E2E Nexus guardrail gate, PLATFORM Band 2ay register |

---
