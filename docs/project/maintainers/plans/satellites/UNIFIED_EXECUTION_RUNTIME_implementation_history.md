# UNIFIED_EXECUTION_RUNTIME — implementation history + LC closeout

**Parent hub:** [`UNIFIED_EXECUTION_RUNTIME.md`](../UNIFIED_EXECUTION_RUNTIME.md)

> **Plan ownership:** Implementation phases and LC closeout below. Historical audit findings/verdicts archived at [docs/audit_results/legacy/plan-audit-history/UNIFIED_EXECUTION_RUNTIME_audit_history.md](../../../../audit_results/legacy/plan-audit-history/UNIFIED_EXECUTION_RUNTIME_audit_history.md).


## Phase SEC-PLANES — Security & Trust Planes (**Closed**)

**Status:** **Done** (2026-06-19) — **17/17 Done** (SEC-PLANES)  
**Source:** Idea audit (2026-06-19) — modular security without duplicate tier  
**Architecture:** [§42.45.2–§42.45.9](../architecture/UNIFIED_EXECUTION_RUNTIME.md#4245-security-and-data-governance)  
**Prerequisites:** Phase SEC **Done** · M.12 **Done** · GOV-DOC.3 **Done**  
**Priority ladder:** **Band 2bb** (§4.0) — **closed**  
**Queue:** [§6.1aw](.#61aw-harness-implementation-queue--security--trust-planes-sec-planes--closed)
**Execution order:** [§6.2bo](.#62bo-phase-sec-planes-execution-order-band-2bb--closed)
**ADR:** [ADR-SEC-001](../adr/entries/2026-06-19/ADR-SEC-001.md)

**Follow-on:** [Phase SEC-PLANES-EVOL](.#phase-sec-planes-evol--enterprise-hardening-active) (Band **2bc**, queue [§6.1bc](.#61bc-harness-implementation-queue--sec-planes-evol-enterprise-hardening--active)).

**Goal:** Deliver a **fully modular, provider-backed security surface** inside UAEP — Security & Trust Planes (S1/S2/S3), `SecurityEnvelope` composition, shipped presets, `intergrax.security_defenses` EP, and encryption enforcement bridge — **without** a standalone Security tier or engine.

**Delivery rule:** One **SEC-PLANES-*** / **SEC-EXT-*** / **SEC-BUNDLE-*** / **ENC-*** ID per PR → update §6.1aw + §42.45.8 maturity → gate green.

### SEC-PLANES — Master register

| ID | Area | Deliverable | Priority | Status | Modules / docs | Acceptance |
|----|------|-------------|----------|--------|----------------|------------|
| SEC-PLANES-DOC.1 | DOC | Security & Trust Planes canon §42.45.2–§42.45.8 | P1 | **Done** | `architecture/UNIFIED_EXECUTION_RUNTIME.md` | Plan cross-link; plane model documented |
| SEC-PLANES-DOC.2 | DOC | Appendix H — Security Planes operator index | P2 | **Done** | `guides/AGENT_CREATION_GUIDE.md` | Preset matrix S1/S2/S3; cross-ref §42.45 |
| SEC-PLANES-ADR-1 | ADR | SecurityDefensePlugin + plane discipline ADR | P2 | **Done** | `docs/project/technical/adr/entries/2026-06-19/ADR-SEC-001.md` | Hub + §42.45 link; no SecurityEngine tier |
| SEC-EXT-1 | EXT | `SecurityDefensePlugin` + `SecurityInspectionResult` | P1 | **Done** | `intergrax/runtime/security/defense_plugin.py` | Protocol unit tests |
| SEC-EXT-2 | EXT | `intergrax.security_defenses` EP + loader | P1 | **Done** | `runtime/security/defense_plugin_loader.py` | EP discovery; lab fixture |
| SEC-EXT-3 | EXT | Wire defense plugins in `security_runtime_bridge` | P1 | **Done** | `security_runtime_bridge.py`, `application_security_wiring.py` | After native V-SEC; before ToolRuntime |
| SEC-EXT-4 | EXT | `defense_plugin_ids` on profile + assembly resolver | P2 | **Done** | `environment_profile.py`, `security_assembly_resolver.py` | Wire-time fail on unknown id (strict) |
| SEC-EXT-5 | EXT | Lab reference plugin + gate tests | P2 | **Done** | `tests/unit/runtime/security` | BEFORE_TOOL_CALL block + trace |
| SEC-EXT-6 | CI | `check_harness_security_defense_plugins.py` | P3 | **Done** | `scripts`, CI workflow | Smoke on strict lab profile |
| SEC-BUNDLE-1 | BUNDLE | Shipped defense bundle manifest | P2 | **Done** | `intergrax/runtime/security/defense_registry.py` | `harness.strict_injection` bundle registered |
| SEC-BUNDLE-2 | BUNDLE | `harness_defense_stack()` + `SecurityEnvelope.production()` | P2 | **Done** | `integrations/registry/presets.py`, `bundles.py` | Composes S1+S2+S3; doc example |
| SEC-BUNDLE-3 | BUNDLE | `bootstrap_security_providers()` | P3 | **Done** | `intergrax/core/security_bootstrap.py` | Shipped bundles + optional EP; **SEC-EVOL-1** wires `catalog_bootstrap` |
| ENC-1 | ENC | `EncryptionEnforcementPolicy` + secrets_store gate | P1 | **Done** | `runtime/security/encryption_policy.py` | RESTRICTED requires backend on strict |
| ENC-2 | ENC | Hook enforcement memory write + tool output | P2 | **Done** | `runtime/security/encryption_middleware.py` | RESTRICTED deny when no secrets backend |
| ENC-3 | ENC | Gate tests + `check_harness_encryption_policy.py` | P2 | **Done** | `scripts`, tests | Assembly fails without secrets backend |
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

**Explicitly excluded:** standalone Security tier; harness blockchain; Tier-3 receipt/attestation products; duplicate PolicyEngine — [§6.3a](.#63a-business-backlog-register-consolidated).

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
**Queue:** [§6.1bc](.#61bc-harness-implementation-queue--sec-planes-evol-enterprise-hardening--active)
**Execution order:** [§6.2bp](.#62bp-phase-sec-planes-evol-execution-order-band-2bc--active)

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
| SEC-EVOL-6 | CI | Catalog bootstrap + EP discovery smoke | P3 | **Done** | `scripts/maintenance/check_harness_security_defense_plugins.py` | Smoke after `bootstrap_catalogs()` |

### SEC-PLANES-EVOL — Workstreams

| Workstream | IDs | Outcome |
|------------|-----|---------|
| **A — Bootstrap & author DX** | SEC-EVOL-1, SEC-EVOL-2, SEC-EVOL-6 | Default EP discovery; lab fixture for third-party authors |
| **B — Observability** | SEC-EVOL-3 | SOC/SIEM-friendly security deny signals on spine |
| **C — Encryption depth** | SEC-EVOL-4 | RESTRICTED transform path beyond deny-only gate |
| **D — Resilience & docs** | SEC-EVOL-5, SEC-EVOL-DOC-1 | Hot-path guardrails + operator author checklist |

**Phase complete when:** 7/7 **Done**; §42.45.8 follow-on table has no **Planned** rows.

**Explicitly excluded:** harness blockchain; Tier-0 KMS; certification artifacts — [§6.3a](.#63a-business-backlog-register-consolidated).

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

**Explicitly excluded:** SOC2/ISO certification; harness-native KMS SDK; Tier-3 product SIEM dashboards — [§6.3a](.#63a-business-backlog-register-consolidated).

---

## Phase SEC-ENT — Enterprise production (**Closed**)

**Status:** **Done** (2026-06-19) — **6/6 Done** (SEC-ENT)  
**Architecture:** [§42.45.11](../architecture/UNIFIED_EXECUTION_RUNTIME.md#424511-enterprise-production-readiness)  
**Queue:** [§6.1bd](.#61bd-harness-implementation-queue--sec-ent-enterprise-production--closed)
**Execution order:** [§6.2bq](.#62bq-phase-sec-ent-execution-order-band-2bd--closed)

**Goal:** Close remaining harness-scope enterprise gaps — live secrets-store encryptor wiring, typed spine payloads, tenant-scope defense guard, ops counters, CI spine audit.

### SEC-ENT — Master register

| ID | Area | Deliverable | Priority | Status | Modules / docs | Acceptance |
|----|------|-------------|----------|--------|----------------|------------|
| SEC-ENT-1 | ENC | Live `SecretsStore` encryptor at host wiring | P1 | **Done** | `security_runtime_bridge.py`, `application_security_wiring.py` | Production hosts use adapter when resolvable |
| SEC-ENT-2 | OBS | Typed security spine payloads | P2 | **Done** | `runtime/security/payloads.py`, `security_bootstrap.py` | event_kind registry bound to schema_id |
| SEC-ENT-3 | CI | `check_harness_security_spine_signals.py` | P2 | **Done** | `scripts`, CI workflow | Platform catalog + registry green |
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
| COST-3 | COST3 | **Host cost CI** — `check_harness_cost_wiring.py` | **Done** | `scripts`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product FinOps dashboards — [§6.3a](.#63a-business-backlog-register-consolidated).

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
| EVAL-3 | EVAL3 | **Host evaluation CI** — `check_harness_evaluation_wiring.py` | **Done** | `scripts`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product quality dashboards — [§6.3a](.#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-08) — **24/24** deliverables Done (CRIT-V-0 through CRIT-V-7)  
**Prerequisites:** Phase EVAL **Done** (registry wiring), Phase FLOW **Done** (graph hooks), Phase M-LLM-R **Done** (typed LLM envelope)  
**Goal:** Deliver production-grade PEV **Verify** infrastructure — L0/L1/L2 critic stack with tier-separated competencies; uplift Evaluation audit layer L2→L3.  
**Priority ladder:** **Band 2ak** (§4.0) — **Done** (2026-06-08). Default queue reverts to §6.1 gate maintenance.  
**Architecture:** [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · canon [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · [ADR-CRITIC-001](adr/entries/2026-06-07/ADR-CRITIC-001.md)  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25 (Evaluation), §7 (Reasoning), §10 (Multi-agent); closes **FAUDIT-EVAL.1** residual  
**Execution order:** [§6.2ak](.#62ak-phase-crit-v-execution-order-band-2ak--closed) · queue: [§6.1ak](.#61ak-harness-implementation-queue--critic-verification-layer-closed)

**Delivery rule:** One **CRIT-V-*** ID per PR → update master table + §6.1ak + gate green.

### CRIT-V — Master register

| ID | Wave | Deliverable | Status | Modules / docs | Acceptance |
|----|------|-------------|--------|----------------|------------|
| CRIT-V-0.1 | 0 | **Architecture RFC** — CVL full spec | **Done** | `architecture/CRITIC_VERIFICATION.md` | Linked from canon §55, README |
| CRIT-V-0.2 | 0 | **ADR-CRITIC-001** — tier-separated PEV verify | **Done** | `docs/project/technical/adr/entries/2026-06-07/ADR-CRITIC-001.md` | Status Accepted; adr index |
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

**Explicitly excluded:** FLOW-8 §42.43 product reference app ([§6.3](.#63-end-of-plan--deferred-product-work-only)); domain rubric packs in Tier-0; mandatory universal LLM-judge on all runs.

**Phase CRIT-V complete when:** CRIT-V-1 through CRIT-V-7 **Done**; Evaluation audit layer ≥ **L3**; gate green; FAUDIT-EVAL.1 closed.

---

---

## Phase CLEAN — Legacy module closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CLEAN-1–4)

**Audit basis:** Phase U-Leg residual; `scripts/maintenance/check_legacy_modules_removed.py`; prior `check_tools_agent_*` audits merged.

**Priority ladder:** closeout between Band 2p and 2q; default queue = **Band 2q** [Phase AS](.#phase-as--agent-assembly-control-plane-closeout).

### CLEAN — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CLEAN-1 | CLEAN1 | **Remove `legacy/chat_router.py`** — YAML assets tested without runtime module | **Done** | `tests/unit/chat_agent` | prompt YAML tests green |
| CLEAN-2 | CLEAN2 | **Remove `tools/tools_agent.py`** — `CatalogToolPlanner` + `ToolPlanningService` canonical | **Done** | `catalog_tool_planner.py`, `tool_planning_service.py` | `test_catalog_tool_planner.py` |
| CLEAN-3 | CLEAN3 | **Unified CI audit** — `check_legacy_modules_removed.py` replaces `check_tools_agent_*` | **Done** | `scripts`, `.github/workflows/unit-tests.yml` | audit script green in CI |
| CLEAN-4 | CLEAN4 | **Docs sync** — plan, HARNESS_ENVIRONMENT, AGENT_CREATION_GUIDE, README, TOOLS | **Done** | `docs/*` | no stale `ToolsAgent` production paths |

**Retained (not CLEAN scope):** `ToolInvocationPlan.from_legacy()` + deprecation tests; `EnginePlan.use_rag`/`use_websearch` on LLM schema; `intergrax/legacy/rag_answers` archive with import guard; diagnostic type names (`CoreLLMUsedToolsAgentAnswerDiagV1`).

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
| SEC-3 | SEC3 | **Host security CI** — `check_harness_security_wiring.py` | **Done** | `scripts`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only security dashboards — [§6.3a](.#63a-business-backlog-register-consolidated).

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
| COST-3 | COST3 | **Host cost CI** — `check_harness_cost_wiring.py` | **Done** | `scripts`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product FinOps dashboards — [§6.3a](.#63a-business-backlog-register-consolidated).

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



**P4.3 delivered (2026-05-27):** `runtime/interrupts`, `runtime/human`, policy in UAEP + NexusLoop.



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

**ADR:** **No ADR needed** for documentation-only catalog. First shipped `llm_guardrail` slug (M.12) requires harness ADR (`docs/project/technical/adr`) for contract + middleware bridge.

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
