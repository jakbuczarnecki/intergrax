# UNIFIED_EXECUTION_RUNTIME — embedded detail

**Parent hub:** [`UNIFIED_EXECUTION_RUNTIME.md`](../UNIFIED_EXECUTION_RUNTIME.md)

### 6.1aw Harness implementation queue — Security & Trust Planes (SEC-PLANES) — **Closed**

**Source:** Idea audit (2026-06-19) — modular security layer without duplicate tier · canon [§42.45.3](../architecture/UNIFIED_EXECUTION_RUNTIME.md#42453-security-and-trust-planes-canonical)  
**Priority ladder:** **Band 2bb** (§4.0) — incremental after §6.1 gate maintenance; **one ID per PR**  
**Prerequisites:** Phase SEC **Done** (SEC-1–3) · Phase M.12 **Done** (llm_guardrail) · GOV-DOC.3 **Done** (`policy_rules` EP)  
**Status:** **Done** (2026-06-19) — **17/17** · **Follow-on:** [Phase SEC-PLANES-EVOL](.#phase-sec-planes-evol--enterprise-hardening-active) (Band **2bc**)

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **SEC-PLANES-DOC.1** | Docs | P1 | **Done** | Security & Trust Planes canon §42.45.2–§42.45.8 | Architecture + plan register linked |
| 2 | **SEC-PLANES-DOC.2** | Docs | P2 | **Done** | `AGENT_CREATION_GUIDE.md` Appendix H — Security Planes operator index + preset matrix | Cross-ref §42.45; no new runtime code |
| 3 | **SEC-PLANES-ADR-1** | ADR | P2 | **Done** | ADR: SecurityDefensePlugin EP + S1/S2/S3 plane discipline | Linked from §42.45 + hub; `check_harness_adr.py` green |
| 4 | **SEC-EXT-1** | Code | P1 | **Done** | `SecurityDefensePlugin` protocol + `SecurityInspectionResult` typed contract | Unit tests on protocol + schema |
| 5 | **SEC-EXT-2** | Code | P1 | **Done** | Entry point group `intergrax.security_defenses` + `register_security_defense_plugins()` | EP discovery gate; lab fixture package |
| 6 | **SEC-EXT-3** | Code | P1 | **Done** | Wire defense plugins via `security_runtime_bridge` → `MiddlewarePipeline` | Runs after native V-SEC middleware; before `ToolRuntime` |
| 7 | **SEC-EXT-4** | Code | P2 | **Done** | `ApplicationSecurityProfile.defense_plugin_ids` + `security_assembly_resolver` validation | Unknown plugin id fails wire-time on strict hosts |
| 8 | **SEC-EXT-5** | Test | P2 | **Done** | Lab reference plugin + gate tests (`tests/unit/runtime/security`) | Plugin on `BEFORE_TOOL_CALL` blocks + traces |
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

**Explicitly excluded:** standalone `SecurityEngine` tier or package; harness-native blockchain integration (M.6 exclusion); Tier-3 attestation/receipt products (product wiring only); new business agents — [§6.3a](.#63a-business-backlog-register-consolidated).

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

**Explicitly excluded:** harness-native blockchain; Tier-0 KMS SDK; SOC2/ISO certification artifacts; new business agents — [§6.3a](.#63a-business-backlog-register-consolidated).

---

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
