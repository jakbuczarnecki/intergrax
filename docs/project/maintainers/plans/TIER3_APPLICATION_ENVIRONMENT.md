# Tier3 Application Environment — Implementation Plan

**Architecture (1:1):** [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Cross-plan — Agent layer (ACP):** Tier-3 hosts supply `ApplicationEnvironmentProfile`, `AgentBinding`, intake `RequestIdentity`, and org envelope — consumed by agent `merge_environment` (architecture ACP §30 · TIER3 §39). Implementation synced in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **Wave 2** (`ACP-DX-2`, `ACP-DX-5`) and **Wave 6** (`ACP-ORG-1..2`). Host PRs that change profile merge order MUST update agent plan acceptance tests.

**Cross-plan — Profile bundles (APP-EVOL-8):** Hierarchical bundles (architecture §22.6 · ADR-APP-003) preserve flat property shims in M1 — no ACP contract change until `spec_version` 2.0 (M3).

**Cross-plan — Application Hosting (`APPLICATION_HOSTING`):** Tier-3 owns only:

```text
application factory/runtime bridge
application definition and composition
ApplicationHost execution hooks
future HarnessApplication.hosting(...) facade integration
```

Application Hosting owns:

```text
HostedApplicationProfile
HostedApplicationHooks
HostedApplicationComponent
HostedApplicationEngine
HostedApplicationSupervisor
InstanceGuard
hosting lifecycle and readiness
signals and shutdown
restart policy execution
generic OS adapters
```

Concrete hosting implementation rows remain tracked exclusively in [`plan/APPLICATION_HOSTING.md`](APPLICATION_HOSTING.md). Do not copy the `APP-HOST-*` backlog into this plan.

**Application authoring canon (APP-CON):** architecture §24–§51 — symmetric to ACP §12–§45 for Tier-3 environments. **Evolution canon (APP-EVOL):** architecture §49. **Operations canon (APP-OPS):** architecture §50. **Freeze audit:** [`guides/GOVERNANCE_CONSISTENCY_AUDIT.md`](../../technical/guides/GOVERNANCE_CONSISTENCY_AUDIT.md). Phases **H-APP-CON** · **H-APP-EVOL** · **H-APP-OPS** · **H-APP-FREEZE** below.

**Fidelity rule:** Every architecture §20–§51 normative row MUST map to a plan ID in [§Architecture fidelity matrix](.#architecture-fidelity-matrix--20-51) and a verification artifact in [§Fidelity verification gates](.#fidelity-verification-gates). Completing the **open APP-\*** backlog is sufficient for implementation to match frozen architecture — no new primitives without ADR.

**Last updated:** 2026-06-17 — **Full Harness LC** (re-validates H-APP + APP-CON/EVOL/OPS closeout).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (TIER3_APPLICATION_ENVIRONMENT plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/TIER3_APPLICATION_ENVIRONMENT_implementation_history.md`](plan/satellites/TIER3_APPLICATION_ENVIRONMENT_implementation_history.md) · [`plan/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md`](plan/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## §6.2y Phase APP backlog execution order (post-freeze)

Recommended PR sequence — one APP ID per PR:

```text
1.  APP-PROD-9      wire production gates to CI
2.  APP-CON-3       env state lifecycle sync on hooks — **Done**
3.  ACP-TOK-1..3    (agent plan) budget enforcement — **Done** · unblocks §43
4.  APP-PROD-7      budget gate on STRICT hosts
5.  APP-CON-5       hook timeout / error handling
6.  APP-CON-6       artifact bundle on ApplicationRunSummary
7.  APP-CON-8       shadow/sandbox cleanup + APP-PROD-8 — **Done**
8.  APP-EVOL-1      EnvironmentSnapshot on intake — **Done**
9.  APP-OPS-1       capability graph STRICT deploy gate — **Done**
10. APP-OPS-2       application ownership on manifest — **Done**
11. APP-CON-7       scenario matrix tests — **Done**
12. APP-EVOL-2/2b   migrations — **Done**
13. APP-EVOL-3..7   evolution + packaging — **Done**
14. APP-OPS-3/4     health score + registries — **Done**
15. APP-CON-DX.*    author guide + audit prompt — **Done**
```

**Cross-plan:** steps 3–4 require [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **ACP-FINISH** / **ACP-TOK-***.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/TIER3_APPLICATION_ENVIRONMENT_implementation_history.md`](plan/satellites/TIER3_APPLICATION_ENVIRONMENT_implementation_history.md) | implementation history |
| [`plan/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md`](plan/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §26 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-09) — AUDIT-IDEAL Tier-3 rows closed

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-3.2 | §3 Intake | Product host intake parity (streaming + durable async default) | P2 | **Done** |
| AUDIT-IDEAL-28.1 | §28 Tier-3 | Durable async queue default beyond SQLite (DEBT-28-01) | P1 | **Done** |
| AUDIT-IDEAL-28.2 | §28 Tier-3 | Queue worker scaffold-default (`INCLUDE_QUEUE_WORKER`) | P1 | **Done** |
| AUDIT-IDEAL-28.3 | §28 Tier-3 | LKW hybrid daemon (CFG-14) | P4 | **Done** |
| AUDIT-IDEAL-28.4 | §28 Tier-3 | Business agents K.1/K.2 certification + deploy | P4 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

### Protocol v2 remediation — TIER_LAYER_BOUNDARIES (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/TIER_LAYER_BOUNDARIES.md`](../../audit_results/2026-08-18/TIER_LAYER_BOUNDARIES.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings — **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-TIER-LAYER-PERSIST.

| Block | Status | Findings | Acceptance intent |
|-------|--------|----------|-------------------|
| **TL-FIX-C** | ACCEPTED / PLANNED | [`AUDIT-20260818-TIER_LAYER_BOUNDARIES-03`](../../audit_results/2026-08-18/TIER_LAYER_BOUNDARIES.md) | Generic deployment/profile contract contains no LKW-specific vocabulary; product-specific deployment configuration is application-owned or typed extension-owned; migrate consumers cleanly after revalidation |
| **TL-FIX-D** | ACCEPTED / PLANNED | [`AUDIT-20260818-TIER_LAYER_BOUNDARIES-04`](../../audit_results/2026-08-18/TIER_LAYER_BOUNDARIES.md) | Public typed execution-adapter/run-service composition mechanism; Legal and Dispute Sim no longer assign `_execution_adapter` directly; focused tests prove inline + queued composition |

Finding 05 remains owned by **TL-FIX-A** in [`PLATFORM_FOUNDATION` plan](PLATFORM_FOUNDATION.md) — not duplicated here.

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Historical Done/READY_FOR_CLOSE rows (including LKW hybrid) remain historical.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).

---

### Protocol v2.2 remediation — INTERFACE_TASK_INTAKE (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/INTERFACE_TASK_INTAKE.md`](../../audit_results/2026-08-18/INTERFACE_TASK_INTAKE.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings — **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-INTERFACE-TASK-INTAKE-PERSIST.

| Block | Status | Findings | Dependencies | Acceptance intent |
|-------|--------|----------|--------------|-------------------|
| **ITI-FIX-A** | ACCEPTED / PLANNED | [`AUDIT-20260818-INTERFACE_TASK_INTAKE-01`](../../audit_results/2026-08-18/INTERFACE_TASK_INTAKE.md), [`04`](../../audit_results/2026-08-18/INTERFACE_TASK_INTAKE.md) | — | One canonical normalized intake contract; surface adapters become thin adapters; preserve typed intake semantics; bounded legacy metadata bridge only |
| **ITI-FIX-D** | ACCEPTED / PLANNED | [`AUDIT-20260818-INTERFACE_TASK_INTAKE-06`](../../audit_results/2026-08-18/INTERFACE_TASK_INTAKE.md) | ITI-FIX-A, ITI-FIX-B, ITI-FIX-C | Replace flag-only intake parity proof with E2E product-host evidence: stream/chunks → `TaskEnvelope` → `Task` → `UnifiedTaskRunner` → `TaskResult`; cross-reference **ITI-FIX-C** for interaction runner convergence |

**Qualification:** Historical AUDIT-IDEAL Done labels (including AUDIT-IDEAL-3.2) remain historical delivery facts. Protocol v2.2 accepted the new intake parity and normalization gaps above — do not silently rewrite historical Done history.

**Remediation rules:** same as TIER_LAYER_BOUNDARIES block above.

---

### Protocol v2.2 remediation — IDENTITY_TRUST (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/IDENTITY_TRUST.md`](../../audit_results/2026-08-18/IDENTITY_TRUST.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings — **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-IDENTITY-TRUST-PERSIST.

#### IDT-FIX-A — Authenticated principal spine

**Status:** `ACCEPTED / PLANNED`
**Source:** [`AUDIT-20260818-IDENTITY_TRUST-01`](../../audit_results/2026-08-18/IDENTITY_TRUST.md), [`AUDIT-20260818-IDENTITY_TRUST-06`](../../audit_results/2026-08-18/IDENTITY_TRUST.md)

**Acceptance criteria:**

- One canonical verified-principal → `RequestIdentity` path for authenticated Tier-3 intake.
- No untrusted metadata/body override of verified `tenant_id` / `user_id` / principal fields.
- Resolve `ActorIdentity` / `RequestIdentity` ownership and wire semantics (writer/reader contract).
- Product/reference-host conformance tests eventually required.
- No credential/token persistence in runtime state.

**Remediation rules:** same as TIER_LAYER_BOUNDARIES block above.

---

### Protocol v2.2 remediation — LLM_ADAPTERS (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/LLM_ADAPTERS.md`](../../audit_results/2026-08-18/LLM_ADAPTERS.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings — **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-BATCH-PERSIST-2.

| Block | Status | Findings | Dependencies | Acceptance intent |
|-------|--------|----------|--------------|-------------------|
| **LLM-FIX-A** | ACCEPTED / PLANNED | [`AUDIT-20260818-LLM_ADAPTERS-01`](../../audit_results/2026-08-18/LLM_ADAPTERS.md), [`04`](../../audit_results/2026-08-18/LLM_ADAPTERS.md) | — | Universal PRE_MODEL/inference boundary for classifier, planner, and per-retry inference |
| **LLM-FIX-B** | ACCEPTED / PLANNED | [`AUDIT-20260818-LLM_ADAPTERS-02`](../../audit_results/2026-08-18/LLM_ADAPTERS.md), [`05`](../../audit_results/2026-08-18/LLM_ADAPTERS.md) | — | Decision-to-execution model/provider binding; trace attributes actual runtime provider |
| **LLM-FIX-C** | ACCEPTED / PLANNED | [`AUDIT-20260818-LLM_ADAPTERS-03`](../../audit_results/2026-08-18/LLM_ADAPTERS.md) | LLM-FIX-A | Governed failover: per-candidate authorization or pre-authorized immutable candidate set |
| **LLM-FIX-D** | ACCEPTED / PLANNED | [`AUDIT-20260818-LLM_ADAPTERS-06`](../../audit_results/2026-08-18/LLM_ADAPTERS.md) | IDT-FIX-D | LLM execution identity closure on planning/classification paths |

**Remediation rules:** same as TIER_LAYER_BOUNDARIES block above.

---

### Protocol v2 remediation — TIER3_APPLICATION_ENVIRONMENT (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/TIER3_APPLICATION_ENVIRONMENT.md`](../../audit_results/2026-08-18/TIER3_APPLICATION_ENVIRONMENT.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings — **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-TIER3-APPLICATION-ENVIRONMENT-PERSIST.

#### T3-COMPOSITION-AUTHORITY-INTEGRITY — typed configuration authority and blocking conformance

**Priority:** P0/P1
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-01`](../../audit_results/2026-08-18/TIER3_APPLICATION_ENVIRONMENT.md), [`04`](../../audit_results/2026-08-18/TIER3_APPLICATION_ENVIRONMENT.md), [`06`](../../audit_results/2026-08-18/TIER3_APPLICATION_ENVIRONMENT.md)

**Acceptance intent:**

- Canonical Tier-3 composition has one typed configuration authority and blocking conformance semantics where required.
- Product/STRICT required profile/roster invariant violations fail closed; advisory mode is explicit policy only.
- Integration-profile authority is contractual — conflicting authoritative inputs rejected or typed-merge resolved.
- Platform-significant wiring artifacts (including policy/governance) use typed contracts, not `Any`.
- Cross-link **TL-FIX-C/D** where ownership overlaps; do not duplicate boundary remediation.

#### T3-RUNTIME-SCOPE-INTEGRITY — canonical event spine and execution-scoped sandbox identity

**Priority:** P0
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-02`](../../audit_results/2026-08-18/TIER3_APPLICATION_ENVIRONMENT.md), [`03`](../../audit_results/2026-08-18/TIER3_APPLICATION_ENVIRONMENT.md)

**Acceptance intent:**

- Tier-3 reuses the canonical `RuntimeEventBus` from the runtime composition root; no silent local bus minting in canonical production wiring.
- Task-scoped sandbox session materializes from canonical execution identity (tenant + TaskId / required scope); no synthetic `harness/bootstrap` runtime ownership in reusable production paths.
- Cross-link [`OBSERVABILITY_EVIDENCE` plan](OBSERVABILITY_EVIDENCE.md) and [`IDENTITY_TRUST` plan](IDENTITY_TRUST.md) / **IDT-FIX-A** — do not create parallel event or identity subsystems.

#### T3-SNAPSHOT-PROVENANCE-INTEGRITY — Task/Run ↔ EnvironmentSnapshot binding

**Priority:** P1/P2
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-TIER3_APPLICATION_ENVIRONMENT-05`](../../audit_results/2026-08-18/TIER3_APPLICATION_ENVIRONMENT.md)

**Acceptance intent:**

- Configuration snapshot evidence is explicitly bound to the Task/Run execution that consumed it.
- Deploy snapshot semantics and execution binding evidence are distinct and provable.
- Reuse canonical execution identity; separate typed binding artifact is acceptable.

**Remediation rules:** same as TIER_LAYER_BOUNDARIES block above.

---
