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

- **Implement / audit default:** Hub §6 · [`plan/satellites`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/TIER3_APPLICATION_ENVIRONMENT_audit_history.md`](plan/satellites/TIER3_APPLICATION_ENVIRONMENT_audit_history.md) · [`plan/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md`](plan/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/TIER3_APPLICATION_ENVIRONMENT.md`](../../technical/guides/audit_slices/TIER3_APPLICATION_ENVIRONMENT.md).
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
| [`plan/satellites/TIER3_APPLICATION_ENVIRONMENT_audit_history.md`](plan/satellites/TIER3_APPLICATION_ENVIRONMENT_audit_history.md) | audit history |
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
