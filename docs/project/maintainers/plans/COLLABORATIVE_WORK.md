# Collaborative Work — Implementation Plan

**Architecture (1:1):** [`architecture/COLLABORATIVE_WORK.md`](../../architecture/COLLABORATIVE_WORK.md)
**Feature coordination:** [`capabilities/plan/MULTIPLAYER_AI.md`](../../capabilities/plan/MULTIPLAYER_AI.md)
**Architecture governance:** [`architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)
**ADR:** [ADR-MP-001](../../technical/adr/entries/2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../../technical/adr/entries/2026-08-11/ADR-MP-002.md)

**Status:** Domain registered — **MP-1A ownership freeze complete; MP-1 runtime implementation NOT STARTED**
**First consumer:** `applications/local_workspace_application` (LKW)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Default:** this read-scope block + active `COLLAB-WORK-*` row only.
- **Architecture hub:** [`architecture/COLLABORATIVE_WORK.md`](../../architecture/COLLABORATIVE_WORK.md) read-scope block only.
- **Feature context:** [`capabilities/plan/MULTIPLAYER_AI.md`](../../capabilities/plan/MULTIPLAYER_AI.md) active `MP-*` section only.

---

## 1. Objective

Deliver platform-owned collaborative identity and authority primitives reusable across Tier-3 applications, with LKW as first consumer.

Delivery order: **platform architecture/contract slice → platform implementation → LKW adoption → LKW live proof**.

---

## 2. Delivery principles

1. **Single domain ownership** — no shared ownership with Nexus, Application Hosting, or LKW.
2. **Semantics before persistence** — contracts and effective-authority model precede storage choices.
3. **Policy enforces; Collaborative Work defines** — membership/delegation source of truth stays here.
4. **Execution stays in Nexus** — `RequestIdentity` and `DelegationSpec` remain execution contracts.
5. **Fail closed** — privileged collaborative mutations require provable effective authority.
6. **No LKW-owned platform types** — consumer integration only.

---

## 3. Implementation waves

### COLLAB-WORK-0 — Architecture and governance

| ID | Task | Status |
|----|------|--------|
| COLLAB-WORK-0A | ADR-MP-001 Collaborative Work Plane ownership | **Done** (MP-1A) |
| COLLAB-WORK-0B | ADR-MP-002 Principal / Membership / Delegation semantics | **Done** (MP-1A) |
| COLLAB-WORK-0C | Domain architecture + plan pair registration | **Done** (MP-1A) |
| COLLAB-WORK-0D | Multiplayer hub ownership synchronization | **Done** (MP-1A) |

COLLAB-WORK-0 closes with **0D Done**. Runtime implementation begins at **COLLAB-WORK-1A** after MP-1 review acceptance.

### COLLAB-WORK-1 — MP-1 architecture / contract slice

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1A |
| **Priority** | P0 |
| **Status** | **PLANNED / NOT STARTED** |
| **Purpose** | Principal, WorkspaceMembership, Delegation, and effective-authority contract slice |
| **Dependencies** | MP-0 accepted; ADR-MP-001 and ADR-MP-002 accepted |
| **Exact scope** | Semantic contracts in `intergrax/contracts/`; effective-authority resolution boundary; fail-closed enforcement hook design |
| **REUSED** | `MeaningfulSideEffectRequest` policy enforcement; `RequestIdentity` as execution-intake bridge only |
| **NEW** | Collaborative Principal, WorkspaceMembership, Delegation authority contracts |
| **Explicit out of scope** | DB models, repositories, APIs, HTTP routes, membership/delegation services, LKW changes, tests for runtime code, MP-2+ rows |
| **Acceptance** | Contracts frozen; effective authority intersection documented; membership explicit; delegation non-amplifying; execution contracts not duplicated |
| **Proof requirements** | Contract tests; isolation/authorization tests; fail-closed tests; delegation non-amplification tests (at implementation gate) |
| **Next step** | MP-1 review acceptance → COLLAB-WORK-1A implementation |

---

## 4. Out of scope (current phase)

- MP-2 WorkItem / Assignment implementation rows
- MP-3…MP-6 architecture or implementation rows
- LKW product adoption (MP-7)
- Runtime Python models beyond contract stubs when implementation gate opens
