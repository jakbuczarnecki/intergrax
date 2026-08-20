## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/CODE_CRAFT_implementation_history.md`](plan/satellites/CODE_CRAFT_implementation_history.md) | implementation history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


# Ephemeral Code Craft — Implementation Plan

**Architecture (1:1):** [`architecture/CODE_CRAFT.md`](../../architecture/CODE_CRAFT.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**ADR:** [`adr/entries/2026-06-10/ADR-CODECRAFT-001.md`](../../technical/adr/entries/2026-06-10/ADR-CODECRAFT-001.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Status:** **ECC-0…ECC-6 Done** · **S7–S11 post-closeout** (2026-06-13) · **Full Harness LC** (2026-06-17)  
**Last updated:** 2026-08-20 — Protocol v2 CODE_CRAFT audit persistence (CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY · CODECRAFT-VERIFICATION-INTEGRITY · CODECRAFT-ISOLATION-INTEGRITY); **P2-ARCH-12** CodeCraft safety boundary.  
**Default queue:** Phase **ECC** **closed** (2026-06-13); Protocol v2 remediation blocks **PLANNED**; default gate maintenance continues in [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (CODE_CRAFT plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/CODE_CRAFT_implementation_history.md`](plan/satellites/CODE_CRAFT_implementation_history.md). §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/CODE_CRAFT.md`](../../architecture/CODE_CRAFT.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Architecture doc alignment (P2-ARCH)

| ID | Scope | Status |
|----|-------|--------|
| **P2-ARCH-12** | Clarify CodeCraft safety boundary and promotion rules | **Done** (2026-06-20) |

---

## Delivery rules

1. **One ECC phase per PR** (or one cohesive sub-slice within a phase) → gate green → update this plan row.
2. **Contract first** — Pydantic models + Protocol before orchestrator wiring.
3. **Trace** — every state transition emits `CODECRAFT_*` (+ `RuntimeEvent` / `TraceEvent` where wired).
4. **Tests** — unit + integration; deterministic; no network in gate tests (mock sandbox).
5. **Reuse Tier-0** — extend sandbox, ToolRuntime, CVL; no parallel exec stacks.
6. **Fail closed** — deny paths must have policy tests.
7. **No product scope creep** — ECC harness only; no K.1/K.2 agents without §6.3 decision.

---

## §6 Maintenance queues

<a id="codecraft-identity-governance-integrity--protocol-v2-codecraft-2026-08-18"></a>

### CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY — Session authority, canonical HITL, override lattice (Protocol v2 · 2026-08-18)

**Status:** `PLANNED`
**Priority:** P0
**Type:** Arch / Wire / Proof
**Source:** [`AUDIT-20260818-CODE_CRAFT-01`](../../audit_results/2026-08-18/CODE_CRAFT.md), [`AUDIT-20260818-CODE_CRAFT-02`](../../audit_results/2026-08-18/CODE_CRAFT.md), [`AUDIT-20260818-CODE_CRAFT-03`](../../audit_results/2026-08-18/CODE_CRAFT.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- bind `CodeCraftSessionManager` and `EphemeralToolRegistryStore` to runtime-trusted tenant/task/run execution identity — every `get_state` / `iterate` / `promote` / `dispose` validates ownership; `craft_id` is not an authorization capability; conflicting `open()` fails closed; reuse canonical UER/identity contracts — no parallel string identity
- remove caller-controlled `hitl_approved` from tool inputs; authorize execution only from canonical Governed Execution / UER approval evidence scoped to exact task/run/craft/action; converge `codecraft.run` and iterative lifecycle on the same approval boundary — coordinate with **PG-FIX-C** / **IDT-FIX-C**
- model host/task `codecraft_mode` override lattice — narrow-only unless trusted policy explicitly approves expansion; host `disabled` cannot become executable from `task_metadata`

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Prior ECC/S7–S11 **Done** rows remain historical delivery facts.
- Verification/promotion gaps owned by **CODECRAFT-VERIFICATION-INTEGRITY** — not duplicated here.
- Isolation/egress gaps owned by **CODECRAFT-ISOLATION-INTEGRITY** — not duplicated here.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-CODE-CRAFT-PERSIST.

<a id="codecraft-verification-integrity--protocol-v2-codecraft-2026-08-18"></a>

### CODECRAFT-VERIFICATION-INTEGRITY — Promotion eligibility and same-sandbox verification (Protocol v2 · 2026-08-18)

**Status:** `PLANNED`
**Priority:** P0
**Type:** Arch / Wire / Proof
**Source:** [`AUDIT-20260818-CODE_CRAFT-04`](../../audit_results/2026-08-18/CODE_CRAFT.md), [`AUDIT-20260818-CODE_CRAFT-05`](../../audit_results/2026-08-18/CODE_CRAFT.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- promotion as evidence-consuming state transition — eligibility checks: last iteration verdict, static gate, execution success, required test result, required HITL, session lifecycle/status; `CraftResultPromoter` must preserve real verification evidence — never fabricate `success=True` / passed gate / `verdict="promote"`
- `promotion_schema_ref` resolves and validates against schema registry when configured — fail closed on missing/invalid schema
- `CraftTestRunner` executes against the exact sandbox session / artifact identity used for craft execution — no independent `resolve_sandbox_session(ctx)` re-resolution during verification
- reuse canonical CVL/verdict contracts where available — no parallel verdict stack

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Authority/HITL gaps owned by **CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY** — not duplicated here.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-CODE-CRAFT-PERSIST.

<a id="codecraft-isolation-integrity--protocol-v2-codecraft-2026-08-18"></a>

### CODECRAFT-ISOLATION-INTEGRITY — Anti-downgrade and network egress enforcement (Protocol v2 · 2026-08-18)

**Status:** `PLANNED`
**Priority:** P0
**Type:** Arch / Wire / Proof
**Source:** [`AUDIT-20260818-CODE_CRAFT-06`](../../audit_results/2026-08-18/CODE_CRAFT.md), [`AUDIT-20260818-CODE_CRAFT-07`](../../audit_results/2026-08-18/CODE_CRAFT.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- isolation tier as minimum security requirement — required `cloud`/`container` fails closed when eligible hosted substrate cannot resolve; no silent local downgrade unless explicit trusted downgrade policy defines allowed downgrade; regulated preset must not silently downgrade
- `network_egress` becomes runtime-enforced substrate capability — `deny` binds to substrate/network policy with provable outbound denial before generated code executes; fail closed when substrate cannot satisfy requested egress posture
- bind enforcement evidence to substrate capability — do not claim universal sandbox security

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Coordinate with Sandbox / Integration substrate owners — no duplicate isolation runtime.
- Authority gaps owned by **CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY** — not duplicated here.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-CODE-CRAFT-PERSIST.

---
