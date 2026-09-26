# Governed Capability Fulfillment — Frozen Architecture

## 1. Metadata / status

| Field | Value |
| ----- | ----- |
| **Task** | `UCA-6C-R6-FREEZE` |
| **Scope** | Governed Capability Acquisition (UCA-6C R6) — architecture freeze record only |
| **Branch policy** | `development` |
| **Status** | **UCA-6C R6 governed capability fulfillment architecture = FROZEN** |
| **Artifact role** | Canonical maintainer **SSOT** for cross-domain ownership, canonical UCA flow, GCF invariants, and post-freeze evolution rules for this surface |

**Freeze record identity:** Git commit containing this freeze record (`FREEZE_RECORD_COMMIT`). Do not conflate with certified code baseline or certification evidence commit.

**Parent reconciliation (precursor, not long-term hub):** [`UCA_6C_CANONICAL_HITL_BOUNDARY_RECONCILIATION.md`](UCA_6C_CANONICAL_HITL_BOUNDARY_RECONCILIATION.md)

---

## 2. Purpose

This document formally **freezes** the architecture accepted by independent enterprise certification **`UCA-6C-R6-CERT-RERUN-3`**. Certification proved correctness at a fixed code baseline; it deliberately left **`ARCHITECTURE FROZEN = NO`** in the historical certification artifact. This freeze record is the **separate, post-certification** governance event that pins ownership, flow, invariants, boundaries, forbidden bypasses, and evolution rules so future work cannot silently drift UCA semantics.

This artifact is **documentation / governance only**. It does not change production behavior, contracts, tests, or CI.

---

## 3. Certified baseline and evidence

| Anchor | SHA | Role |
| ------ | --- | ---- |
| **CERTIFIED_UCA_CODE_BASELINE** | `a38f70bc878a4e61807ce93cb0bfa600fdaca168` | Audited **production / test seam** for UCA R6; frozen architecture semantics |
| **FINAL_CERTIFICATION_EVIDENCE_COMMIT** | `09f37f2c6b21a65803410c67847545ad50523c4a` | Final certification evidence (docs + recorded results) for **UCA-6C-R6-CERT-RERUN-3** |
| **FREEZE_RECORD_COMMIT** | *(this document’s containing commit)* | Declares architecture freeze; docs-only |

**Separation:** Evidence and freeze commits may advance on `development` without moving **CERTIFIED_UCA_CODE_BASELINE**. Parallel repository work (Memory, RAG, global program docs, static debt elsewhere) does not automatically recertify or re-baseline UCA.

---

## 4. Scope of freeze

Frozen surface includes, at minimum:

- Canonical **capability fulfillment flow** (discovery → gap → acquisition → qualification → binding → Execution admission → lifecycle → tools → governance → HITL → durable reentry)
- **Canonical ownership matrix** (§7)
- **GCF-INV-001–010** (§8)
- **Boundary rules**: contract-first, layer boundaries, no private cross-domain implementation coupling, no reflection/dynamic dispatch bypass, strong typing on certified seams
- **Forbidden bypass paths** (§11)
- **Extension model** via existing contracts without alternate lifecycle or authority (§12)

Freeze applies to **UCA / Governed Capability Fulfillment** cross-domain architecture, not the entire Integrax product.

---

## 5. Explicit out-of-scope

UCA R6 architecture freeze does **not** mean:

- Whole **Integrax** architecture is frozen (Core Platform freeze remains separate: [`INTEGRAX_CORE_PLATFORM_FREEZE.md`](../qualification/INTEGRAX_CORE_PLATFORM_FREEZE.md))
- All **Integrations** are frozen or migration-complete
- **RAG** or **Memory** roadmaps are closed
- Ban on new **providers**, **plugins**, or future **scenarios**
- Global **Pyright** / **Ruff** cleanliness for the full repo
- Assignment of UCA to remediate out-of-scope Integrations static debt noted in certification

---

## 6. Canonical capability fulfillment flow

```text
Worker recovery
    ↓
canonical capability discovery
    ↓
true capability gap
    ↓
Capability Acquisition coordinates acquisition
    ↓
Capability Qualification qualifies acquired capability
    ↓
binding
    ↓
canonical Execution request
    ↓
Execution Engine admission
    ↓
Execution Engine owns execution lifecycle
    ↓
ExecutionIdentityAuthority owns execution identity
    ↓
bound capability execution
    ↓
ToolRuntime
    ↓
exact tool invocation
    ↓
Governance authorities
    ↓
canonical Execution-owned HITL when required
    ↓
durable suspended-operation reentry
    ↓
same protected operation continues
```

**Hard rule:** **UCA / Autonomous Work ends its responsibility before Execution Engine owns execution lifecycle.** Worker capability recovery resume is **not** execution pause/resume; true execution pause/resume is **Execution Engine** via **`ExecutionContinuationPort`** on an already-admitted execution.

**Must not exist (architecturally):**

- Second execution lifecycle or execution runtime
- Second HITL or parallel governance path for the same concern
- UCA-owned or AW-owned **execution** resume
- Local governance bypass
- Direct tool execution bypassing **ToolRuntime**
- Public **Nexus** dependency in UCA flow
- Local minting of execution identity by UCA/AW consumers
- Pre-approval transport from AW/UCA into future invocation (`governance_approval_evidence` on ingress / dispatch / intake)

---

## 7. Canonical ownership matrix

Ownership is **frozen** as accepted in final certification. Each concern has exactly one canonical owner.

| Concern | Canonical owner |
| ------- | ----------------- |
| Need | Consumer |
| Capability discovery | Capability Catalog |
| Marketplace recommendation | Marketplace |
| Generic acquisition coordination | Capability Acquisition |
| Acquired-subject qualification | Capability Qualification |
| Provider / environment qualification | Core Qualification |
| Binding | Qualification / domain handoff |
| Worker responsibility recovery | Autonomous Work |
| Execution lifecycle | Execution Engine |
| Execution identity | ExecutionIdentityAuthority |
| Tool invocation | Tools / ToolRuntime |
| Policy / governance decision | Governance |
| Agent Governance approval | Agent Runtime Governance |
| Declarative HITL | Declarative Policy |
| Meaningful Side Effect authority | MSE Governance |
| Human execution pause/resume | ExecutionContinuationPort / Execution Engine |
| Durable suspended operation | SuspendedExecutionOperationStore boundary |
| Code synthesis | CodeCraft |
| Sandbox execution | Sandbox |
| Nexus orchestration | Execution Engine internals only |
| Relational category semantics | RelationalStoreIntegrationContract |
| SQLite provider mechanics | SQLite integration provider |
| Runtime SQLite composition | `runtime/persistence` |
| Collaborative Work persistence contract | Collaborative Work |
| Tool Registry runtime boundary | Tools subsystem |

**Invariant:** No duplicate semantic owner; no alternate bypass path for the same concern (certification architecture checkpoint 18× YES).

---

## 8. Frozen GCF invariants

| ID | Frozen rule | Meaning (summary) |
| -- | ----------- | ----------------- |
| **GCF-INV-001** | coordination ≠ ownership | Coordinators (e.g. acquisition) do not own lifecycle, identity, or execution |
| **GCF-INV-002** | qualification ≠ authorization | Qualification of capability/subject does not substitute governance authorization |
| **GCF-INV-003** | acquisition ≠ lifecycle | Acquisition coordination does not own execution lifecycle |
| **GCF-INV-004** | binding ≠ execution | Binding/handoff does not execute or resume execution lifecycle |
| **GCF-INV-005** | capability growth ≠ authority growth | New capability does not expand governance/execution authority by side effect |
| **GCF-INV-006** | no second HITL | Single canonical Execution-owned HITL path |
| **GCF-INV-007** | no second Execution Engine | Single execution lifecycle owner |
| **GCF-INV-008** | no public Nexus dependency | Nexus is EE-internal; UCA must not depend on public Nexus surface |
| **GCF-INV-009** | ToolRuntime mandatory | Tool invocation goes through ToolRuntime boundary |
| **GCF-INV-010** | true gap after canonical discovery | True capability gap only after complete canonical discovery |

Certification evidence: **GCF-INV-001–010 = PASS** at **CERTIFIED_UCA_CODE_BASELINE** (see §15). Do not add alternate invariants here; evolve only via Class C reopen.

---

## 9. Frozen boundary rules

### Contract-first

Platform integration uses **contracts / protocols / ports**, not ad hoc concrete coupling. Providers, plugins, adapters, and strategies must remain **replaceable** through the existing contract surface.

### Layer boundaries

No domain may assume another domain’s responsibility (lifecycle, identity, governance decision, tool invocation enforcement, durable suspended-operation store, etc.).

### Private implementation boundary

Cross-domain use of **private** methods, fields, or helpers is forbidden for architectural dispatch and integration.

### Reflection and dynamic dispatch

Architectural bypass via `getattr`, `setattr`, `hasattr` probing, `eval`, `exec`, or dynamic string dispatch **instead of** typed contracts is incompatible with frozen architecture.

### Strong typing

Do not replace semantic contracts with `dict[str, Any]`, bare `object`, or loose payloads where a **hard contract** exists or is required on certified seams.

**Note:** This freeze record does not introduce new enforcement mechanisms; existing gates and certification tests remain the operational evidence.

---

## 10. Contract-first / pluginability rules

**Frozen architecture ≠ frozen product.**

Allowed extension:

```text
existing contract
    ↓
plugin / provider / strategy / adapter
    ↓
sanctioned composition
```

Without:

- Changing canonical owner for a frozen concern
- Bypassing Governance or Execution
- Alternate lifecycle or second runtime
- Semantic drift of frozen public contracts without **Architecture Reopen** (Class C)

Do not invent new extension points in this document; describe only the accepted model.

---

## 11. Forbidden bypasses

The following are **architecturally forbidden** for UCA / governed capability fulfillment:

| Bypass | Why forbidden |
| ------ | ------------- |
| AW/UCA **execution** resume | Only **ExecutionContinuationPort** / EE owns execution pause/resume |
| Pre-approval / `governance_approval_evidence` on AW→EE ingress | Approval at **exact** tool invocation under admitted execution |
| Consumer-supplied root **ExecutionId** on UCA ingress | **ExecutionIdentityAuthority** / EE mints identity |
| Direct tool invoke outside **ToolRuntime** | GCF-INV-009 |
| Second HITL stack | GCF-INV-006 |
| Parallel Execution Engine or lifecycle owner | GCF-INV-007 |
| Public Nexus import/surface in UCA packages | GCF-INV-008 |
| Acquisition or qualification owning lifecycle | GCF-INV-003, GCF-INV-001 |
| Binding step performing execution | GCF-INV-004 |
| Declaring capability gap before canonical discovery completes | GCF-INV-010 |
| Registry mutation as acquisition “shortcut” | Violates coordination-only acquisition |
| Reflection / untyped payload bypass | §9 |

---

## 12. Extension model

Post-freeze feature work should add capability through **Class A** extensions (§13) unless proven Class B or required Class C. Extensions must preserve §6 flow and §7 owners.

---

## 13. Post-freeze change classification reference

UCA post-freeze evolution reuses platform **A / B / C** governance — **single SSOT:**

[`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md`](../qualification/INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md)

Do **not** create a parallel UCA-only A/B/C framework. Map UCA-impacting changes as follows:

### Class A — Safe extension

Uses existing contracts; does **not** change frozen ownership, GCF invariants, canonical flow, or public frozen semantics.

*Example:* new provider/plugin/strategy behind an existing port.

### Class B — Core-compatible implementation change

Touches frozen **implementation** surface without changing public contract semantics, canonical owner, lifecycle, authority, or canonical flow. Requires appropriate regression and independent audit per platform governance.

### Class C — Architecture / contract evolution

Touches **any** of: GCF-INV-001–010, canonical owner, public frozen contract semantics, Execution ownership, Execution identity authority, Governance/HITL authority, ToolRuntime mandatory boundary, canonical flow, domain boundary.

**Requires:**

```text
Architecture Reopen
→ scoped analysis
→ appropriate gates
→ scoped/full recertification
→ independent GitHub audit
→ explicit re-freeze
```

**Fail-safe:** If change cannot be proved Class A or B → **Class C**.

---

## 14. Architecture reopen rule

Any Class C change to this frozen surface must produce a new certification cycle and a **new explicit re-freeze** record. Until then, **CERTIFIED_UCA_CODE_BASELINE** and semantics in this document remain authoritative.

---

## 15. Evidence / certification anchors

| Artifact | Role |
| -------- | ---- |
| [`UCA_6C_R6_ENTERPRISE_CERTIFICATION.md`](../qualification/UCA_6C_R6_ENTERPRISE_CERTIFICATION.md) | **Evidence authority** for UCA-6C-R6-CERT-RERUN-3 (historical `ARCHITECTURE FROZEN = NO` preserved) |
| [`UCA_6C_CANONICAL_HITL_BOUNDARY_RECONCILIATION.md`](UCA_6C_CANONICAL_HITL_BOUNDARY_RECONCILIATION.md) | Reconciliation precursor; flow and GCF intent |
| [`EXECUTION_ENGINE_OWNERSHIP_MODEL.md`](EXECUTION_ENGINE_OWNERSHIP_MODEL.md) | Execution ownership model |
| ADR: [`ADR-UCA-6C-AGENT-GOVERNANCE-CANONICAL-HITL-RECONCILIATION.md`](../../technical/adr/entries/2026-09-23/ADR-UCA-6C-AGENT-GOVERNANCE-CANONICAL-HITL-RECONCILIATION.md) | HITL reconciliation |
| ADR: [`ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md`](../../technical/adr/entries/2026-09-22/ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md) | Continuation integration |
| ADR: [`ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY.md`](../../technical/adr/entries/2026-09-22/ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY.md) | Durable reentry |

**Certification summary (pointer only, not duplicated):** UCA-6C-R6-CERT-RERUN-3 — GCF-INV-001–010 PASS; C-01–C-20 PASS; T1–T8 qualification evidence; B1–B12 CLOSED; architecture checkpoint 18× YES; blocker count = 0; independent GitHub acceptance of evidence commit required before relying on freeze in production governance.

**Representative gate families (existing, not a new framework):**

- `tests/unit/runtime/architecture/test_uca6c_r6_architecture_gates.py`
- `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py`
- `tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py`

Plus UCA certification test families listed in the enterprise certification document.

---

## 16. Baseline vs HEAD policy

```text
repository HEAD does not automatically advance frozen UCA baseline
```

Later commits on `development` (including Memory/RAG or documentation) do **not** change **CERTIFIED_UCA_CODE_BASELINE** unless explicitly updated through **scoped/full recertification** and **explicit re-freeze**.

| Term | Meaning |
| ---- | ------- |
| **CERTIFIED_UCA_CODE_BASELINE** | Frozen UCA **code** semantics |
| **FINAL_CERTIFICATION_EVIDENCE_COMMIT** | Frozen **certification evidence** anchor |
| **FREEZE_RECORD_COMMIT** | This architecture freeze declaration |
| **Repository HEAD** | Current tip; may diverge from UCA baseline by design |

---

## 17. Freeze verdict

```text
UCA-6C R6 ARCHITECTURE FREEZE = ACCEPTED

CERTIFIED_UCA_CODE_BASELINE
= a38f70bc878a4e61807ce93cb0bfa600fdaca168

FINAL_CERTIFICATION_EVIDENCE_COMMIT
= 09f37f2c6b21a65803410c67847545ad50523c4a

GCF-INV-001–010
= FROZEN

CANONICAL OWNERSHIP
= FROZEN

CANONICAL UCA FLOW
= FROZEN

POST-FREEZE EVOLUTION
= GOVERNED BY EXISTING INTEGRAX A/B/C CHANGE CONTROL
```

**Not claimed:** `WHOLE INTEGRAX ARCHITECTURE = FROZEN`

---

**Independent audit:** Freeze acceptance on GitHub requires audit of the actual **FREEZE_RECORD_COMMIT**, this document’s content, and consistency with **FINAL_CERTIFICATION_EVIDENCE_COMMIT** and **CERTIFIED_UCA_CODE_BASELINE** — not this file alone before commit exists.
