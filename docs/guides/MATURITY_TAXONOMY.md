# Intergrax — Maturity Taxonomy

**Status:** Canonical (2026-06-20)  
**Audience:** Architects, reviewers, implementation agents, external auditors  
**Audit ID:** P2-ARCH-02  
**Related:** [`SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md) · [`LAYER_COMPLETION_MODE.md`](LAYER_COMPLETION_MODE.md) · [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](INTERGRAX_DEVELOPMENT_STRATEGY.md) · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## 1. Purpose

Intergrax documentation and closeout artifacts use overlapping maturity words — *L3*, *L4*, *L5*, *production-ready*, *enterprise-ready*, *frozen*, *partial*, *beta*, *done*, *stable*, *scaffold*, *target*, *implemented* — without a shared definition. That ambiguity is dangerous: readers may assume production safety when only architecture or partial implementation exists.

This document is the **single cross-layer maturity vocabulary**. It defines **four independent axes**. A subsystem may be strong on one axis and weak on another; never collapse them into one headline label.

**This file is not a second canon.** Domain architecture remains authoritative for subsystem semantics. When maturity changes, update the domain pair first, then adjust any cross-layer summary that references it.

---

## 2. How to use

| Situation | Action |
|-----------|--------|
| **New or updated architecture doc** | Add a [Maturity Statement](#required-maturity-block) (or update existing one) |
| **Layer closeout / audit** | Score all four axes explicitly; map legacy labels before sign-off |
| **Code review / plan item** | Do not infer production readiness from architecture-only gates |
| **External audit** | Require four-axis statement; treat undifferentiated *L4* / *production-ready* as incomplete evidence |
| **LLM session start** | Hub → this file (labeling rule) → one domain pair |

**Relationship to other guides:**

| Guide | Role |
|-------|------|
| [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) | What **must never** break (independent of maturity level) |
| **This file** | How to **describe** how mature a subsystem is on each axis |
| [ADAPTIVE_HARNESS_INTELLIGENCE.md](../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md#governance-boundary) | AHI governance boundary, risk classes, production auto-apply rule |
| [ELASTIC_CAPACITY_AND_SCALING.md](../architecture/ELASTIC_CAPACITY_AND_SCALING.md#production-boundary) | ECP production boundary, scaling governance, autoscaling maturity claims |
| [CODE_CRAFT.md](../architecture/CODE_CRAFT.md#codecraft-safety-boundary) | CodeCraft safety boundary, execution modes, sandbox and promotion rules |
| [LAYER_COMPLETION_MODE.md](LAYER_COMPLETION_MODE.md) | Workflow for closing a domain layer; convergence scores must map to four axes |
| [INTEGRAX_HARNESS_AUDIT_MAP.md](INTEGRAX_HARNESS_AUDIT_MAP.md) | Audit layers and evidence expectations |

---

## 3. Labeling rule (normative)

A subsystem **MUST NOT** be described as **"production-ready"**, **"enterprise-ready"**, **"L5"**, **"complete"**, or **"done"** unless the document states **which maturity axis** is meant and gives the level on **all four axes** that apply.

**Correct example:**

```text
Architecture maturity: A4
Implementation maturity: I3
Production readiness: P2
Evidence maturity: E3
```

**Incorrect example:**

```text
This layer is L5 production-ready.
```

**Correct rewrite:**

```text
Architecture maturity: A5 (stable, enforced by invariants)
Implementation maturity: I4 (integrated across Nexus path)
Production readiness: P3 (controlled production candidate — tenant allowlist only)
Evidence maturity: E3 (integration/smoke on golden scenarios)
```

When only one axis is relevant (e.g. a design-only ADR), state the axis explicitly and mark others **N/A** with a one-line reason in **Notes**.

---

# Maturity Taxonomy

Four **independent** axes. Levels are **ordinal within an axis only** — **A4 does not imply I4 or P4**.

---

## 1. Architecture maturity

Describes how clearly a layer or subsystem has a defined architectural model, contracts, and boundaries — **in documentation and enforced design**, not in code volume.

| Level | Name | Meaning |
|-------|------|---------|
| **A0** | Not designed | No architectural intent; ad hoc or unknown |
| **A1** | Concept sketched | Problem and rough shape documented; contracts missing |
| **A2** | Architecture draft | Sections exist but gaps, conflicts, or open decisions remain |
| **A3** | Architecture defined | Normative contracts, boundaries, and canon sections are written and internally consistent |
| **A4** | Architecture validated against adjacent layers | Upstream/downstream pairs reviewed; cross-layer rules mapped (see [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md)) |
| **A5** | Architecture stable and enforced by invariants/checks | Canon stable; CI gates, invariants, or closeout scripts enforce the model |

**Typical signals:** domain pair completeness, ADR coverage, `SYS-INV-*` alignment, absence of unresolved architecture conflicts.

---

## 2. Implementation maturity

Describes how much of the architecture is **actually built in code** and wired into the harness — not plan status or doc completeness.

| Level | Name | Meaning |
|-------|------|---------|
| **I0** | Not implemented | No meaningful runtime code; may exist only in docs |
| **I1** | Skeleton / placeholder | Stubs, scaffolds, or types without usable behavior |
| **I2** | Partial implementation | Some paths work; major capabilities, integrations, or edge cases missing |
| **I3** | Usable implementation | Core behavior works for intended lab/dev scenarios |
| **I4** | Integrated implementation | Wired through Nexus / UAEP / policy / observability path as designed |
| **I5** | Hardened implementation with tests and operational safeguards | Broad test coverage, failure modes handled, ops hooks (metrics, rollback, quotas) in place |

**Typical signals:** pytest coverage on real paths, gate scripts green for the domain, absence of `NotImplementedError` on primary flows.

---

## 3. Production readiness maturity

Describes suitability for **real production environments** — safety, operability, limits, and blast radius — independent of how complete the architecture doc is.

| Level | Name | Meaning |
|-------|------|---------|
| **P0** | Not production safe | Known data-loss, security, or unbounded side-effect risks; dev-only |
| **P1** | Lab only | Safe in isolated harness lab; no production assumptions |
| **P2** | Internal prototype | Usable by the team behind controlled configs; undocumented limits |
| **P3** | Controlled production candidate | Production deploy allowed with explicit allowlists, feature flags, or tenant gates |
| **P4** | Production-ready with documented limits | Operated in production with known SLOs, runbooks, and documented boundaries |
| **P5** | Enterprise-grade with operational evidence | Multi-tenant or regulated posture; sustained operational proof (incidents, scale, compliance evidence) |

**Typical signals:** runbooks, on-call playbooks, quota/budget enforcement, HITL gates, release-board sign-off, operational evidence windows.

**Important:** **P4** and **P5** are the only levels where **"production-ready"** or **"enterprise-ready"** may appear — and only when paired with explicit **A**, **I**, and **E** levels in the same statement.

---

## 4. Evidence maturity

Describes **what proof exists** that the subsystem behaves as claimed — independent of intent (architecture) or code presence alone.

| Level | Name | Meaning |
|-------|------|---------|
| **E0** | No evidence | Assertions only |
| **E1** | Design rationale only | ADRs, architecture sections, or gate **contracts** without execution proof |
| **E2** | Unit-level evidence | Unit tests or isolated component checks |
| **E3** | Integration / smoke evidence | Multi-component or golden-scenario smoke in CI or lab |
| **E4** | End-to-end scenario evidence | Full harness paths (task → agent → tools → verify) on representative scenarios |
| **E5** | Real production / external deployment evidence | Measured behavior in production or customer-controlled deployment over a defined window |

**Typical signals:** CI artifacts, `phase_*_closeout_gate` reports, audit result files, operational evidence scripts (e.g. release-cycle JSON), production dashboards.

**Important:** Architecture gate evidence (e.g. `maturity_gate_evidence.py`, Phase V L3/L4 **contract** CI) often corresponds to **E1–E2**, not **E4** or **E5**, unless explicitly tied to executed scenarios.

---

## Legacy maturity labels

Historical Intergrax docs use shorthand that **does not map 1:1** to a single axis. **Do not delete legacy labels repo-wide in one pass.** For **new docs and edits**, translate to the four axes and keep legacy text only with an explicit mapping note.

| Legacy label | Typical historical meaning | Map to four axes (check each) |
|--------------|---------------------------|-------------------------------|
| **L3** | Phase V architecture hardening / contract closeout | Often **A4–A5** + **E1–E2**; **not** automatically **P4** or **I5** |
| **L4** | Governance or runtime adaptive loop milestone (e.g. L4-O … L4-V modes) | Split: governance **A4/A5** + **E2** vs runtime loop **I3–I4** + **E3–E4** |
| **L5** | Informal "max maturity" in older tables | **Deprecated** — replace with explicit **A/I/P/E** tuple |
| **production-ready** | Layer Completion "State B" or marketing language | Requires **P4** or **P5** plus stated **A**, **I**, **E** |
| **enterprise-ready** | Regulated / multi-tenant aspiration | Requires **P5** + **E5** evidence cited |
| **frozen** | Plan queue frozen (Band 3) or "State C" in LCM | **Process state**, not **A5** — say *plan frozen* vs *architecture stable* |
| **partial** | Incomplete implementation | Usually **I2**; state what works |
| **beta** | Exposed but limited | Usually **P2–P3** + **I3** |
| **done** / **complete** | Plan task or phase checkbox | Map to **I** and **E**; **done** in plan ≠ **P4** |
| **stable** | API or architecture stability | Usually **A5** or **I4** — specify which |
| **scaffold** / **target** | Scaffold code or future intent | **I1** or **A1–A2** respectively |
| **implemented** | Code exists | Usually **I2–I4** — specify integration depth |

**Phase / gate codes (L4-O, L4-R, …):** lifecycle **mode names** for Adaptive Harness Intelligence — document under **Implementation** and **Evidence** axes for the mode in question, not as a global "L4" badge for the whole platform.

**Migration policy:** domain pair owners add a [Maturity Statement](#required-maturity-block) when touching a file; bulk retrofits are optional follow-up work tracked in plan hubs.

---

## Required maturity block

Architecture and plan hub documents **SHOULD** include this block at the **top** (after title/metadata) or **bottom** (before maintenance). Use `N/A` only with a short **Notes** justification.

```markdown
## Maturity Statement

- Architecture maturity: A_
- Implementation maturity: I_
- Production readiness: P_
- Evidence maturity: E_
- Notes:
  - ...
```

**Example (honest partial state):**

```markdown
## Maturity Statement

- Architecture maturity: A4
- Implementation maturity: I3
- Production readiness: P2
- Evidence maturity: E3
- Notes:
  - Legacy label in §12: "L4 runtime target" — governance L4 Done (A5/E2); closed-loop runtime I3/E3.
  - Not production-ready; lab and golden-scenario smoke only.
```

---

## 5. Cross-axis patterns (informative)

Common **safe** combinations readers should understand:

| Pattern | Typical tuple | Reader takeaway |
|---------|---------------|-----------------|
| Architecture-first | A4, I1, P1, E1 | Designed but not built — do not deploy |
| Lab-usable | A3, I3, P2, E3 | Good for dev; not production |
| Gate-green, ops-thin | A5, I4, P2, E2 | CI contracts pass; need ops evidence before P4 |
| Production candidate | A4, I4, P3, E4 | Controlled rollout with allowlists |
| Enterprise posture | A5, I5, P5, E5 | Full four-axis claim allowed |

---

## 6. Maintenance

| Event | Action |
|-------|--------|
| Domain maturity changes | Update domain pair Maturity Statement; adjust hub summary if needed |
| New closeout gate or evidence script | Document which **E** level it satisfies in domain architecture |
| Layer Completion sign-off | Replace undifferentiated "production-ready" with four-axis tuple in final report |
| External audit | Require Maturity Statement on audited domain pair |

**Plan row:** [`PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md) **P2-ARCH-02** (canonical taxonomy).

---

## 7. Reading order

1. This file — [Labeling rule](#3-labeling-rule-normative) + four axes  
2. [`SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md) — rules that apply regardless of maturity  
3. Hub [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
4. One domain pair for your task  
5. Layer closeout: [`LAYER_COMPLETION_MODE.md`](LAYER_COMPLETION_MODE.md)
