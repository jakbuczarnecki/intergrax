# Skills — Implementation Plan

**Architecture (1:1):** [`architecture/SKILLS.md`](../../architecture/SKILLS.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Last updated:** 2026-08-20 — Protocol v2 SKILLS audit persistence (SKILLS-AUTHORITY-INTEGRITY · SKILLS-IDENTITY-PROVENANCE · SKILLS-EVIDENCE-SYNC); SK-EXP through SK-EXP5 **Done** (plan header **150** skills · **42** bundles — drift vs architecture gate count tracked as SKILLS-EVIDENCE-SYNC); SK-BRIDGE.1/2 **Done**.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (SKILLS plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/SKILLS_implementation_history.md`](plan/satellites/SKILLS_implementation_history.md). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/SKILLS.md`](../../architecture/SKILLS.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/SKILLS_implementation_history.md`](plan/satellites/SKILLS_implementation_history.md) | implementation history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-12.1 | §12 Skills | LangGraph-compatible skill pack import path | P2 | **Done** |
| AUDIT-IDEAL-12.2 | §12 Skills | Dynamic skill selection L4 hook (AHI) | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

<a id="skills-authority-integrity--protocol-v2-skills-2026-08-18"></a>

### SKILLS-AUTHORITY-INTEGRITY — Explicit host Skill/Tool availability, fail-closed bootstrap, profile consistency (Protocol v2 · 2026-08-18)

**Status:** `PLANNED`
**Priority:** P1
**Type:** Arch / Wire / Proof
**Source:** [`AUDIT-20260818-SKILLS-01`](../../audit_results/2026-08-18/SKILLS.md), [`AUDIT-20260818-SKILLS-03`](../../audit_results/2026-08-18/SKILLS.md), [`AUDIT-20260818-SKILLS-05`](../../audit_results/2026-08-18/SKILLS.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- fail-closed production registration — missing host `SkillRegistry` / `SkillProfile` projection must not ambient-bootstrap `register_all_catalog_bundles=True`; explicit named laboratory/bootstrap mode only if retained
- non-expanding `ToolProfile` authority — skill-required `tool_ids` validate ⊆ host `ToolProfile.enabled`; static environment validation with actionable diagnostics; remove silent `extend_tool_profile_for_skills()` append semantics
- fail-fast `SkillProfile` references — explicit enabled skill/bundle ids unknown to catalog/registry fail environment validation; no silent ignore in `build_registry_from_profile()` / `is_skill_enabled()`
- coordinate with **TOOLS-GOVERNED-BOUNDARY-INTEGRITY** monotonic tool authority — reuse canonical owners, no second permission subsystem

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Prior SK-EXP / SK-BRIDGE **Done** rows remain historical delivery facts.
- Version/provenance gaps owned by **SKILLS-IDENTITY-PROVENANCE** — not duplicated here.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-SKILLS-PERSIST.

<a id="skills-identity-provenance--protocol-v2-skills-2026-08-18"></a>

### SKILLS-IDENTITY-PROVENANCE — Skill version identity and resolved capability provenance (Protocol v2 · 2026-08-18)

**Status:** `PLANNED`
**Priority:** P1
**Type:** Arch / Wire / Proof
**Source:** [`AUDIT-20260818-SKILLS-02`](../../audit_results/2026-08-18/SKILLS.md), [`AUDIT-20260818-SKILLS-04`](../../audit_results/2026-08-18/SKILLS.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- one explicit Skill version-identity model — (A) version-pinned `skill_id@version` resolution, or (B) logical `skill_id` with runtime/profile explicitly owning resolved version; no id-only resolution against registry current version while agent declares versioned manifests
- canonical `ResolvedSkillPack` provenance — retain or reference resolved snapshot (skill_ids, tool_ids, prompt/policy refs, max risk tier) bound to registered agent/runtime revision for audit, reproducibility, and future bridge enforcement
- do not duplicate Skill graph across parallel structures; do not invent compatibility aliases

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Operator must choose model A vs B before implementation — document decision in architecture hub.
- Authority/bootstrap/profile gaps owned by **SKILLS-AUTHORITY-INTEGRITY** — not duplicated here.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-SKILLS-PERSIST.

<a id="skills-evidence-sync--protocol-v2-skills-2026-08-18"></a>

### SKILLS-EVIDENCE-SYNC — Catalog count single source of truth (Protocol v2 · 2026-08-18)

**Status:** `PLANNED`
**Priority:** P2
**Type:** Docs / Proof
**Source:** [`AUDIT-20260818-SKILLS-06`](../../audit_results/2026-08-18/SKILLS.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- align architecture and plan current catalog counts from one authoritative gate/register (`test_sk_exp_skill_bundles.py` / `SHIPPED_SKILL_PLUGINS`)
- preserve historical counts only when explicitly labeled historical (e.g. SK-EXP closeout snapshots)
- documentation-owned remediation — no runtime implementation required unless gate/register itself drifts

**Remediation rules:**

- Revalidate gate count against then-current `development` HEAD before doc sync.
- Do not mark finding **CLOSED** until both owning docs publish aligned current counts.
- **Not implemented** by audit persistence task AUDIT-20260818-SKILLS-PERSIST — finding recorded; plan header drift intentionally preserved until this block executes.

---
