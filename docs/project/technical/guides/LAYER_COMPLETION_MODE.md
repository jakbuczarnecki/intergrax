# Intergrax — Layer Completion Mode

**Status:** Canonical (2026-06-17)  
**Audience:** Maintainers, architects, implementation agents  
**Related:** [INTERGRAX_DEVELOPMENT_STRATEGY.md](INTERGRAX_DEVELOPMENT_STRATEGY.md) · [`architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) · [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) · [MATURITY_TAXONOMY.md](MATURITY_TAXONOMY.md) · [audit_results/AUDIT_PROTOCOL.md](../../../audit_results/AUDIT_PROTOCOL.md) · [EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md](../../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md#architecture-vs-implementation-rules-boundary) (architecture vs implementation rules)

**Cursor iteration (every session):** [`.cursor/rules/intergrax-iteration.mdc`](../../../../.cursor/rules/intergrax-iteration.mdc) — single-item gate maintenance; **this guide** is the extended workflow for **closing an entire harness layer** to architectural maturity.

---

## 1. Purpose

Work **only** on the **designated architecture layer** and its **direct dependencies**.

Do **not** audit the whole repository unless required to judge alignment of that layer.

The goal is not merely to implement existing architecture. The goal is to bring the layer to the **best achievable** architectural, production, and implementation state.

---

## 2. Scope and relationship to other instructions

| Instruction | Granularity | When |
|-------------|-------------|------|
| [`.cursor/rules/intergrax-iteration.mdc`](../../../../.cursor/rules/intergrax-iteration.mdc) | One coherent plan item | Default Cursor sessions |
| **This guide (Layer Completion Mode)** | Full domain layer → maturity | Deep layer closeout (journal: “Layer Completion Mode”) |
| [audit_results/AUDIT_REMEDIATION_PROTOCOL.md](../../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md) | Remediate accepted audit findings after platform audit | Ad-hoc closeout without audit evidence |
| [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) | Cross-domain never-violate rules | Skim **before every** Step 1 and Step 6 |
| [AGENT_INSTRUCTIONS.md](AGENT_INSTRUCTIONS.md) | Hard repo rules, verification bundle | On demand (`@docs/project/technical/guides/AGENT_INSTRUCTIONS.md`) |
| [AGENTS.md](../../../../AGENTS.md) | Cursor auto-load stub (tiers, boundaries) | Always (Cursor) |

**Operator paste:** canonical multi-domain bootstrap lives under [`../../maintainers/bootstrap`](../../maintainers/bootstrap/README.md) (English). Personal PL notes may stay in `docs/_external` (gitignored). **This file is the linkable canonical definition** for single-domain LCM steps.

---

## 3. Repository bootstrap (before Step 1)

Read **only** documents required for the designated layer — do **not** load all of `docs`.

| Priority | Document | Role |
|----------|----------|------|
| 1 | [INTERGRAX_DEVELOPMENT_STRATEGY.md](INTERGRAX_DEVELOPMENT_STRATEGY.md) | Strategic goal, decision hierarchy, work cycle |
| 2 | [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) | `SYS-INV-*` never-violate index (P2-ARCH-01) — mandatory skim |
| 3 | [intergrax_runtime_architecture.md](../../architecture/intergrax_runtime_architecture.md) | Hub — domain pair picker |
| 4 | `docs/project/architecture/<DOMAIN>.md` + `docs/project/maintainers/plans/<DOMAIN>.md` | Canon + plan for **this layer only** (1:1 basename) |
| 4b | [`capabilities/README.md`](../../capabilities/README.md) + matching feature pair | When closing a **multi-layer feature** — feature architecture + feature plan, then smallest domain-owned slice |
| 5 | [audit_results/AUDIT_PROTOCOL.md](../../../audit_results/AUDIT_PROTOCOL.md) | Adversarial layer audit procedure and campaign model |
| 6 | [AGENT_INSTRUCTIONS.md](AGENT_INSTRUCTIONS.md) | Tier imports, scope, verification commands |
| 7 | [IDEAL_HARNESS_AI_ARCHITECTURE.md](IDEAL_HARNESS_AI_ARCHITECTURE.md) | North star for Step 1A strategic review |

**Reading rule:** one `architecture/<DOMAIN>.md` ↔ `plan/<DOMAIN>.md` pair per iteration. No monolithic implementation plan file.

**Architectural safety fuse:** if domain plan, domain architecture, and ideal architecture conflict — **STOP**, explain to the operator, propose doc updates **before** touching code.

**Strategic frame:** the Harness is the product; agents are replaceable. Default platform queue is gate maintenance ([`plan/PLATFORM_FOUNDATION.md`](../../maintainers/plans/PLATFORM_FOUNDATION.md) §6.1) unless the operator selects otherwise. Phase K and §6.3 — **do not start** without explicit reprioritization.

**Language:** repository artifacts (code, docs, tests, commits) — **English**. Operator-facing session communication — operator session language.

---

## 4. Objective

**Governance prerequisite** ([`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)): layer closeout assumes ownership is established, architecture and plan are canonical for the domain, the capability is not hidden inside an application, and first-adopter proof follows platform implementation (`PLATFORM-INV-002`, `PLATFORM-INV-005`).

Bring the designated layer to a state that is:

- complete,
- consistent,
- production-ready,
- aligned with architecture,
- aligned with the implementation plan,
- ready for long-term evolution.

**Maturity vocabulary:** terms like *complete*, *production-ready*, and *frozen* (Step 12 State C) **MUST** be expressed as explicit [four-axis maturity](MATURITY_TAXONOMY.md) (**A** / **I** / **P** / **E**) in the final report — see [MATURITY_TAXONOMY.md §3](MATURITY_TAXONOMY.md#3-labeling-rule-normative).

---

## 5. Step 1 — Layer audit

**First:** skim [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md). Classify every `SYS-INV-*` violation at minimum as **P0** (see Step 3).

Analyze:

- layer architecture documentation,
- implementation plan,
- current implementation,
- inbound dependencies,
- outbound dependencies,
- existing contracts,
- public APIs,
- extension points,
- configuration model,
- testing model.

Detect:

- `SYS-INV-*` violations (full index: [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md)),
- architectural gaps,
- missing contracts,
- missing components,
- unimplemented capabilities,
- wrong abstractions,
- documentation inconsistencies,
- technical debt,
- architecture rule violations.

---

## 6. Step 1A — Strategic architecture review

Before updating architecture, perform a strategic review of the layer.

Assume current architecture may be incomplete. Actively look for:

- missing capabilities,
- missing extension points,
- missing contracts,
- missing production mechanisms,
- missing runtime components,
- modern architectural patterns,
- patterns used in comparable agent frameworks,
- limitations of current design,
- future scalability risks,
- maintainability risks.

Challenge existing assumptions. Do **not** treat current architecture as ideal.

For each proposal present:

| Field | Content |
|-------|---------|
| Name | Short identifier |
| Description | What changes |
| Rationale | Why |
| Benefits | What improves |
| Implementation cost | Effort / risk |
| Complexity impact | On platform |
| Recommendation | **Recommend** · **Optional** · **Reject** |

Do **not** update architecture yet. Present proposals to the operator. Wait for accept/reject. Proceed only after acceptance.

---

## 7. Step 2 — Architecture and documentation update

After accepted changes, update **first**:

- architecture documentation,
- technical documentation,
- implementation plan,
- roadmap (if applicable),
- diagrams,
- contracts.

Documentation is the source of truth. Do **not** start implementation before documentation updates are complete.

Significant architecture decisions require an ADR per [`docs/project/technical/adr/README.md`](../adr/README.md) (or explicit “no ADR needed” in plan/journal).

When finished:

- propose a commit **only when the operator asks**,
- describe changes made.

---

## 8. Step 3 — Problem classification

Assign a severity level to each problem.

### P0 — Critical architecture defect

Prevents correct layer behavior or breaks fundamental architecture assumptions.

Examples:

- `SYS-INV-*` violation (e.g. agent orchestrates, kernel replans, tools bypass `ToolRuntime`, Tier-0 imports `agents`),
- wrong responsibility split,
- conflicting contracts,
- incorrect control flow,
- tier boundary violation.

### P1 — Production gap

Does not break architecture but prevents treating the layer as production-ready.

Examples:

- missing key components,
- missing tests,
- missing error handling,
- missing observability,
- missing security controls.

### P2 — Hardening

Quality and resilience improvements (extra validation, logging, safeguards).

### P3 — Optimization

Ergonomics, performance, maintainability.

### P4 — Future evolution

Ideas not required to declare the layer complete.

---

## 9. Step 4 — Implementation plan

Compare:

- target state in architecture,
- current implementation state.

Prepare sprints needed to reach alignment. Each sprint must include:

- scope,
- goal,
- completion criteria,
- file list,
- component list.

Sprints address **P0 and P1 only**. Place **P2, P3, P4** in the layer backlog.

---

## 10. Step 5 — Sprint execution

Execute sprints sequentially.

After each sprint:

- update documentation (**affected domain pair only**),
- update tests,
- run verification (minimum: `uv run pytest -m "gate and not no_ci" -q` plus scripts from [AGENT_INSTRUCTIONS.md](AGENT_INSTRUCTIONS.md) and [SYSTEM_INVARIANTS.md §7](SYSTEM_INVARIANTS.md#7-ci-enforcement-map-selected) relevant to the layer),
- verify alignment with architecture **and** [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md),
- commit **only when the operator explicitly requests**.

Advance to the next sprint **only** when the current sprint is closed and no P0/P1 blocker remains.

Do not stop without a real blocker — but **do not** expand scope beyond the designated layer without operator approval.

---

## 11. Step 6 — Final audit

After all sprints:

1. Re-audit the layer (including domain-relevant `SYS-INV-*` checklist).
2. Verify architecture alignment.
3. Verify implementation alignment.
4. Verify documentation alignment.
5. Verify contract alignment.
6. Verify implementation plan alignment.

If new **P0 or P1** issues appear → return to Step 2 and iterate.

If only **P2, P3, or P4** remain → do **not** start another full iteration; present backlog to the operator.

---

## 12. Architectural convergence check

After the final audit, score maturity (0–100% each):

- Architecture Completeness
- Production Readiness
- Documentation Consistency
- Implementation Consistency

Also publish a [Maturity Statement](MATURITY_TAXONOMY.md#required-maturity-block) (**A** / **I** / **P** / **E**) — percentages alone are not sufficient for cross-doc comparison.

### State A — Not ready

P0 or P1 issues remain. Layer needs another iteration.

### State B — Architecturally mature

No P0 or P1 issues. Layer is production-ready. Remaining work is P2–P4 only. **Recommend closing** layer completion work.

### State C — Frozen

Layer was previously declared mature. Further iterations yield little value. Changes are mainly hardening, optimization, or future extensions.

---

## 13. Completion criteria

A layer may be declared **complete** when:

- no P0 issues remain,
- no P1 issues remain,
- documentation is current,
- implementation plan is current,
- tests pass,
- **no open `SYS-INV-*` violations in layer scope**,
- layer is ready for production use.

P2, P3, and P4 do **not** block completion — record them in the layer backlog.

---

## 14. Final session report

Before ending the session, present:

### Changes made

- architectural,
- implementation,
- documentation.

### Sprints completed

- sprint list,
- completion criteria,
- test results.

### Commits

- commit list (if the operator requested commits),
- commit descriptions.

### Invariant compliance

- `SYS-INV-*` violations found and fixed,
- residual risks referencing [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md).

### Remaining risks

- all P2, P3, P4 items.

### Maturity scores

- Architecture Completeness
- Production Readiness
- Documentation Consistency
- Implementation Consistency

### Recommendation

One of:

- **Continue Iteration**
- **Architecturally Mature**
- **Frozen**

---

## 15. Maintenance

| Event | Action |
|-------|--------|
| Process change | Update this guide; keep personal operator paste in sync if used |
| New cross-domain rule | Update [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) first; reference here only if Step 1/6 checklist changes |
| Journal entry cites “Layer Completion Mode” | Link to this file |

**ADR:** no ADR needed for this guide unless process semantics change platform contracts.
