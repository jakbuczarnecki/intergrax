# PDC-X — Proposed roadmap closure text (PDC-P9-04 input) — DRAFT

> **DRAFT — proposal for the Operator, not a verdict.** Nothing in this file is written into the roadmap. The PDC-X closure line is **not delegated** (`DELEGATION-PDC-CLOSE.md`, "Not delegated"; PDC-P9-04); the Operator signs it once, after the P8 result `<P8 verdict>` at `<GLOBAL_AUDITED_SHA>`. If P8 does not give PASS or PASS_SCOPED with the exceptions below, **none of this text is used** and the stage reports `PARTIAL_EXACT_BLOCKERS` (roadmap §3.2 exit criteria; section 5 below).

## 1. Which vocabulary applies (open question, with a recommendation)

| Source | Wording |
| --- | --- |
| Master plan, PDC-P9-04 (`PLAN_T-IGX-036-A.md`) | `[x] CLOSED — INDEPENDENTLY ACCEPTED (…SCOPED if applicable) — PUBLISHED` via Protocol 03 B6 / IIAC-B6; "Authorized vocabulary only" |
| IIAC protocol, "Closure vocabulary (canonical)" (`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md:22-29`) | `{STAGE} — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`; "Use `(SCOPED)` and limitations in manifest and pass-record body — **not** in a false global roadmap headline." |
| `OD-PDC-IIAC-SCOPE.md` (APPROVED Option A) | Extends policy `IIAC-2026-11-05.1` to PDC-X; verdict vocabulary PASS, PASS_SCOPED (only for Operator-authorized exclusions), PARTIAL, FAIL |
| Precedent MCP-SERVER-X (roadmap §3 row and §5 ledger) | `CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED` with "(SCOPED, …)" in the body and `IIAC certification CERTIFIED_PASS_SCOPED` |
| Precedent EVOL-X / EBH-6 / DUP-X | `CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)` / `(AUTOMATION_FIRST SCOPED) — PUBLISHED` (human-B6 era wording) |

**Contradiction:** the master plan says "INDEPENDENTLY ACCEPTED" (B6 wording); the approved process for PDC-X is IIAC, whose canonical title is "INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION)". **Recommended: Variant A (IIAC canonical)**, because `OD-PDC-IIAC-SCOPE` Option A is the approved process and the verdict on B6 is not delegated. Variant B is given for completeness. The Operator chooses: `<Variant A | Variant B>`.

## 2. Proposed roadmap text (SCOPED variant: exceptions apply)

Placeholders are filled only from results that exist at signing. "SCOPED" is justified only because the exception register is non-empty; if P8 confirms no exception, drop the parentheses and the limitation sentence.

### 2.1 Headline status (Variant A, recommended)

```text
PDC-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED
```

with the body qualifier `(IIAC SCOPED)` as for `MCP-SERVER-X` and `ENT-AUDIT-X` (roadmap line 572).

### 2.2 Variant B (master-plan wording)

```text
[x] CLOSED — INDEPENDENTLY ACCEPTED (SCOPED) — PUBLISHED
```

### 2.3 Roadmap section 3 table row (currently line 498, status cell `**[ ] PLANNED / MANDATORY / CURRENT**`)

Replace the status cell with:

```text
**[x] CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED** (IIAC SCOPED; global audit `<P8 verdict>` @ `<GLOBAL_AUDITED_SHA>`; closure `<FINAL_CLOSURE_PUBLICATION_SHA>`)
```

and replace the sentence "**No FRZ promotion**; **225** unchanged. **Next:** **`ARCH-FREEZE`**. Product implementation **not started**." (it is stale: product work was implemented in P5-P7) with:

```text
FRZ coverage reconciled: **225 / 26** unchanged; promotions `<none | the rows signed in PDC_FRZ_PROMOTION_PROPOSAL.md>`. Declared exceptions and debt: `PDC_EXCEPTION_REGISTER.md`, `PDC_DEBT_REGISTER.md`. **Next:** **`ARCH-FREEZE`** (not entered; predecessor note satisfied by `PDC_ARCH_FREEZE_HANDOFF.md`).
```

### 2.4 Roadmap section 3.2 "Status" line (currently line 1847: `**[ ] PLANNED / MANDATORY / CURRENT**` (P0 roadmap registration only; no product change))

```text
**Status:** **`[x] CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`** (IIAC SCOPED). Global audit `<P8 verdict>` @ `<GLOBAL_AUDITED_SHA>`; EBH-7 replay run `<replay run id>` (not a B6 verdict); closure publication `<FINAL_CLOSURE_PUBLICATION_SHA>`.
```

and replace the "Implementation boundary (this roadmap revision)" paragraph (line 1884: "registration and planning only; **production delta = 0**; **no FRZ promotion**.") with:

```text
**Closure boundary:** PDC-X changed product paths in P5-P7 (domain batches B-D1..B-D8, P6 integration repairs, T-IGX-041-A..I) and certifies **Dimension A** only; live provider and environment qualification stays with `PROD-Q` (Path B, OD-PDC-PATHB-01). `dist_scale_x` and Docker (C10/C15) are not qualified and are deferred (declared exception, DELEGATION-PDC-WAVE2 decision 4). FRZ: 225 / 26 unchanged; see `PDC_FRZ_PROMOTION_PROPOSAL.md` as signed.
```

### 2.5 Roadmap "Next program stage" and "Position" lines (currently lines 570 and 572)

Line 570 (replace the whole sentence):

```text
**Next program stage (formal Dimension A):** **`ARCH-FREEZE`** — after **`MCP-SERVER-X`** and **`PDC-X`** = **`CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`** (IIAC SCOPED); handoff [`PDC_ARCH_FREEZE_HANDOFF.md`](../qualification/platform_domain_completeness/PDC_ARCH_FREEZE_HANDOFF.md); dossier [`PDC_INDEX.md`](../qualification/platform_domain_completeness/PDC_INDEX.md) (§3.2).
```

Line 572, replace "**`PDC-X`** current; **`ARCH-FREEZE`** blocked until **`PDC-X` = CLOSED** (not entered)." with:

```text
**`PDC-X` = CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED** (IIAC SCOPED); **`ARCH-FREEZE`** not entered.
```

### 2.6 Roadmap section 5 ledger row for PDC-X (currently line 2163)

```text
| PDC-X | BASELINE `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` · dossier [`PDC_INDEX.md`](../qualification/platform_domain_completeness/PDC_INDEX.md) · ticket T-IGX-036-A | **`CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`** (IIAC SCOPED, P0-P9 @ development; decision `<OD id of the Operator's closing signature>`; [final acceptance report](../qualification/platform_domain_completeness/PDC_FINAL_ACCEPTANCE_REPORT.md)); P7 domain audits IA-D1..IA-D8 @ `c08efe46a74a20595468d8d8bf06eab2dca5f573` (D5 FAIL, others PARTIAL) and repairs T-IGX-041-A..I (PRs #568-#576); global audit `<P8 verdict>` @ `<GLOBAL_AUDITED_SHA>`; EBH-7 replay `<replay run id>` (not a B6 verdict); limitations: `PDC_EXCEPTION_REGISTER.md` (`<n>` declared exceptions incl. `dist_scale_x`/Docker C10/C15, F-D1-04, F-D5-02 remainder, F-D7-03 / PDC-D7-R014, four D3 `fixture_strict_adapter_parity` tests), `PDC_DEBT_REGISTER.md`; FRZ `<no promotion | promotions as signed>` (**225** unchanged); **not ARCH-FREEZE**; **not production readiness**; **Next:** `ARCH-FREEZE`. |
```

Related: the ARCH-FREEZE ledger row (line 2164) already reads "after `MCP-SERVER-X` = CLOSED and `PDC-X` = CLOSED"; unchanged.

## 3. Proposed text of the closure note (for the closure pull request and the final acceptance report)

```text
PDC-X (Platform Domain Completeness and Certification) is closed as an independently audited,
scoped stage of the Intergrax platform program (Dimension A only).

What was done: a closed-world inventory and requirement matrix of 172 rows in eight domains
(D1 agent lifecycle, D2 context and knowledge, D3 model runtime, D4 host runtime,
D5 evaluation, D6 runtime security, D7 background and scheduling, D8 artifact lifecycle),
implementation of the evidence-backed gaps (P5), cross-domain integration scenarios XD-01..XD-12 (P6),
independent per-domain audits at c08efe46a (P7), repairs of the real defects they found
(T-IGX-041-A..I), and one independent global audit (P8): <P8 verdict> at <GLOBAL_AUDITED_SHA>.

What the closure does not claim: production readiness or live provider qualification (PROD-Q, Path B);
qualification of dist_scale_x and Docker (C10/C15, deferred); a B6 verdict from the EBH-7 replay
(run <replay run id>); the freeze-SHA audit or VBN acceptance (owned by ARCH-FREEZE).

Declared exceptions (each with a reopen condition) are listed in PDC_EXCEPTION_REGISTER.md; documented
debt in PDC_DEBT_REGISTER.md. FRZ: <no row promoted | rows promoted as signed in PDC_FRZ_PROMOTION_PROPOSAL.md>;
the checklist stays at 225 criteria in 26 families.

Signed by the Operator: <Operator signature and date>. The orchestrator, the implementers and the
auditors did not write this line.
```

## 4. Related edits the closure change must make (checklist for the certifier; not executed here)

| Item | Where | Note |
| --- | --- | --- |
| PDC index statuses | `PDC_INDEX.md` lines 29-44 (`PDC_FINAL_ACCEPTANCE_REPORT`, `PDC_ARCH_FREEZE_HANDOFF` currently "Not started") | set after the drafts become final documents |
| SHA ladder | `PDC_SHA_LADDER.md`: `Dn_AUDITED_SHA`, `AUDIT_EXECUTION_SHA`, `GLOBAL_AUDITED_SHA`, `FINAL_CLOSURE_PUBLICATION_SHA` rows (PDC-P9-02); the ladder has no entries yet for the P7 repair merges #568-#576 (reported by the D5/D6 and D7/D8 extractions) | ancestry verification `<result>` |
| ARCH-FREEZE note | `ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md` header gate note (OD-PDC-ARCHF-01: note-only; task count stays 54) | no new WBS row |
| Checklist | only if option P of `PDC_FRZ_PROMOTION_PROPOSAL.md` is signed | atomic with the roadmap edit (update protocol) |
| Roadmap section 2.0.2 linkage row (line 240) | states "no new `FRZ-PDC-*` IDs; checklist 225 unchanged unless separately authorized" | stays true |

## 5. Fallback text if the global audit does not close the stage

```text
PDC-X — PARTIAL_EXACT_BLOCKERS: <exact blockers from the P8 report, one line each, with finding ids and the
owning ticket>. The stage stays [ ] PLANNED / MANDATORY / CURRENT in the roadmap; ARCH-FREEZE is not entered.
```

Per `DELEGATION-PDC-CLOSE.md` item 2, findings that need a product repair go through one repair loop (plan signed by a delegate, merge by the orchestrator, one more EBH-7 re-pin); a third loop, a finding of a new kind, a hotspot path outside named files, a change of the meaning of a gate or a new authority returns to the Operator.
