# PDC-X — Proposed roadmap closure text (PDC-P9-04) — Variant A chosen in chat, awaiting final confirmation

> **Operator's answers recorded (chosen by the Operator in chat (2026-11-16), awaiting final confirmation).** Transcription: [`operator_decisions/OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (English rendering of the Operator's chat answer of 2026-11-16; unsigned: `Signed: <pending — the Operator confirms the final text in chat>`). Outcome A; all 18 "Operator decision" exceptions declared; the 13 rows open with no exception are documented debt (rows stay `PARTIAL`, no new tickets); FRZ option N; debt register accepted; limitations acknowledged; ARCH-FREEZE handoff published, ARCH-FREEZE not entered.

The Variant A edits of section 3 are applied, as the exact text of section 3, in the **last commit** of the draft closure pull request, which can be dropped if the Operator does not confirm.

> **DRAFT — proposal for the Operator, not a verdict.** Nothing in this file is written into the roadmap, the checklist or any status line. The PDC-X closure line is **not delegated** (`DELEGATION-PDC-CLOSE.md`, "Not delegated"; PDC-P9-03, PDC-P9-04); the Operator signs it once (`PDC_FINAL_ACCEPTANCE_REPORT.md` section 1). Prepared at `origin/development` `f6f335aaef3d1e9ee204bd531f52f8a779566326` (2026-11-16). The closure line is **not written** here: it is shown only inside fenced proposal blocks.

## 0. Facts the text must respect (each read from a file in this repository)

| Fact | Source |
| --- | --- |
| P8 global audit verdict: **`PASS_SCOPED`** at `G` = `969cbd964c679d563b02e55f065b452d46fdf90b`, "only" with the declared exceptions; "not a PASS", "not a repeat-run PASS", "not a verdict on B6 or on certification"; it was `PARTIAL` at `d39a520dd` and at `969cbd964` before the exception record | `PDC_INDEPENDENT_AUDIT.md` sections 13.7, 14.6, 15 |
| Certifier binding manifest verdict `PASS`, scoped, 19 limitations; "not a verdict on B6, not a certification of the whole PDC-X stage and not an FRZ or roadmap action" | `PDC_P8_04_ACTIVATION_CERTIFIER_REPORT.md` |
| IIAC activation `IN_FORCE` from phase-1 commit `9f05749bc23592868f3d0a7f505f9edbb2cfd025` (phase 2 `293bb5b3c`, PR #601 merge `f6f335aae`); any later product-path change voids it (ACT-N11) | `IIAC_POLICY_ACTIVATION_RECORD.json`; certifier report |
| EBH-7 L3 replay at pin `d39a520dd`, run `43738431866`: 2 failed, 720 passed, 9 skipped; **not re-run** by the Operator's condition; **not a B6 PASS**; the EBH-7 identity gate stays a known red class | `DELEGATION-PDC-LOOP2.md`; `DELEGATION-PDC-EXCEPTIONS.md`; `PDC_SHA_LADDER.md` |
| C10/C15 freeze not lifted; `dist_scale_x` and Docker not qualified | `DELEGATION-PDC-WAVE2.md` decision 4; `DELEGATION-PDC-EXCEPTIONS.md` |
| Declared exceptions: 7 cross-cutting (`EX-G-01..07`), 39 per-domain (`EX-D1..D8`), 14 accepted at P8 (`EX-L-01..14`); debt register sections 1-12 | [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md), [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md) |
| FRZ: 225 / 26 unchanged; no row promoted by this package | [`PDC_FRZ_PROMOTION_PROPOSAL.md`](./PDC_FRZ_PROMOTION_PROPOSAL.md) |

## 1. Exit criteria of roadmap section 3.2 against these facts (the Operator decides how to read them)

Roadmap §3.2 (line 1880) says `PDC-X` = CLOSED "only when all hold" and "otherwise the stage reports `PARTIAL_EXACT_BLOCKERS`". Literal reading against the facts:

| Exit criterion (quoted) | Fact at `f6f335aae` | Met literally? |
| --- | --- | --- |
| "D1–D8 each independently accepted (PASS, or PASS_SCOPED only where every excluded requirement is Operator-authorized outside frozen scope)" | The eight P7 domain verdicts are `PARTIAL` (D1-D4, D6-D8) and `FAIL` (D5, one row). The global audit replaced the per-domain re-audits (`DELEGATION-PDC-CLOSE.md` decision 2) and did **not** re-certify rows; the certifier records each domain as "`PASS_SCOPED` inside the global verdict" and does not rewrite the P7 verdicts. Some exclusions are of the "Operator decision" kind and are authorized only if the signature names them (register section 2, 6). | **Not literally.** Met only by the scoped reading, and only for the exclusions the signature names |
| "cross-domain integration PASS" | Audit check 2 was `PARTIAL` at `cba32d247` because of XD-12 (G-06); G-06 is `VERIFIED_FIXED` at `d39a520dd` (audit 13.2). A restated check-2 result at `G` is **UNVERIFIED** (no addendum repeats the check table). | **UNVERIFIED** |
| "tenant/security PASS" | Audit checks 5 and 6 are `PASS_SCOPED` (declared remainders: strict-host `tenant_id=None`, F-D8-04, F-D2-11 tenant part; F-D1-01 HIGH, F-D6-04, F-D6-02). | **Scoped only** |
| "FRZ coverage reconciled (225 / 26 unchanged)" | Yes: checklist unchanged, 225 / 26. | Yes |
| "independent global audit PASS" | `PASS_SCOPED`, not `PASS`. | **No (scoped)** |
| "`MCP-SERVER-X` = CLOSED" | Yes (roadmap line 572; audit section 4). | Yes |
| "all OD-PDC-* decided" | All `OD-PDC-*` rows of `PDC_OPERATOR_DECISIONS.md` read `APPROVED` (read at drafting; the closing signature is a new decision). | Yes |
| "final certification accepted under the authorized protocol" | The certifier's binding manifest is `PASS` scoped; the certifier states it is **not a certification of the whole PDC-X stage**. | **Not as a stage certification** |
| "roadmap closure and ARCH-FREEZE handoff published" | Done only by the closure pull request after the Operator's signature. | Pending |

So the Operator chooses between three signed outcomes (variants in section 2): **A** (scoped closure in the IIAC vocabulary), **B** (the master-plan wording), **C** (do not close: `PARTIAL_EXACT_BLOCKERS`). The roadmap text is the Operator's to approve; this draft cannot supply the interpretation of "PASS" versus `PASS_SCOPED`. `OD-PDC-IIAC-SCOPE.md` (APPROVED Option A) defines `PASS_SCOPED` as the verdict "only for Operator-authorized exclusions" and the MCP-SERVER-X precedent closed on `PASS_SCOPED` with the IIAC headline.

## 2. Which closure vocabulary (open question, with a recommendation)

| Source | Wording |
| --- | --- |
| Master plan, PDC-P9-04 (`PLAN_T-IGX-036-A.md`) | `[x] CLOSED — INDEPENDENTLY ACCEPTED (…SCOPED if applicable) — PUBLISHED` via Protocol 03 B6 / IIAC-B6; "Authorized vocabulary only" |
| IIAC protocol, "Closure vocabulary (canonical)" (`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md:22-29`) | `{STAGE} — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`; "Use `(SCOPED)` and limitations in manifest and pass-record body — **not** in a false global roadmap headline." |
| `OD-PDC-IIAC-SCOPE.md` (APPROVED Option A) | Extends policy `IIAC-2026-11-05.1` to PDC-X; verdict vocabulary PASS, PASS_SCOPED (only for Operator-authorized exclusions), PARTIAL, FAIL |
| Precedent MCP-SERVER-X (roadmap §3 row and §5 ledger) | `CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED` with "(SCOPED, …)" in the body |
| Precedent EVOL-X / EBH-6 / DUP-X | `CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)` / `(AUTOMATION_FIRST SCOPED) — PUBLISHED` (human-B6 era wording) |

**Contradiction:** the master plan says "INDEPENDENTLY ACCEPTED" (B6 wording); the approved process for PDC-X is IIAC, whose canonical title is "INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION)". The verdict on B6 is not delegated and no B6 claim exists. **Recommended: Variant A (IIAC canonical).** Operator chooses: `<Variant A | Variant B | Variant C>`.

## 3. Proposed roadmap text, Variant A (recommended) — placeholders only where a value does not yet exist

Placeholders: `<FINAL_CLOSURE_PUBLICATION_SHA>` (merge of the closure pull request) and `<OD id of the Operator's closing signature>`. Nothing else is a placeholder.

### 3.1 Headline status

```text
PDC-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED
```

with the body qualifier `(IIAC SCOPED)` as for `MCP-SERVER-X` and `ENT-AUDIT-X` (roadmap line 572). Variant B headline: `[x] CLOSED — INDEPENDENTLY ACCEPTED (SCOPED) — PUBLISHED`.

### 3.2 Roadmap section 3 table row (currently line 498, status cell `**[ ] PLANNED / MANDATORY / CURRENT**`)

Replace the status cell with:

```text
**[x] CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED** (IIAC SCOPED; global audit PASS_SCOPED @ 969cbd964c679d563b02e55f065b452d46fdf90b; closure <FINAL_CLOSURE_PUBLICATION_SHA>)
```

and replace the sentence "**No FRZ promotion**; **225** unchanged. **Next:** **`ARCH-FREEZE`**. Product implementation **not started**." (stale: product work was implemented in P5-P7) with:

```text
FRZ coverage reconciled: **225 / 26** unchanged; no FRZ row promoted. Declared exceptions and debt: `PDC_EXCEPTION_REGISTER.md`, `PDC_DEBT_REGISTER.md`. PASS_SCOPED covers the scoped rows only. **Next:** **`ARCH-FREEZE`** (not entered; handoff `PDC_ARCH_FREEZE_HANDOFF.md`).
```

### 3.3 Roadmap section 3.2 "Status" line (currently line 1847) and "Implementation boundary" (line 1884)

```text
**Status:** **`[x] CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`** (IIAC SCOPED). Global audit PASS_SCOPED @ `969cbd964c679d563b02e55f065b452d46fdf90b` (not a PASS); EBH-7 L3 replay run 43738431866 FAILED and was not re-run (not a B6 verdict); closure publication `<FINAL_CLOSURE_PUBLICATION_SHA>`.
```

```text
**Closure boundary:** PDC-X changed product paths in P5-P8 (domain batches B-D1..B-D8, P6 integration repairs, T-IGX-041-A..I, repair loop 1 T-IGX-041-M, repair loop 2 T-IGX-041-P/-Q) and certifies **Dimension A** only; live provider and environment qualification stays with `PROD-Q` (Path B, OD-PDC-PATHB-01). `dist_scale_x` and Docker (C10/C15) are not qualified and the freeze is not lifted. No production-readiness claim; no B6 claim. FRZ: 225 / 26 unchanged, no row promoted.
```

### 3.4 Roadmap "Next program stage" and "Position" lines (currently lines 570 and 572)

Line 570 (replace the whole sentence):

```text
**Next program stage (formal Dimension A):** **`ARCH-FREEZE`** — after **`MCP-SERVER-X`** and **`PDC-X`** = **`CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`** (IIAC SCOPED); handoff [`PDC_ARCH_FREEZE_HANDOFF.md`](../qualification/platform_domain_completeness/PDC_ARCH_FREEZE_HANDOFF.md); dossier [`PDC_INDEX.md`](../qualification/platform_domain_completeness/PDC_INDEX.md) (§3.2).
```

Line 572, replace "**`PDC-X`** current; **`ARCH-FREEZE`** blocked until **`PDC-X` = CLOSED** (not entered)." with:

```text
**`PDC-X` = CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED** (IIAC SCOPED); **`ARCH-FREEZE`** not entered.
```

### 3.5 Roadmap section 5 ledger row for PDC-X (currently line 2163)

```text
| PDC-X | BASELINE `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` · dossier [`PDC_INDEX.md`](../qualification/platform_domain_completeness/PDC_INDEX.md) · ticket T-IGX-036-A | **`CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`** (IIAC SCOPED, P0-P9 @ development; decision `<OD id of the Operator's closing signature>`; [final acceptance report](../qualification/platform_domain_completeness/PDC_FINAL_ACCEPTANCE_REPORT.md)); P7 domain audits IA-D1..IA-D8 @ `c08efe46a74a20595468d8d8bf06eab2dca5f573` (D5 FAIL, others PARTIAL) and repairs T-IGX-041-A..I (PRs #568-#576), P8 repair loops T-IGX-041-M (#584) and T-IGX-041-P/-Q (#594, #593); global audit PASS_SCOPED @ `969cbd964c679d563b02e55f065b452d46fdf90b` (IA-GLOBAL addendum 3, #597); certifier binding manifest PASS (19 limitations), IIAC activation IN_FORCE @ `9f05749bc23592868f3d0a7f505f9edbb2cfd025` (#601); EBH-7 L3 replay run 43738431866 FAILED, not re-run, not a B6 verdict; limitations: `PDC_EXCEPTION_REGISTER.md` (7 cross-cutting, 39 per-domain and 14 P8 declared exceptions incl. `dist_scale_x`/Docker C10/C15, F-D1-04, F-D5-02 remainder, F-D7-03 / PDC-D7-R014, four D3 `fixture_strict_adapter_parity` tests, G-05 residual, E1-E11, `trace_bridge.v1`, G-28), `PDC_DEBT_REGISTER.md`; FRZ no promotion (**225** unchanged); **not ARCH-FREEZE**; **not production readiness**; **Next:** `ARCH-FREEZE`. |
```

Related: the ARCH-FREEZE ledger row (line 2164) already reads "after `MCP-SERVER-X` = CLOSED and `PDC-X` = CLOSED"; unchanged.

## 4. Proposed text of the closure note (for the closure pull request and the final acceptance report)

```text
PDC-X (Platform Domain Completeness and Certification) is closed as an independently audited,
scoped stage of the Intergrax platform program (Dimension A only).

What was done: a closed-world inventory and requirement matrix of 172 rows in eight domains
(D1 agent lifecycle, D2 context and knowledge, D3 model runtime, D4 host runtime,
D5 evaluation, D6 runtime security, D7 background and scheduling, D8 artifact lifecycle),
implementation of the evidence-backed gaps (P5), cross-domain integration scenarios XD-01..XD-12 (P6),
independent per-domain audits at c08efe46a (P7, verdicts PARTIAL and one FAIL), repairs of the real
defects they found (T-IGX-041-A..I), and one independent global audit (P8) with two repair loops:
PASS_SCOPED at 969cbd964c679d563b02e55f065b452d46fdf90b, only with the declared exceptions. PASS_SCOPED
covers the scoped rows only; it is not a PASS.

What the closure does not claim: production readiness or live provider qualification (PROD-Q, Path B);
qualification of dist_scale_x and Docker (the C10/C15 freeze is not lifted); a B6 verdict - the EBH-7 L3
replay (run 43738431866) failed and was not re-run, so B6 is not passed; a certification of every row
of the 172-row matrix (rows were not re-certified one by one); the freeze-SHA audit or VBN acceptance
(owned by ARCH-FREEZE).

Declared exceptions (each with a reopen condition) are listed in PDC_EXCEPTION_REGISTER.md; documented
debt in PDC_DEBT_REGISTER.md. FRZ: no row promoted; the checklist stays at 225 criteria in 26 families.

Signed by the Operator: <Operator signature and date>. The orchestrator, the implementers, the
auditors and the certifier did not write this line.
```

## 5. Related edits the closure change must make (checklist for the certifier; not executed here)

| Item | Where | Note |
| --- | --- | --- |
| PDC index statuses | `PDC_INDEX.md` | updated by the P9 package pull request (statuses "Draft for signature"; not "closed") |
| SHA ladder | `PDC_SHA_LADDER.md` | updated by the P9 package pull request; `FINAL_CLOSURE_PUBLICATION_SHA` stays `—` until the closure merge |
| ARCH-FREEZE note | `ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md` header gate note (OD-PDC-ARCHF-01: note-only; task count stays 54) | no new WBS row; **not edited here** |
| Checklist | only if option P of `PDC_FRZ_PROMOTION_PROPOSAL.md` is signed after a further FRZ-row-level audit | atomic with the roadmap edit (update protocol); **not edited here** |
| Roadmap sections 3, 3.2, 5 and the Position lines (498, 570, 572, 1847, 1884, 2163) | `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` | edited only by the certifier after the Operator's signature; **not edited here** |
| Roadmap section 2.0.2 linkage row (line 240) | states "no new `FRZ-PDC-*` IDs; checklist 225 unchanged unless separately authorized" | stays true |
| Traceability validator | `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` must report `FRZ=225 families=26` on the closure pull request | **not run in this docs-only pass** |

## 6. Variant C — if the Operator does not close the stage

```text
PDC-X — PARTIAL_EXACT_BLOCKERS: exit criteria of roadmap section 3.2 not met literally:
(1) the global audit is PASS_SCOPED, not PASS (969cbd964c679d563b02e55f065b452d46fdf90b);
(2) the eight P7 domain verdicts were not re-graded (PARTIAL x7, FAIL x1) and rows were not re-certified;
(3) exceptions of the Operator-decision kind (PDC_EXCEPTION_REGISTER.md section 2) not named by the signature;
(4) rows open with no exception (PDC_EXCEPTION_REGISTER.md section 6, item 8);
(5) no stage-level certification (the certifier binding manifest is scoped and says so);
(6) the EBH-7 L3 replay failed and was not re-run (B6 not passed).
The stage stays [ ] PLANNED / MANDATORY / CURRENT in the roadmap; ARCH-FREEZE is not entered.
```

A third repair loop returns to the Operator (`DELEGATION-PDC-CLOSE.md` decision 2; `DELEGATION-PDC-LOOP2.md`: "one loop, no further loop").
