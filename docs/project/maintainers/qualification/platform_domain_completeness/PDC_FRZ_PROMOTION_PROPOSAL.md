# PDC-X — FRZ promotion proposal (PDC-P9-03) — option N chosen in chat, awaiting final confirmation

> **Operator's answers recorded (chosen by the Operator in chat (2026-11-16), awaiting final confirmation).** Transcription: [`operator_decisions/OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (English rendering of the Operator's chat answer of 2026-11-16; unsigned: `Signed: <pending — the Operator confirms the final text in chat>`). Outcome A; all 18 "Operator decision" exceptions declared; the 13 rows open with no exception are documented debt (rows stay `PARTIAL`, no new tickets); FRZ option N; debt register accepted; limitations acknowledged; ARCH-FREEZE handoff published, ARCH-FREEZE not entered.

**Effect:** option N: no FRZ row is promoted; the checklist is untouched; counters stay **225 / 26**. Sections 5-7 stay as information only.

> **DRAFT — proposal for the Operator, not a verdict.** No FRZ status is changed by this file. `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` is untouched. FRZ promotion is **not delegated** (`DELEGATION-PDC-CLOSE.md`, "Not delegated"; `DELEGATION-PDC-WAVE2.md`); it needs the Operator's signature and the certifier role, and any promotion would rest on the P8 result: global audit **`PASS_SCOPED`** at `G` = `969cbd964c679d563b02e55f065b452d46fdf90b` (`PDC_INDEPENDENT_AUDIT.md` section 15), which is **not row-level FRZ evidence** (section 3, item 3). **I do not change any FRZ status, the checklist or the roadmap.** Proposed default: **Option N (no promotion)**.

## 1. Rules applied

| Rule | Source |
| --- | --- |
| Existing FRZ identifiers only; no `FRZ-PDC-*` row; checklist cardinality **225 criteria in 26 families** must not change. | `OD-PDC-FRZ-01.md` (APPROVED Option A, 2026-11-12); roadmap §3.2 |
| Promotion only for rows whose evidence PDC produced, through the checklist update protocol, with no cardinality change. | master plan `PLAN_T-IGX-036-A.md` PDC-P9-03 |
| `FRZ-* = PASS` only when **all four** exist: exact GitHub SHA, code or contract evidence at that SHA, qualification/gate/test evidence, **independent audit evidence**; an implementation report alone is not sufficient. | checklist, "PASS evidence rule" |
| Allowed final statuses: `OPEN`, `PASS`, `BLOCKED`, `N/A — WITH EVIDENCE`. (The checklist also carries historical `PASS — SCOPED` variants for EVOL-X/ENT-AUDIT-X rows; this proposal does not create new ones.) | checklist, "Allowed statuses" |
| Update protocol after independent closure: atomically update (1) roadmap stage/status and section 5 ledger and (2) checklist Status and Evidence for every FRZ row advanced. | checklist, "Update protocol" |
| Path B: live-scoped criteria (`FRZ-PRD-01`, `FRZ-PRD-02`, live portions of `FRZ-PRD-03`, `FRZ-PRD-08`, `FRZ-SEC-04`) stay `OPEN` until the `PROD-Q` live gate; historical or scoped evidence does not auto-promote. | checklist, "Path B"; `OD-PDC-PATHB-01.md` |
| `FRZ-TEN-*`: "historical stage closure does not auto-PASS tenant isolation". | checklist, "Initial checklist policy" |
| Precedent: MCP-SERVER-X promoted **no** FRZ row because its P8 audit cited no FRZ id; notes would need a row-level argument from the independent auditor naming the FRZ id. | `mcp_server_x/MCP_SERVER_X_P8_FRZ_EVIDENCE_NOTES.md` |

## 2. Baseline numbers (read from the checklist and the matrix; the checklist, the roadmap and the matrix JSON are **unchanged** between `cba32d247` and `f6f335aae` (`git diff` empty))

| Item | Value |
| --- | --- |
| FRZ criteria / families in the checklist | **225 / 26** (parsed: 225 distinct `FRZ-*` ids, 26 families) |
| Statuses today | 108 `PASS`, 109 `OPEN`, 6 `PASS — SCOPED`, 1 `PASS (EVOL-X SCOPED)`, 1 `PASS — COMPENSATING (SCOPED)` (matches the MCP-SERVER-X P8 validator line `PASS=108 OPEN=109`) |
| PDC requirement rows | 172 (D1 22, D2 18, D3 24, D4 20, D5 18, D6 28, D7 23, D8 19), all `NOT_CERTIFIED` in the requirements matrix; the P8 certifier matrix (`PDC_DOMAIN_CERTIFICATION_MATRIX.md`) records **domain** dispositions only, no row re-grade |
| FRZ ids referenced by PDC rows | 111: 50 already `PASS`, 1 `PASS — SCOPED`, 1 `PASS (EVOL-X SCOPED)`, 59 `OPEN` (`PDC_FRZ_COVERAGE_MATRIX.md`; 114 ids are **not** referenced by any PDC row and stay with their stages or with ARCH-FREEZE) |
| Counter check after any change | `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` must report `FRZ=225 families=26` (command taken from the MCP-SERVER-X P8 record; **not run in this docs-only pass**: the certifier runs it at the closure SHA, `<validator result at closure>`; the global audit reports `pdc_x_requirements.py --check` OK for 172 rows and the checklist byte-identical to the P0 baseline, audit check 9) |

## 3. Method

1. For every `OPEN` FRZ row referenced by a PDC row, take each referencing PDC row and classify it from the P7 audits, the signed repair plans and the exception register: `PASS_AT_AUDIT` (the P7 audit rated PASS), `REPAIRED` (audit PARTIAL, finding fully in scope of a merged repair; **needs re-audit**), `REPAIRED_LIMIT` (repaired with a stated limitation), `EXCEPTION` (rests on an exception of the register), `OPEN_NO_EXC` (PARTIAL with no repair and no exception), `SCOPED_PRE` (already `PASS_SCOPED` by an earlier authorization).
2. An `OPEN` FRZ row is a **candidate** only if every referencing PDC row is `PASS_AT_AUDIT` or `REPAIRED` (and none is `SCOPED_PRE`), and the row is not Path B live-scoped.
3. Candidates are **conditional, and the condition is not met by the P8 result**. The P7 audits rate PDC rows, not FRZ rows. The global audit's check 9 ("FRZ mapping") is `PASS` for the mapping only (225 / 26, no status moved); its check 15 (requirement coverage) is `PASS_SCOPED` and states that **rows were not re-certified one by one**; the certifier matrix (section 5) states "not a row-by-row recertification of the 172 requirement rows". The checklist's PASS evidence rule needs **independent audit evidence** for the FRZ row, so no candidate has it today. The verdict is `PASS_SCOPED`, not `PASS`, and it holds only with the accepted exceptions of `PDC_EXCEPTION_REGISTER.md`.
4. A PDC requirement row is evidence for part of a FRZ criterion; several criteria are owned by other stages (column "Primary closing stage(s)"). The Operator should decide whether PDC evidence suffices for criteria that list `PROD-Q`, `CTRL-X`, `QUAL-X` or `EBH-*` as closing stages. This draft cannot decide that.
5. Row classes come from the digests of the per-domain extraction taken **before** repair loops 1 and 2 (non-PASS rows are listed explicitly there; all other rows are treated as P7 audit PASS). They were **not re-derived** after T-IGX-041-J..Q and the accepted exceptions EX-L-01..14; class assignments are therefore **inferred** wherever a plan states no target rating (marked `*` for `REPAIRED`).

## 4. Options for the Operator (one signature)

| Option | Content | Consequence |
| --- | --- | --- |
| **N — no promotion** (precedent MCP-SERVER-X) | Attach PDC evidence notes only; statuses stay as they are; counters 225/26. | Safest; leaves the 19 candidates `OPEN` for ARCH-FREEZE to promote through their own stages. |
| **P — conditional promotion** | Possible only **after a further independent audit session names the FRZ ids with PASS evidence at `G`** (none exists at `f6f335aae`); then promote only the section 5 candidates it names; update roadmap ledger and checklist atomically; run the traceability validator. | Section 6 and 7 rows are **not** promoted in either option. Without that audit, signing P promotes nothing. |

Proposed default if the Operator signs without choosing: **N**. This is a proposal only; the single signature block is in `PDC_FINAL_ACCEPTANCE_REPORT.md` section 1.

## 5. Candidates for promotion (conditional; evidence missing, see section 3 item 3) — 19 rows

`*` = the PDC row was `PARTIAL` at the P7 audit and is repaired by the merged ticket per the plan; the repair was not re-audited. Evidence pointers for every row below: the P7 audit report of the named domain (`PDC_P7_D<n>_AUDIT_REPORT.md` at `c08efe46a74a20595468d8d8bf06eab2dca5f573`), the matrix row, the repair PR of the domain (D1 #574, D2 #575, D3 #569, D4 #571, D5 #568, D6 #573, D7 #570, D8 #572; section 5 of the handoff draft), and `PDC_INDEPENDENT_AUDIT.md` section 15 at `G` = `969cbd964c679d563b02e55f065b452d46fdf90b` (no FRZ-row evidence; see section 3).

| FRZ | Criterion | Primary closing stage(s) in the checklist | Referencing PDC rows (all `PASS_AT_AUDIT` unless marked `*`) |
| --- | --- | --- | --- |
| FRZ-CTR-04 | public/internal contracts classified | EBH-2*, EBH-3, EBH-4 | D4-R014* |
| FRZ-CTR-05 | contract responsibility narrow/cohesive | EBH-2*, EBH-3, EBH-4 | D1-R005 |
| FRZ-DOC-02 | no conflicting semantic authority | EBH-6, ARCH-FREEZE | D4-R019, D5-R016, D6-R021, D6-R025 |
| FRZ-DOC-05 | public contracts manifest current | EBH-6, ARCH-FREEZE | D4-R014* |
| FRZ-OWN-02 | one canonical contract per responsibility | EBH-2*, EBH-3 | D2-R017, D4-R001, D8-R017, D8-R018 |
| FRZ-OWN-04 | no shadow authority | EBH-2*, EBH-3 | D1-R004, D6-R003 |
| FRZ-PLG-03 | provider discovery has explicit owner | EBH-2*, EBH-5 | D1-R020 |
| FRZ-PLG-04 | provider selection has explicit owner | EBH-2*, EBH-5 | D3-R003 |
| FRZ-REG-02 | every corrected blocker has regression gate | QUAL-X, **EBH-5** (RPL-G vendor-coupling gates) | D1-R022, D4-R020, D6-R028 |
| FRZ-REG-03 | negative tests prove gates detect violations | QUAL-X | D1-R004, D2-R007 |
| FRZ-REG-05 | stale inventories = 0 | QUAL-X | D1-R021, D3-R024 |
| FRZ-REL-03 | idempotency ownership explicit | CTRL-X, PROD-Q | D7-R010*, D7-R011 |
| FRZ-REL-05 | external operation termination semantics explicit | CTRL-X, PROD-Q | D3-R011* |
| FRZ-REL-08 | bounded concurrency and work-admission semantics explicit | HARNESS-W4, CTRL-X, PROD-Q | D7-R021* |
| FRZ-REL-11 | provider throttling/rate-limit handling cannot create unbounded retry/fallback or bypass | HARNESS-W4, CTRL-X, PROD-Q | D3-R004 |
| FRZ-RPL-01 | representative default provider replaceable | **EBH-5** (primary), **EBH-7** (convergence) | D3-R016, D3-R024 |
| FRZ-SEC-07 | security claim bounded by explicit threat model/evidence | CTRL-X, PROD-Q | D4-R019, D5-R006, D6-R021 |
| FRZ-SEC-10 | platform vs infrastructure responsibility for encryption/data residency/provider exposure is explicit and evidence-backed | CTRL-X, PROD-Q | D6-R023 |
| FRZ-TYP-05 | string-dispatch substitute for typed contract = 0 | EBH-2* | D3-R009*, D5-R007 |

Cautions on the candidates: (a) rows whose closing stage lists `PROD-Q` (FRZ-REL-03, -05, -08, -11, FRZ-SEC-07, -10) are Path B territory in part: promote only if the Operator accepts that the architecture-scoped part is what PDC evidences; (b) FRZ-DOC-02 and FRZ-DOC-05 also list ARCH-FREEZE as closing stage; (c) FRZ-REG-02, -03, -05 and FRZ-CTR-05 are broad criteria ("every corrected blocker has regression gate", "stale inventories = 0") that two or three PDC rows only partly evidence; (d) D4-R014 (`FRZ-CTR-04`, `FRZ-DOC-05`), D7-R010/R021 and D3-R011/R009 are `REPAIRED*` rows that depend on a re-audit.

**Effect of the accepted P8 exceptions on the candidates (inferred by domain, not row-level; no source names the rows).** The exceptions of `PDC_EXCEPTION_REGISTER.md` section 3 that bear on the domains of the referencing rows: D1 (EX-D1-07 durable CAS, EX-L-09 G-12): FRZ-CTR-05, -OWN-04, -PLG-03, -REG-02, -REG-03, -REG-05; D2 (EX-L-04, -05, -12, -14): FRZ-OWN-02, -REG-03; D3 (EX-L-01 scenario reds, EX-L-07 F-D3-02 pins, EX-L-08 stream resilience claim): FRZ-PLG-04, -REG-05, **-REL-05 (stream termination semantics) and -REL-11 (retry/rate-limit on streams) in particular**, -RPL-01, -TYP-05; D5 (EX-L-09 strict host, EX-L-13): FRZ-DOC-02, -SEC-07, -TYP-05; D7 (EX-L-03 queue recovery not CAS, EX-L-06, EX-L-10): FRZ-REL-03, -REL-08; D8 (EX-L-02): FRZ-OWN-02. FRZ-CTR-04, -DOC-05 (D4) and FRZ-SEC-10, parts of -DOC-02 (D6) have no new accepted exception in their domain. If Option P is ever taken, every candidate must be re-derived against that list first.

## 6. Not promoted because of an exception, an open row or a limitation — 34 rows blocked by exceptions, 6 other

Rows blocked by at least one PDC row that rests on an exception of [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md) or is open without exception (the register id is in brackets):

| FRZ | Criterion | Blocking PDC rows |
| --- | --- | --- |
| FRZ-BND-04 | private implementation leakage = 0 | D4-R015 (EX-D4-05) |
| FRZ-BND-05 | cross-layer construction only sanctioned composition | D4-R002 (EX-D4-03) |
| FRZ-CTR-01 | consumers depend on contracts | D2-R004 (open, no exception); D2-R010 (repaired with a limitation); D3-R001 (EX-D3-04) |
| FRZ-CTR-06 | contracts do not encode concrete vendor/runtime implementation | D3-R001 (EX-D3-04); D3-R019 (open, no exception); D4-R015 (EX-D4-05) |
| FRZ-DEBT-04 | PARTIAL inside frozen scope = 0 | D2-R005 (EX-D2-03) |
| FRZ-DEBT-05 | DEFERRED inside frozen scope = 0 | D1-R018 (EX-D1-06); D1-R019 (pre-existing PASS_SCOPED); D3-R020 (pre-existing PASS_SCOPED); D4-R011 (pre-existing PASS_SCOPED); D4-R012 (pre-existing PASS_SCOPED); D4-R013 (pre-existing PASS_SCOPED); D7-R006 (EX-D7-02); D7-R008 (EX-D7-02); D8-R010 (EX-D8-02) |
| FRZ-DOC-01 | canonical docs match code | D1-R009 (open, no exception) |
| FRZ-DOC-03 | ownership docs current | D7-R022 (open, no exception) |
| FRZ-OBS-01 | canonical event/evidence spine | D1-R017 (open, no exception); D5-R013 (EX-D5-02); D6-R020 (EX-D6-01/02) |
| FRZ-OBS-05 | terminal state observable | D1-R014 (EX-D1-05) |
| FRZ-OBS-06 | side-effect paths observable | D8-R016 (EX-D8-01) |
| FRZ-OWN-01 | every major concern has exactly one semantic owner | D7-R022 (open, no exception) |
| FRZ-OWN-03 | one sanctioned composition owner | D4-R002 (EX-D4-03) |
| FRZ-OWN-05 | no duplicated semantic mechanism | D5-R005 (EX-D5-01) |
| FRZ-PLG-01 | extensible mechanisms expose platform contracts | D4-R013 (pre-existing PASS_SCOPED); D5-R005 (EX-D5-01) |
| FRZ-PLG-05 | activation/admission explicit | D1-R003 (EX-D1-01); D6-R016 (EX-D6-02/03) |
| FRZ-PLG-08 | dynamic registration reversible/scoped when supported | D1-R009 (open, no exception); D1-R014 (EX-D1-05); D1-R018 (EX-D1-06) |
| FRZ-REG-06 | deterministic qualification tests | D5-R002 (open, no exception) |
| FRZ-REG-09 | critical invariant protected by code/test, not docs only | D5-R014 (EX-D5-03/04); D6-R007 (open, no exception) |
| FRZ-REL-01 | timeout ownership explicit | D3-R012 (EX-D3-01) |
| FRZ-REL-02 | retry ownership explicit | D7-R009 (EX-D7-02) |
| FRZ-REL-04 | cancellation propagates correctly | D4-R005 (EX-D4-01); D7-R014 (EX-D7-01) |
| FRZ-REL-06 | provider failure typed | D3-R013 (EX-D3-03) |
| FRZ-REL-07 | no hidden infinite retry/fallback | D7-R009 (EX-D7-02); D7-R013 (EX-D7-02) |
| FRZ-REL-09 | queue/backpressure and overload behavior explicit | D7-R012 (pre-existing PASS_SCOPED); D7-R013 (EX-D7-02) |
| FRZ-REL-10 | resource saturation/exhaustion has deterministic reject/degrade/fail-c | D2-R010 (repaired with a limitation); D8-R012 (EX-D8-02) |
| FRZ-RPL-04 | vendor/backend switch does not modify consumers | D1-R015 (open, no exception); D8-R015 (pre-existing PASS_SCOPED) |
| FRZ-SEC-02 | secrets references / late resolution | D6-R009 (EX-D6-02) |
| FRZ-SEC-03 | raw secrets do not cross unrelated layers | D6-R007 (open, no exception) |
| FRZ-SEC-05 | sandbox/isolation boundaries qualified | D6-R010 (open, no exception); D6-R012 (pre-existing PASS_SCOPED) |
| FRZ-SEC-06 | provider/plugin trust qualified | D1-R003 (EX-D1-01); D1-R013 (EX-D1-02/03/04); D6-R016 (EX-D6-02/03) |
| FRZ-SEC-08 | sensitive/tenant data propagation across platform/provider boundaries  | D2-R008 (pre-existing PASS_SCOPED); D2-R014 (EX-D2-02); D6-R015 (open, no exception); D8-R006 (EX-D8-02/03/04); D8-R010 (EX-D8-02) |
| FRZ-SEC-09 | retention / deletion / redaction responsibility is explicit for platfo | D2-R015 (EX-D2-03); D6-R022 (EX-D8-01); D8-R008 (EX-D8-01); D8-R009 (EX-D8-01) |
| FRZ-TYP-01 | semantic boundaries strongly typed | D2-R004 (open, no exception) |

Other `OPEN` referenced rows not promoted:

| Reason | FRZ | Criterion | Primary closing stage(s) | Referencing rows / blockers |
| --- | --- | --- | --- | --- |
| LIVE | FRZ-PRD-01 | production qualification independent of lab/harness | PROD-Q | D3-R021 (pre-existing PASS_SCOPED) |
| LIVE | FRZ-SEC-04 | production auth bypass = 0 | CTRL-X, PROD-Q | D6-R002 (pre-existing PASS_SCOPED); D6-R017 (EX-D6-04); D6-R026 (pre-existing PASS_SCOPED) |
| BLOCKED_PRE | FRZ-DEBT-06 | every N/A/outside-scope item evidence-backed | ARCH-FREEZE | D1-R019 (pre-existing PASS_SCOPED); D4-R011 (pre-existing PASS_SCOPED) |
| BLOCKED_PRE | FRZ-PLG-02 | external structural implementation works without core patch | EBH-2*, EBH-5 | D3-R020 (pre-existing PASS_SCOPED) |
| BLOCKED_PRE | FRZ-RPL-02 | custom implementation proof exists | **EBH-5** (primary), **EBH-7** | D3-R017 (pre-existing PASS_SCOPED) |
| CONDITIONAL_LIMIT | FRZ-SEC-01 | tenant isolation proven | CTRL-X, PROD-Q | D2-R002 (repaired with a limitation); D6-R005 (repaired with a limitation) |

Reason legend: `LIVE` = Path B live-scoped, stays `OPEN` until the `PROD-Q` live gate; `BLOCKED_PRE` = a referencing row is `PASS_SCOPED` by an earlier Operator authorization (OD-PDC-SCOPE-02/-03, OD-PDC-PATHB-01), so the criterion cannot be shown as a clean `PASS`; `CONDITIONAL_LIMIT` = tenant isolation rows repaired with a stated limitation and `FRZ-TEN-*`/`FRZ-SEC-01` policy forbids auto-promotion from stage evidence.

Note on `FRZ-DEBT-06` ("every N/A/outside-scope item evidence-backed"): the exception register and `OD-PDC-SCOPE-02` ("Recorded as FRZ-DEBT-06 evidence-backed exclusions") are the evidence PDC offers, but the row is owned by ARCH-FREEZE and the exceptions of the register are not signed until the Operator signs; it is therefore not proposed.

Blocking references above that read "(open, no exception)" were taken from the pre-P8 drafts and are unchanged by P8 (see register section 6, item 8). New blockers from EX-L-01..14 are listed in section 5 and are **not** merged into this table.

## 7. Rows already `PASS` / scoped that PDC references (no promotion needed)

50 referenced rows are already `PASS` (for example `FRZ-AUD-05`, `FRZ-CFG-01..03`, `FRZ-EXE-01/02/06/07`, `FRZ-GOV-01..06/08`, `FRZ-REC-01..04/10`, `FRZ-STA-*`, `FRZ-TEN-01/04..08/11/12`, `FRZ-TRC-*`), plus `FRZ-EVOL-06` (`PASS — SCOPED`) and `FRZ-EVOL-07` (`PASS (EVOL-X SCOPED)`). PDC evidence is incremental there: it may be listed in the evidence column as a note, but no status changes. Exceptions that could weaken such a row at P8 (for example EX-D7-01 against `FRZ-REL-04`, which is `OPEN` and blocked above) are recorded in the exception register, not here.

## 8. Proposed protocol for the single signature (not executed)

1. Operator selects option N or P (section 4) and signs in `PDC_FINAL_ACCEPTANCE_REPORT.md` section 1 (decision 5): `<Operator signature — not written by the orchestrator, an auditor or a delegate>`.
2. Certifier (not the implementer, not the auditor) prepares one atomic change: roadmap section 3 row, section 5 ledger row and Position line; checklist `Status` and `Evidence` cells of exactly the signed FRZ rows (option P only); the evidence cell cites the P7 report, the repair PR, the FRZ-row-level audit named in section 4 and, for scoped rows, the exception id. (Option N: evidence notes only, no checklist edit.)
3. Validate: `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` reports `FRZ=225 families=26`, `dangling_open_frz=0` (`<result>`); `uv run python scripts/qualification/pdc_x_requirements.py --check` (`<result>`).
4. Nothing in the checklist is rewritten for rows that stay `OPEN`; `ARCH-FREEZE` keeps owning every row listed in section 6.

## 9. Open questions for the Operator

1. Option N or P (section 4)?
2. Is PDC evidence enough for criteria whose closing stage is `PROD-Q`, `CTRL-X`, `QUAL-X` or `EBH-*` (cautions a-c)?
3. Row classes marked `*` assume the merged repairs hold; P8 re-verified repaired HIGH and MEDIUM findings (audit 7, 13) but did not re-grade rows, and LOW by-products and test-only repairs were only partly in its sampling. If Option P is ever taken, drop or re-audit FRZ-CTR-04, -DOC-05, -REL-03, -REL-05, -REL-08, -TYP-05.
4. The coverage matrix lists checklist statuses at generation time; the checklist is byte-identical at `cba32d247` and `f6f335aae` (empty `git diff`). If it changes before signing, recompute.
