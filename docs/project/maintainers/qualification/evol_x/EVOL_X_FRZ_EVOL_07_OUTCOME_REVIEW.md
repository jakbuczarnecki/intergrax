# EVOL-X - FRZ-EVOL-07 Outcome A/B review record

Status: REVIEW RECORD (read-only analysis). It authorizes nothing, signs nothing, promotes no checklist row and contains no
`Approved ...` line.

## 1. Reviewer session independence statement

- Reviewer: an AI agent session (Claude Code, model claude-sonnet-5-5), started fresh for this question only.
- It had no role in EVOL-X implementation, test execution, audit, certification, publication or signing, and it shares no
  session id with any of those roles. It started from the repository state only (no prior conversation context).
- Assurance level: `process_enforced` (separation by session/process discipline, not by cryptographic or organizational
  identity proof). This record is an advisory technical review for the operator; it is not an organizational G6 audit and not
  a named-human auditor act.
- Worktree: detached, outside the repository, at `origin/development` = `b288ba756ef736f3d372832d8e684dbb19901ca4`
  (`docs(evol-x): record B6_ACCEPTANCE_PUBLICATION_SHA (phase 2, non-self-referential)`). No pytest or other test run was
  executed; all findings come from reading code, data files and evidence.

## 2. Question

Is FRZ-EVOL-07 Outcome A or Outcome B, as defined in `evol_x/b6_technical_qualification/EVOL_X_B6_FRZ_EVOL_SCOPED_READINESS.md`
and `evol_x/EVOL_X_OD_EVOL_12_DECISION_PACKAGE.md`?

- Outcome A: the canonical checklist wording of FRZ-EVOL-07 may be promoted to `PASS (EVOL-X scoped)` with the mandatory
  footnote `Qualification-layer deploy/rollback safety; not live PROD-Q production enforcement` (OD-EVOL-05b Option 1),
  without overstating what was qualified.
- Outcome B: the wording cannot honestly be promoted; FRZ-EVOL-07 stays OPEN.

Secondary: does anything obviously block the OD-EVOL-12 Option A text for the other FRZ-EVOL rows (report only, no deep audit)?

## 3. Evidence checked

All paths are relative to `D:/Projekty/_wt/intergrax/frz07-rev` (HEAD `b288ba756ef7`).

Governance and wording:
- `docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` lines 47, 273-282
  (FRZ-EVOL-07 row: requirement `deployment / rollback / roll-forward qualified or unsafe refusal explicit`; owner
  `EVOL-X, PROD-Q (supporting)`; status OPEN; note `Rollback not "safe" from binary restart alone - durable state + contract
  compatibility required (S7)`).
- `evol_x/b6_technical_qualification/EVOL_X_B6_FRZ_EVOL_SCOPED_READINESS.md` (Outcome A/B definitions; agent constraint).
- `evol_x/EVOL_X_OD_EVOL_12_DECISION_PACKAGE.md` (Option A text; Outcome B alternative; status PENDING).
- `evol_x/EVOL_X_B6_RECORD_AND_FRZ_ROW_FORMATS.md` (six-cell row, status/notes rendered from the certified disposition).
- `evol_x/EVOL_X_ENFORCEMENT_BOUNDARY_W1.md` rows "Unsafe rollback (durable state)" and "FRZ-EVOL-07 scoped acceptance".
- `evol_x/EVOL_X_OPERATOR_DECISION_REGISTER_W1.md` (OD-EVOL-05b: SIGNED / APPROVED 2026-11-01, Option 1, no W4-04 port;
  OD-EVOL-12 still PENDING).
- `evol_x/b6_technical_qualification/A4_FRZ_GOVERNANCE_READINESS.md`.

Preflight (B-06 / B-11 FRZ blocks) - `scripts/qualification/evol_x_closure_publication_preflight.py`:
- `FRZ_DISPOSITION_RESTRICTIONS["FRZ-EVOL-07"] = ("LIM-05", "NOT claimed")` (certification disposition must keep both).
- `FRZ_ROW_RESTRICTIONS["FRZ-EVOL-07"] = ("qualification-layer", "not claimed")` (promoted row text, case-insensitive).
- `_gate_b_frz` (lines ~2452-2500): exactly one row, requirement equals canonical literal, status starts with PASS, SCOPED
  disposition cannot become unrestricted PASS, every `LIM-nn` of the disposition must appear in the row, restriction phrases
  must appear, and `(status, notes)` must equal `render_promoted_row(<certified disposition>)` with exactly six cells (B-11).
- `scripts/qualification/evol_x_frz_evol_canonical_matrix.py` (canonical requirement literals match the checklist).

Evidence:
- `evol_x/evol_x_p03_independent_replay/evol_frz_evidence.v1.json`: row FRZ-EVOL-07 `passed: true`,
  `source_commit_sha = 3c4a641b3e48a13f437e830a0562e2a80b03aabb` (TES), behavioral test
  `tests/qualification/evol_x/frz/test_evol_x_frz_07_deploy_rollback.py::test_frz_evol_07_s7_arms_pass`,
  `scope_footnote`: "qualification-layer assessor (DM-01..07) only; production ActivationService.rollback durable-state
  enforcement is NOT claimed (OD-EVOL-05b Option 1)". The file is byte-identical at FINAL `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1`
  and at HEAD. TES is an ancestor of HEAD.
- Post-TES drift check (`git diff 3c4a641b3... HEAD`): no change under `intergrax/agent_distribution`, `testing_support`,
  `tests/qualification/evol_x/{data,frz,experiments}`. The only post-TES product-tree changes are
  `intergrax/platform/enterprise_audit/*` (ENT-AUDIT-X), `intergrax/applications/_shared/health_score_wiring.py` and
  `intergrax/platform/__init__.py`; none is on the S7 path.
- Code actually read: `testing_support/change_safety/assess/deployment_safety.py` (DM-01..07 assessor, sidecar snapshot port),
  `tests/qualification/evol_x/data/deployment_safety_matrix.v1.json`, `.../deployment_na_register.v1.json` (NA-01..04 OUT_OF_SCOPE),
  `tests/qualification/evol_x/experiments/arms/s7.py`, `tests/qualification/evol_x/adapters/ad_harness.py`,
  `tests/qualification/evol_x/frz/test_evol_x_frz_07_deploy_rollback.py`, `tests/qualification/evol_x/_constants.py`
  (`FRZ_EVOL_07_SCOPE_FOOTNOTE`), `testing_support/change_safety/workflow.py` (lines 165-177),
  `intergrax/agent_distribution/activation.py` (`ActivationService.rollback`, lines 423-530).
- Independent audit / certification (read for consistency, not relied on as proof):
  `iiac_evidence/EVOL-X/audit/frz_evol_acceptance_matrix.md`, `.../audit/IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md` (section 8
  W7-03 determination; F-C-06; LIM-05), `.../certification/EVOL_X_IIAC_CERTIFICATION_RECORD.json` (`frz_evol_dispositions`),
  `.../certification/EVOL_X_IIAC_CERTIFICATION_ASSESSMENT.md` row 07,
  `.../b6_acceptance/EVOL_X_IIAC_B6_ACCEPTANCE_RECORD.md` (operator-signed 2026-11-07; audited implementation SHA = TES).

## 4. Findings

### 4.1 What the evidence really shows (verified against code, not prose)

F1. The assessor is real and total. `assess_deploy_safety` evaluates `DeployFacts` (operation rollback/roll_forward,
evidence_present, contract_compat, persisted_schema_compatible, durable_forward_only_applied, inflight_pins, drain_possible)
against an ordered first-match matrix that ends in a fail-closed catch-all. Missing evidence, UNKNOWN contract compatibility,
forward-only durable state on rollback, incompatible persisted schema, incompatible contract, and in-flight pins without drain
all return typed `REFUSE` reasons. Unknown operations fall to the catch-all `REFUSE/INSUFFICIENT_EVIDENCE`. This directly
matches the checklist text "qualified or unsafe refusal explicit" and the note "durable state + contract compatibility
required". Roll-forward is covered (DM-07). The loader fails if DM-01..07 are not all present.

F2. S7 has a positive and a negative arm. Positive: oracle rows match matrix outcomes AND the real in-memory
`ActivationService` rollback succeeds on the clean path. Negative: a forward-only sidecar snapshot pair derives
`durable_forward_only_applied=True` and the assessor returns DM-04 REFUSE with `false_safe=0`.

F3. The limitation is real and is stated in the evidence itself. `ActivationService.rollback` (activation.py 423-530) checks
serving pointer, expected-revision conflicts, rollback eligibility of the prior revision, artifact revalidation hook (default:
digest present) and the atomic commit. It contains no durable-schema, forward-only-migration or in-flight-pin check, and
nothing under `intergrax/` references `change_safety` or `assess_deploy_safety`. The S7 negative arm records
`ad_layer_alone_succeeds=True`: the production rollback accepts exactly the rollback the assessor refuses (audit finding F-C-06,
LIM-05; NA-01 OUT_OF_SCOPE). So the claim cannot be "unsafe rollback is blocked"; it can only be "unsafe rollback is
explicitly assessed and refused by the qualification-layer assessor".

F4. Honest scope boundaries of the qualification-layer claim (they strengthen the need for the footnote, they do not defeat
it):
- The assessor is a pure function over declared facts. In S7 the facts come from a hand-authored sidecar snapshot manifest
  in the oracle (OD-EVOL-06), not from a live database, real migrations or a real deployment.
- In `testing_support/change_safety/workflow.py`, deploy facts are optional (`deploy_facts: DeployFacts | None`); a change
  workflow without them simply has no deploy assessment. The refusal is explicit only when facts are supplied.
- The positive arm uses in-memory fakes (`FakeInMemoryRuntimeDeploymentAdapter`, in-memory stores); this is contract-level
  evidence, not vendor or PROD-Q evidence.
- NA-02 (live PROD-Q slice), NA-03 (DIST-SCALE-X provenance) and NA-04 (binary restart / multi-instance rolling behaviour)
  are registered OUT_OF_SCOPE.

F5. The Outcome A/B question is already pre-framed by a signed operator decision. OD-EVOL-05b is SIGNED/APPROVED
(2026-11-01) with Option 1 (qualification-only scope, no W4-04 production precondition port). The checklist wording asks for
"qualified or unsafe refusal explicit", which the assessor satisfies at the qualification layer; the enforcement boundary
document anticipated exactly this "scoped acceptance" reading. The checklist row is not rewritten by this reading: the
requirement literal stays unchanged and only status/notes change.

F6. Evidence integrity. The evidence row is bound to TES; no S7-relevant code or data changed after TES; the evidence file is
unchanged since FINAL; the operator B6 acceptance record (audited implementation SHA = TES, `PASS_SCOPED`) is present at HEAD.
I did not re-run tests (not permitted), so test status rests on the recorded evidence and the independent auditor's recorded
replay (19/19 for FRZ-EVOL-07 at TES and HEAD), which I did not independently reproduce.

### 4.2 Reasons for Outcome B that I tried to establish

R1. "Production rollback is not gated, so FRZ-EVOL-07 cannot be PASS." Rejected as a blocker: the canonical text does not say
production enforcement; the owner column lists PROD-Q only as "supporting"; the note requires durable state + contract
compatibility to be considered, which the assessor does; and the limitation is machine-enforced into the promoted row (see 4.3).
It would be decisive only if the row were promoted without scope text, which the preflight forbids.

R2. "The operator-drafted footnote understates." Partly valid (see 4.3): "not live PROD-Q production enforcement" can be read as
"PROD-Q has not yet qualified an existing production enforcement". In fact no production enforcement of durable-state rollback
safety exists, and the production path demonstrably accepts the unsafe rollback. The short footnote is therefore too weak as
the only scope text. This is a wording defect fixable inside Outcome A, not a reason for B, because the certified disposition
and the preflight already require the stronger wording.

R3. "Hand-authored facts are not deployment evidence." Valid as a limitation, but the claim is explicitly an assessor
(decision-logic) claim; it does not claim fact acquisition from live systems. Acceptable if the footnote says
"qualification-layer" and names the assessor scope. No decisive B argument.

R4. "A reader may take PASS as platform rollback safety." Mitigated by status `PASS (EVOL-X SCOPED)`, the LIM-05 id and the
mandatory text in the same row, all machine-checked by B-06/B-11.

R5. Procedural: the OD package is still PENDING, the operator register row for OD-EVOL-12 is PENDING, and the checklist rows
are still OPEN. That is the expected pre-promotion state and does not bear on the Outcome question.

No finding survives as a technical or honesty obstacle to a scoped promotion.

### 4.3 Footnote wording check

Candidate (OD-EVOL-12 / OD-EVOL-05b): "Qualification-layer deploy/rollback safety; not live PROD-Q production enforcement".

- It stays inside what was qualified on the "qualification-layer" side, but it is incomplete for three reasons:
  (a) it does not state that production `ActivationService.rollback` is not gated and would accept a rollback the assessor
  refuses (LIM-05, F-C-06); (b) "PROD-Q" is a qualification program name, so "not live PROD-Q production enforcement" can be
  misread as "enforcement exists but is not yet PROD-Q-qualified"; (c) as a literal it fails the preflight: B-06 requires the
  row to contain "qualification-layer" and "not claimed" and the id `LIM-05`, and B-11 requires the notes cell to equal the
  rendering of the certified disposition exactly.
- The certified disposition is
  `ELIGIBLE_PASS_EVOL_X_SCOPED(LIM-05; qualification-layer deploy/rollback safety; production ActivationService.rollback durable-state enforcement is NOT claimed)`.
  `render_promoted_row` renders it as status `PASS (EVOL-X SCOPED)` and notes
  `LIM-05; qualification-layer deploy/rollback safety; production ActivationService.rollback durable-state enforcement is NOT claimed`.
  This text is accurate against the code (F3) and satisfies the preflight.
- Conclusion: the short OD footnote is acceptable only as an additional gloss, never as a replacement for the rendered notes.

### 4.4 Other rows (FRZ-EVOL-01..06, 08..10) - report only, not deep-audited

All ten checklist rows are currently `OPEN` with the canonical requirement literals identical to
`FRZ_EVOL_CANONICAL_REQUIREMENTS`. The evidence file shows `passed: true` at TES for all ten. Issues with the OD-EVOL-12
Option A text:

B1. Status wording does not match the certified dispositions. The OD text authorizes FRZ-EVOL-01..06 and 09 as
`PASS (EVOL-X scoped)`. The certification record dispositions are: 02, 03, 04, 05 = `ELIGIBLE_PASS` (renders `PASS`, `-`);
01 = `ELIGIBLE_PASS_SCOPED(LIM-03)`; 06 = `ELIGIBLE_PASS_SCOPED(LIM-01,LIM-03,LIM-04; ...)`; 08, 09, 10 = `ELIGIBLE_PASS_SCOPED(...)`
(render `PASS - SCOPED`); only 07 is `ELIGIBLE_PASS_EVOL_X_SCOPED` (renders `PASS (EVOL-X SCOPED)`). Because B-11 forces
status and notes to equal the certified rendering, the OD sentence cannot be applied literally to 01-06, 08-10. The OD text
should authorize "promotion to the status rendered from each certified disposition" rather than naming one status for
all rows.

B2. Limitations dropped from the OD text. Rows 01 (LIM-03), 06 (LIM-01/03/04 and "EVOL_CONSERVATIVE_FULL scope only"), 08
(LIM-02, "M-09H not measured"), 09 (LIM-06) and 10 (LIM-07) carry mandatory limitation ids/text that the preflight requires in
the promoted row. The OD text mentions only "evidence footnotes referencing evol_frz_evidence.v1.json" and, for 08/10, a
passive-dossier statement; it omits M-09H NOT_MEASURED for FRZ-EVOL-08, which the preflight requires verbatim (`M-09H`,
`not measured`). Also, the evidence bundle's own `scope_footnote` is empty for 01-06 and 09, so "evidence footnotes" cannot
supply the scope text for those rows; the scope text must come from the certified dispositions.

B3. FRZ-EVOL-08 substance (from the auditor matrix, not re-derived by me): "what failed" and "which deploy introduced
regression" are covered only through seed attribution, with M-09H NOT_MEASURED. The row must keep that scope; an unscoped or
purely "passive dossier" wording would overstate.

B4. Fixed-order dependency: the OD package preconditions require `B6_ACCEPTANCE_PUBLICATION_SHA` published and the independent
Outcome A confirmation; the B6 acceptance record is signed and published at HEAD, so only this Outcome confirmation and the
operator OD-EVOL-12 act remain.

None of B1-B4 affects the FRZ-EVOL-07 Outcome; B1 and B2 should be corrected in the OD-EVOL-12 text before signature, or the
operator should state that statuses and notes follow the certified dispositions.

## 5. Conclusion

Justification: The canonical wording of FRZ-EVOL-07 ("qualified or unsafe refusal explicit") is satisfied at the qualification
layer by a total, fail-closed DM-01..07 assessor with typed refusal reasons, an S7 positive and negative arm bound to TES, a
384-combination exhaustive check recorded by the independent audit, and evidence unchanged since FINAL with no relevant code
drift after TES; OD-EVOL-05b (Option 1) is signed and the enforcement-boundary document explicitly allows this scoped reading.
What is not qualified is production enforcement: `ActivationService.rollback` has no durable-state check and the S7 record shows
it accepts the rollback the assessor refuses (LIM-05, F-C-06), the assessor works on declared sidecar facts rather than live
state, and PROD-Q is untouched. A scoped promotion is therefore honest only if the promoted row carries that limitation
verbatim; because the preflight (B-06/B-11) forces the row's status and notes to equal the certified rendering, which includes
LIM-05 and "NOT claimed", Outcome A holds on that condition. The shorter OD footnote ("not live PROD-Q production enforcement")
is not sufficient on its own and must not replace the rendered notes. This outcome does not authorize anything; promotion still
requires the operator's separate OD-EVOL-12 act and the Gate B checks.

Acceptable footnote / notes text for the FRZ-EVOL-07 row (status cell `PASS (EVOL-X SCOPED)`):

`LIM-05; qualification-layer deploy/rollback safety; production ActivationService.rollback durable-state enforcement is NOT claimed`

Optional additional gloss (only after the text above, never instead of it): `Qualification-layer deploy/rollback safety; not live PROD-Q production enforcement`.

Blocker for OD-EVOL-12 Option A as drafted: not for FRZ-EVOL-07 itself, but the OD text must be aligned with the certified
dispositions (findings B1, B2) before the operator signs, or the preflight will reject the promoted rows.

Outcome: A
