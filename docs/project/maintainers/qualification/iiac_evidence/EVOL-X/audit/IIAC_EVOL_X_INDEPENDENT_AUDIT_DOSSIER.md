# EVOL-X — Independent IIAC audit dossier (Session C)

**Verdict:** `EVOL-X — INDEPENDENT IIAC AUDIT PASS` — manifest verdict **`PASS_SCOPED`**
**Role:** `independent-quality-auditor` · **Policy:** `IIAC-2026-11-05.1` (`IN_FORCE`) · **Date:** 2026-11-05 (+01:00)
**Audited implementation / TES:** `3c4a641b3e48a13f437e830a0562e2a80b03aabb` · **Evidence FINAL:** `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1`
**Audit session head:** `06a459afbab3aa8cea89e4118865d55e7b537a34`

> This verdict is an **audit** verdict. It is **not** EVOL-X closure, **not** FRZ promotion, **not** a roadmap or ladder change. The auditor does not self-certify. Closure belongs to the separate `certification-governance-agent` publication.

## 1. Verdict and basis

| Item | Result |
|---|---|
| Independent Mode A @ FINAL | PASS (exit 0, `evidence_head == FINAL`) |
| Independent Mode B @ TES (`--certification`, auditor's own JUnit) | PASS (exit 0), 290/290, outcome map identical to the published bundle |
| Regression at current HEAD (`06a459afb`) | 290/290 passed, outcome map identical |
| CI provenance (run `41337077334`) | success; TES checkout assertion present in job log |
| Mode C (adversarial) | 12/12 Mode A tamper cases, 5/5 Mode B replay cases, 3 CLI negatives rejected; 2 of the Mode B rejections come from an unhandled exception (F-C-01) |
| Independent re-derivation | 27/27 artifact digests; 16/17 metric readings identical (M-09A re-derived by Mode A); 0 missed production consumers in 60 real changes; 0 under-selection in 56 real paths; exhaustive DM (384) and WM (128) enumeration |
| FRZ-EVOL-01..10 | 10 rows evaluated individually: 4 `PASS` (02, 03, 04, 05), 6 `PASS — SCOPED` (01, 06, 07, 08, 09, 10); see `frz_evol_acceptance_matrix.md` |
| Findings | 0 `PROVEN_ARCHITECTURAL_DEFECT`, 0 `IN_SCOPE_BLOCKER`, 0 `MISSING_EXECUTION_EVIDENCE` against the scoped requirements; 11 recorded findings (3 MEDIUM, 4 LOW, 2 PROCEDURAL, 2 INFO) |

The verdict is `PASS_SCOPED`, not an unqualified `PASS`, because the audited scope is the one the operator approved: `AUTOMATION_FIRST_SCOPED` (OD-EVOL-13, `M-09H NOT_MEASURED`), qualification-only FRZ-EVOL-07 (OD-EVOL-05b Option 1), and the `EVOL_CONSERVATIVE_FULL` selection scope (OD-EVOL-10). Every scoped item is listed as LIM-01..LIM-10 and must travel with any promotion text. Nothing was upgraded to a wider claim.

## 2. Phase C0 — prerequisites

| Check | Result | Evidence |
|---|---|---|
| IIAC `IN_FORCE` | yes, `effective_from_sha = f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` | `IIAC_POLICY_ACTIVATION_RECORD.json` |
| `verify_iiac_policy_activation.py --require-in-force` | exit 0 | `evidence/logs/iiac_gate_c0.txt` |
| Activation SHA is ancestor of HEAD | yes | `evidence/logs/baseline_facts.txt` |
| HEAD == expected start | `06a459afbab3aa8cea89e4118865d55e7b537a34`, equal to `origin/development` | same |
| Independence from EVOL-X implementers and the activation session | satisfied as a fresh Claude Code session (id `a18c2476-3b59-4cc7-bc0f-fa22b25edcfc`); no commit, no tracked-file edit | `audit_scope.json`; operator attestation record still pending (F-C-09) |
| TES / FINAL / evidence SHAs reconstructed | yes, from `EVOL_X_SHA_LADDER.md`, then verified against git (all commits, all ancestors of HEAD, TES ancestor of FINAL, TES tree = MANIFEST `parent_test_execution_tree`) | `evidence/logs/baseline_facts.txt` |

The ladder rows `B6_ACCEPTANCE_PUBLICATION_SHA` and `FORMAL_CLOSURE_PUBLICATION_SHA` are **PENDING**. The auditor did not set them.

## 3. Exact-SHA baseline and drift

| Role | SHA |
|---|---|
| PLAN_PUBLICATION_SHA | `b4fca4bc413a650b2e5a8dd8826669a9025864c5` |
| TEST_EXECUTION_SHA (TES) | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_TECH_PUBLICATION_SHA (historical) | `1e175ff8f678d605aa841ae52362486a56ed006d` |
| EVIDENCE_FINAL_PUBLICATION_SHA (FINAL) | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| Qualified CI run | `41337077334` @ TES |
| Internal B6 / closure publication (not certifying) | `ee2c89a0d18b43d7667e7ee2c5df3d446f2a73e1` |
| IIAC certification publication (governance) | `a5fb6cb51ecd982284fd53ebc41e284e18b5523d` |

Drift analysis:

- TES → FINAL: only `scripts/qualification/evol_x_finalize_automation_first_scoped.py`, `evol_x_run_m09a_automated_batch.py` and the evidence bundle. 0 files under `intergrax/ agents/ applications/ testing_support/ tests/`.
- FINAL → HEAD: 0 bundle changes. 0 EVOL-X-scope changes (`testing_support/change_safety`, `tests/qualification/evol_x`, `tests/fixtures/evol_x`) between TES and HEAD.
- TES → HEAD: 34 files under `intergrax/ agents/ applications/` changed. They belong to ENT-AUDIT-X implementation commits by the repository owner (e.g. `intergrax/platform/enterprise_audit/**`, an import reorder in `intergrax/applications/_shared/health_score_wiring.py`). They are outside the EVOL-X audited tree and the EVOL-X suite still passes 290/290 on top of them.

## 4. Inventory (complete EVOL-X surface)

| Surface | Count @ TES | Notes |
|---|---:|---|
| `testing_support/change_safety/**` | 46 files (192,980 bytes) | implementation: discovery, graph, assess, selection, diagnostics, evidence, metrics |
| `tests/qualification/evol_x/**` | 80 files → **290 VL3 node ids** | gates 217, experiments 25, FRZ 22, metrics 26 (the rest are adapters, data and helper modules) |
| `tests/fixtures/evol_x/**` | 26 files | S1..S8 fixture trees |
| FINAL evidence bundle | 27 artifacts | MANIFEST, digests, outcomes, metrics, FRZ, 16 experiment verdicts, M-09A set |
| EVOL-X records under `qualification/evol_x/**` | 360 git blobs at HEAD (inventory file lists every blob SHA) | W1 documentation, B6 technical qualification, internal closure, OD records |

`evol_x_record_inventory.txt` lists every file with its git blob SHA at HEAD. Records were **classified and used as pointers**; the audit's evidence is executable (replay, probes, recomputation), as ECAP §1.7 requires. Documents read: master plan v3 Parts II, VII, VIII, IX, X; IIAC protocol, activation record, migration matrix, publication contract and manifest schema; ECAP §1.6, §1.7, §2.B, §12, §13 and the FRZ-EVOL family; FRZ checklist EVOL rows, status vocabulary and PASS evidence rule; roadmap §3.0.9; SHA ladder; OD register header rows and OD-EVOL-13 policy; experiment/metric spec; internal B6 acceptance record and internal FRZ matrix; Protocol 03 automated record; B6 findings register; CI qualification record; the 102-ID reconciliation (section-level only, so the auditor produced `part_vii_id_matrix.json`). Not read line by line: the remaining ~340 narrative records (W1 documentation, other closure/terminal reports); they are inventoried with blob SHAs and none was used as evidence.

## 5. Method and execution

Replay environment: Windows 11, Python 3.12.11, uv 0.13.4, `uv sync --extra dev-ci --frozen` (`uv.lock` sha256 `faa8445d…` identical at TES and FINAL), short-path detached worktrees under `build/iiac-c/`, pytest basetemp outside the worktree. The auditor did not use the implementers' orchestrator output as evidence; it ran the canonical P03 CLI and its own probes. Full command matrix: `test_matrix.txt`.

### 5.1 Mode A @ FINAL
`verify-evidence-publish` exit 0, `evidence_head == bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1`. The auditor additionally recomputed all 27 `ARTIFACT_DIGESTS` over git blob bytes with its own script: 27 ok, 0 bad, MANIFEST `artifact_paths` equals digest keys.

### 5.2 Mode B @ TES
VL3 manifest check: 290 node ids. `pytest tests/qualification/evol_x`: **290 passed in 767.63 s**. `verify-test-execution --evidence-ref FINAL --certification --replay-junit <own junit>` exit 0. The canonical outcome map of the auditor's replay equals `evol_junit_outcomes.v1.json` key-for-key and outcome-for-outcome. Auditor JUnit sha256 `fe8abfce17f9c08584bffae211a1cd309900b631c9a36b50da97f55909bc2a05`; CI JUnit sha256 `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4` (not byte-comparable by design; the comparison is semantic).

### 5.3 CI provenance
`gh run view 41337077334 --log`: success; the step `Checkout TEST_EXECUTION_SHA` runs `git checkout "${TEST_EXECUTION_SHA}"` then `test "$(git rev-parse HEAD)" = "${TEST_EXECUTION_SHA}"`; `290 passed in 530.02s`. The run's `headSha` `a00674d6…` is the workflow-definition commit on `main` (PR #380 merge), not the test tree, exactly as the CI record says. A Windows replay and an Ubuntu CI run both pass the same committed digests: this is independent evidence for CW-08 cross-platform determinism.

### 5.4 Regression at HEAD
The same suite at `06a459afb`: 290 passed in 777.07 s, outcome map identical to the TES replay. No cross-stage regression from ENT-AUDIT-X or IIAC changes.

### 5.5 Mode C — adversarial
| Probe | Cases | Result |
|---|---:|---|
| Mode A tamper, adversary re-signs `ARTIFACT_DIGESTS` over blob bytes (metric flipped, FRZ row failed, FRZ-07 footnote removed, experiment arm failed, forged `M-09H` MEASURED, stale `source_commit_sha`, junit outcome flipped, `human_m09h_measured` forged, diagnostic output byte tamper, experiment verdict deleted, profile flipped to `HUMAN_FULL`, plus a no-resign byte tamper) | 12 | 12/12 detected, each by the correct check (`probe_tamper_v2.log`) |
| Mode B replay tamper (test failed, skipped, node missing, forged extra node, empty replay) | 5 | 5/5 rejected; MB3 and MB5 by `KeyError` (F-C-01) |
| CLI: wrong `--evidence-ref` (TES), certification without `--replay-junit`, JUnit missing one testcase | 3 | exit 2, exit 2, exit 1 (traceback; F-C-01) |
| Selection safety: 56 stratified real paths + 3 unknown paths | 59 | 0 under-selection against an independent AST oracle; unknown paths → `CONSERVATIVE_FULL`; selected-set sizes only 57 or 347 (F-C-04) |
| Impact graph vs independent AST reverse-import closure: 60 real changes | 60 | 0 production modules missed of 75,380 closure elements; 4,161 test modules missed (F-C-03) |
| Deployment matrix, all 384 `DeployFacts` combinations | 384 | no eligible verdict for missing evidence, forward-only rollback, incompatible persisted schema, unknown operation or non-drainable pins |
| Workflow matrix, all 128 `WorkflowChange` inputs | 128 | 0 in-flight adoptions of a changed definition without rejection/migration |
| Extension-safety classification of every tracked file | all | no `intergrax/*.py` outside the 9 frozen surfaces classifies `EXTENSION_ONLY` (all `UNCLASSIFIED`, Class C) |

One auditor invariant (I4: no `ELIGIBLE` with a non-`COMPATIBLE` contract) flagged 4 combinations; review showed they are DM-07 roll-forward with `MIGRATION_REQUIRED` and a forward-only migration already applied, which is the migrated state by definition. Dismissed as an over-strict auditor invariant, not a defect.

### 5.6 Independent re-derivation of metrics
`collect_all(inventory, head)` at TES vs the published `evol_metrics.v1.json`: 16 of 17 readings identical in numerator, denominator, population digest and status. The one difference is `EVOL-M-09A`, which is a not-measured placeholder inside `collect_all` by design and is produced by the automated M-09A batch at FINAL; Mode A re-derives its registry oracle, cause identities and diagnostic digests (passed). Thresholds in `test_evol_x_metrics.py` equal the ratified canonical values (OD-EVOL-03, no lowering).

## 6. Results by audit dimension

| Dimension | Result |
|---|---|
| Architectural and functional invariants | `testing_support/change_safety` imports nothing from `intergrax/ agents/ applications/` at TES (static scan: 0 matches); production tree does not import `testing_support`/`tests` (gate passed); no duplicate authority gates passed (no parallel suite registry, schema authority is the platform registry, T1/T2 derived from QUAL-X). The assessor is read-only and mints no permission. |
| Implementation correctness and change safety | impact derivation is graph-based with 19 typed REL extractors; unknown → conservative; matrices (DM/WM) are data-driven, first-match, with a fail-closed catch-all; both exhaustively enumerated by the auditor. Defects: F-C-01 (verifier KeyError), F-C-03 (test ownership), F-C-07 (registry reach). None creates a false-pass path. |
| Completeness of mandatory acceptance criteria | `part_vii_id_matrix.json`: 105 rows covering every ID in master plan Part VII (A 10, B 10, C 10, D 10, E 8, F 17, G 17, H 11, I 12). Mechanical exit criteria from roadmap §3.0.9 all hold: unclassified surfaces 0, orphan consumers 0 (S2), unknown-impact bypass 0 (S6/M-04), unsafe in-flight adoption 0 (S3), unqualified rollback claimed safe 0 (S7, qualification layer), unauthorized core edit 0 (S8). |
| Mode A / B replay | PASS (see §5.1–5.4). |
| Mode C | PASS (§5.5). |
| Negative testing and fail-closed | 52 `test_evol_x_p03_integrity` negatives plus the auditor probes; every probe is rejected; one rejection by exception (F-C-01). |
| SHA provenance and evidence integrity | PASS: ancestry, tree binding, digests, MANIFEST has no self-referential field, TES→FINAL touches no product file, bundle unchanged FINAL→HEAD. |
| Cross-stage dependencies | EVOL-X consumes QUAL-X catalog and COMPAT policy without re-owning them; ENT-AUDIT-X references EVOL-X only through `ENT-A-EVOL-*` checks and path mapping, and uses its own impact map; the EVOL-X suite passes at HEAD. RC-7 baseline defect still reproduces (F-C-11, out of scope). |
| Known limitations and residual risks | LIM-01..LIM-10 below. |
| FRZ-EVOL-01..10 completeness | all 10 evaluated individually (`frz_evol_acceptance_matrix.md`). |

## 7. Findings register

Full ECAP §13 fields are in `EVOL-X_audit_findings.json`.

| ID | Sev | Class | Summary | Blocks scoped PASS |
|---|---|---|---|---|
| F-C-01 | LOW | NONBLOCKING_FUTURE_IMPROVEMENT | `verify_mode_b` raises `KeyError` when the replay omits a manifest node; fail-closed but exit 1 + traceback, not exit 2 + diagnosis | no |
| F-C-02 | LOW | TRACKED_FREEZE_DEBT | FRZ-EVOL-06 `scope_footnote` is empty in FINAL; plan Part II-B requires the scope footnote | no (LIM-01 carried) |
| F-C-03 | MEDIUM | TRACKED_FREEZE_DEBT | TESTED_BY edges are direct-import only: 4,161 test modules (2,803 suites) in the AST closure of 60 changes are not enumerated; 0 production misses | no (LIM-03) |
| F-C-04 | LOW | TRACKED_FREEZE_DEBT | M-03 is measured on six self-surface arcs that all select the same 347 nodes; declared oracle is null so the cross-check is tautological | no (LIM-04) |
| F-C-05 | MEDIUM | TRACKED_FREEZE_DEBT | FRZ-EVOL-08: dossier + M-09A cover change-time diagnostics only; M-09H NOT_MEASURED; no failure/deploy-identity ingestion | no (LIM-02) |
| F-C-06 | MEDIUM | TRACKED_FREEZE_DEBT | FRZ-EVOL-07: production `ActivationService.rollback` accepts a rollback the assessor refuses (`ad_layer_alone_succeeds=True`) | no (LIM-05); blocks any production claim |
| F-C-07 | LOW | NONBLOCKING_FUTURE_IMPROVEMENT | Frozen-core registry names 9 surfaces; `capability_edge_catalog.json` and 405 markdown files under `intergrax/` classify EXTENSION_ONLY | no (LIM-06) |
| F-C-08 | PROCEDURAL | ENVIRONMENT_TEST_ISSUE | strict role-separation validator would fail for `audit_execution_sha = HEAD` because of unrelated ENT-AUDIT-X drift; preflight still carries legacy OD gates and accepts filename-only matches | certifier action |
| F-C-09 | PROCEDURAL | MISSING_EXECUTION_EVIDENCE | no operator-attested Session C handoff record; implementer ids unrecorded; certifier fields unknown | certifier/operator action |
| F-C-10 | INFO | ENVIRONMENT_TEST_ISSUE | CRLF/short-path pitfalls; the auditor's first tamper probe was invalid and is superseded | no |
| F-C-11 | INFO | TRACKED_FREEZE_DEBT | RC-7 `test_reference_harness_strict.py` ImportError still reproduces at HEAD | no (out of scope) |

## 8. W7-03 determination — FRZ-EVOL-07 canonical wording (master plan Part X)

Part X requires the independent auditor to confirm whether the exact canonical checklist wording supports a qualification-only scoped claim before any FRZ-EVOL-07 promotion.

- **Wording** (checklist row FRZ-EVOL-07): *deployment / rollback / roll-forward qualified or unsafe refusal explicit*; note: *rollback not 'safe' from binary restart alone — durable state + contract compatibility required (S7)*.
- **Evidence:** the DM-01..07 assessor evaluates durable-state and contract-compatibility facts and returns typed `REFUSE` reasons; S7 shows the safe path (`dm_outcomes=7/7`, real AD rollback harness) and the unsafe path (DM-04 refusal, `false_safe=0`); the auditor's 384-combination enumeration found no unsafe `ELIGIBLE`.
- **Determination:** the wording **supports** a scoped claim that unsafe rollback is *explicitly refused at the qualification layer* and the safe case is *qualified*. It does **not** support a claim that unsafe rollback is blocked in operator environments: the same S7 record shows the production rollback layer alone would succeed (`ad_layer_alone_succeeds=True`).
- **Allowed wording after certification:** `PASS (EVOL-X scoped)` with the footnote *qualification-layer deploy/rollback safety; production ActivationService.rollback durable-state enforcement is NOT claimed (OD-EVOL-05b Option 1)*. Forbidden: any global, PROD-Q or production-enforcement claim.

## 9. Limitations that must travel with the verdict

| ID | Limitation |
|---|---|
| LIM-01 | Selection scope is `EVOL_CONSERVATIVE_FULL` (T1 mandatory, T2 Pop-A, T3 EVOL VL3, Adj). It is not `PLATFORM_CONSERVATIVE_FULL`/T4. DIM-B and the RC-7 test are excluded by OD-EVOL-10. |
| LIM-02 | `AUTOMATION_FIRST_SCOPED` (OD-EVOL-13): `EVOL-M-09H` is NOT_MEASURED; M-09A proves tool-side attribution (≤120 s, seed match) only; the human attribution SLA (≤900 s median) was never measured. FRZ-EVOL-08 is change-time diagnostics; incident and deploy correlation is ENT-AUDIT-X. |
| LIM-03 | TESTED_BY ownership is direct-import only (F-C-03); production consumer detection is complete in the sampled population. |
| LIM-04 | M-03 evidence shows internal consistency on self-surface arcs under oracle O-X, not discrimination power (F-C-04). |
| LIM-05 | FRZ-EVOL-07 is qualification-layer only; production rollback is not gated (OD-EVOL-05b Option 1; W4-04 not authorized). |
| LIM-06 | Frozen-core registry covers 9 surfaces; the rest is fail-safe UNCLASSIFIED (Class C); data files outside the registry are exempt. |
| LIM-07 | FRZ-EVOL-10: executable triggers T1..T9 and freshness counter are evidenced; the scenario-wave audit cycle and post-freeze requalification run are ENT-AUDIT-X's, not executed here. |
| LIM-08 | The audited tree is TES/FINAL. ENT-AUDIT-X and IIAC changes after TES are outside this audit; only the EVOL-X suite's continued pass at HEAD is asserted. |
| LIM-09 | Local replay ran on Windows only; Ubuntu evidence is the CI job log (checkout assertion and test count verified), not a second local run. |
| LIM-10 | `assurance_level = process_enforced`: no cryptographic proof of session separation; the manifest and handoff record are the controls. |

## 10. Authority, operator decisions and the IIAC migration

The auditor made **no** authority decision. Mapping under `IIAC_REQUIREMENT_MIGRATION_MATRIX.md`:

| Legacy gate | State in repo | IIAC treatment |
|---|---|---|
| OD-EVOL-01..10, 13 | signed/approved | Treated as the approved audit scope (LIM-01, LIM-02, LIM-05). |
| OD-EVOL-11 named human auditor | not named | Replaced by `auditor_session_id` + handoff record (operator attestation pending, F-C-09). |
| OD-EVOL-12 FRZ-08/10 promotion | PENDING in register | Certifier gate + per-row evidence (`frz_evol_acceptance_matrix`). |
| OD-EVOLX-08 parent closure | PENDING in register | Certifier + `FORMAL_CLOSURE_PUBLICATION_SHA` + preflight. |
| `ECAP-P4-EVOL-SCOPE-AUTO-B6` | `NOT_IN_FORCE`, superseded by IIAC | No dual activation; not relied on. |
| Internal B6 / OD-IC-01 / Protocol 03 automated record | historical | Not promoted. Reused only as pointers; every conclusion above was re-derived. |

## 11. Handoff to the certification-governance-agent

See `EVOL_X_IIAC_CERTIFICATION_HANDOFF.md`. Summary: complete the manifest fields (certifier ids, `certification_publication_sha`), obtain the operator handoff attestation, run `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py --strict-product-diff`, reconcile the legacy preflight rows with the migration matrix, publish `B6_ACCEPTANCE_PUBLICATION_SHA`/`FORMAL_CLOSURE_PUBLICATION_SHA` in dedicated commits (not in the commit that first creates the pass record), then promote FRZ rows one by one with the scoped text from the matrix.

## 12. Disclosures

- The auditor ran under Claude Code, model `claude-sonnet-5-5`; role registry gives `independent-quality-auditor` `may_commit: false`, so the auditor **did not commit or push**. All artifacts are untracked files for the authorized publication.
- Auditor tooling error (F-C-10): the first tamper probe hashed CRLF working-tree bytes and produced "detected" rows for the wrong reason; superseded and preserved.
- Probe scripts evolved during the audit (`probe_impact.py` is the final v4; logs v1–v3 predate its last edits and are superseded, v4 is relied on).
- The implementers' earlier Protocol 03 record and internal B6 record were read as context, not as evidence.
- Scratch worktrees (`build/iiac-c/*`, git-ignored) were removed after evidence was copied.
- `audit_execution_sha` in the manifest is FINAL (last exact-SHA replay checkout), with the repository HEAD at audit time recorded separately; see F-C-08.

## 13. Evidence index

`EVIDENCE_INDEX.json` lists the SHA-256 of every file in this directory (excluding itself). Key digests are repeated in the manifest draft.
