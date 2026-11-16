# EVOL-X W7-06 step 2 - D10 digest chain attestation (B6 party, verifier session)

Date: 2026-11-08. Subject: EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md section 4 ("D10 digest chain") and the
"Reconciliation rule", plus the section 6 oracle row (CI JUnit canonical hash).

## 1. Verifier statement

This attestation was produced by an AI session (Claude Code subagent, model Sonnet 5.5) acting as an
independent verifier. It did not implement, publish, certify or sign anything in EVOL-X. Assurance level:
process_enforced. Independence from the implementing sessions is NOT provable from git; it rests on this
session having been started fresh and having written no tracked file. No tracked file was edited, nothing was
pushed, no branch was created, no Approved line was written, and the main checkout was only read (two
earlier-session logs were read for command syntax only; their results were not relied on and every check below
was re-run).

Environment: fresh clone of https://github.com/jakbuczarnecki/intergrax.git at
D:/Projekty/_wt/intergrax/_scratch/d10/c, detached; venv outside the clone (uv venv v --python 3.12;
UV_PROJECT_ENVIRONMENT=v, uv sync --extra dev-ci --frozen); PYTHONPATH unset except PYTHONPATH=<clone> on the
verifier commands; one python/uv process at a time. origin/development at verification time was
cc86e4d805ed6a601bbe642ddf8f09243a4d36d0 (PR #396), later than the W7-05 merge 65ccf681a (verified to be an
ancestor of it).

Pinned baseline (handoff record section 2 and EVOL_X_SHA_LADDER.md; the two agree):
TES 3c4a641b3e48a13f437e830a0562e2a80b03aabb; FINAL bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1;
qualified CI run 41337077334; plan publication b4fca4bc413a650b2e5a8dd8826669a9025864c5;
canonical JUnit sha256 69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4.

## 2. Check table

BUNDLE = docs/project/maintainers/qualification/evol_x/evol_x_p03_independent_replay
T = TES, F = FINAL, OD = origin/development.

| ID | Check | Command (abridged) | Exit | Result |
|----|-------|--------------------|------|--------|
| C1a | TES ancestor of FINAL (reconciliation rule) | git merge-base --is-ancestor T F | 0 | PASS |
| C1b | TES ancestor of origin/development | git merge-base --is-ancestor T OD | 0 | PASS |
| C1c | FINAL ancestor of origin/development | git merge-base --is-ancestor F OD | 0 | PASS |
| C2a | MANIFEST @ F: test_execution_sha, implementation_sha, independent_replay_sha | git show F:BUNDLE/MANIFEST.json | 0 | PASS, all three = 3c4a641b... |
| C2b | MANIFEST plan_publication_sha equals ladder PLAN_PUBLICATION_SHA | git show; read EVOL_X_SHA_LADDER.md | 0 | PASS, b4fca4bc413a650b2e5a8dd8826669a9025864c5 in both |
| C2c | MANIFEST profile, dirty tree, command | same | 0 | AUTOMATION_FIRST_SCOPED, dirty_tree=false, `uv run pytest tests/qualification/evol_x -q`, uv_extra dev-ci; no self-referential evidence_publication_sha field (conforms to master plan V-A) |
| C2d | Ladder TES / FINAL / CI / plan agree with handoff section 2 | read ladder and handoff record | 0 | PASS |
| C3a | ARTIFACT_DIGESTS: independent recompute of all 27 artifacts (sha256 of git blob bytes @ F via git show + sha256sum, no repo code) vs ARTIFACT_DIGESTS.json | shell loop + diff | 0 | PASS: 27/27 equal; key set equals MANIFEST artifact_paths (27) |
| C3b | Mode A (repo procedure, master plan V-0) @ FINAL | python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish (HEAD=F) | 0 | {"status":"EVIDENCE_INTEGRITY_OK","mode":"A","disposition":"final","evidence_head":"bbf3e33b..."}; slow (over 15 min: it rebuilds the closed-world inventory @ TES) |
| C4a | VL3 manifest evol_vl3_execution_manifest.v1.json @ TES tree | git show T:tests/qualification/evol_x/data/... + jq | 0 | PASS: 290 pytest_node_ids, source_authority master plan v3 II-A-T3, uv_extra dev-ci, content digest 3aba584c...bcc2f |
| C4b | VL3 blob identical at T, F and OD (no drift after TES) | git rev-parse X:path | 0 | PASS: blob c30fd7029ede248bbfe453ef85dafb3cb3756695 at all three |
| C4c | VL3 node ids equal bundle evol_test_execution_manifest.v1.json node_ids | diff of sorted lists | 0 | PASS: 290 = 290, source_commit_sha = TES, same manifest_content_digest |
| C4d | VL3 manifest self-check @ FINAL and @ TES | python scripts/qualification/evol_x_vl3_manifest.py --check | 0 / 0 | "VL3 manifest OK (290 node ids)" both |
| C4e | FRZ bundle evol_frz_evidence.v1.json @ F | git show + grep (and Mode A p03 checks) | 0 | PASS: exactly FRZ-EVOL-01..10, all passed=true, source_commit_sha = TES |
| C4f | Metrics evol_metrics.v1.json @ F | git show + inspection | 0 | EVOL-M-09A: MEASURED, passed true, 3/3, threshold 1.0, population digest aa0224aa...; EVOL-M-09H: NOT_MEASURED, passed null, 0/0 (note: "human drill not performed; OD-EVOL-13"); source_commit_sha = TES |
| C4g | Experiments S1..S8 (negative + positive arm each, 16 verdicts) @ F | git show + grep | 0 | PASS: 16/16 present, passed=true, source_commit_sha = TES |
| C4h | Other bundle artifacts bound to TES (inventory_binding, m03_causal_audit, M-09A qualification, C1) | git show + grep | 0 | PASS, all = TES |
| C4i | Bundle unchanged FINAL..OD; tests/qualification/evol_x and testing_support unchanged TES..OD | git diff --name-only | 0 | PASS (empty diffs). Only non-doc changes TES..FINAL: scripts/qualification/evol_x_finalize_automation_first_scoped.py and evol_x_run_m09a_automated_batch.py (evidence tooling, in commits 7d1523085 and bbf3e33bc) |
| C4j | Impact/selection modules under tests/qualification/evol_x/ @ TES (handoff row 7) | git ls-tree | 0 | present (gates/test_evol_x_selection*.py, gates/test_evol_x_intake_and_impact.py, frz/test_evol_x_frz_01,02,03,06*, adapters/selection_inputs.py); 80 files under the dir |
| C5a | CI run 41337077334 metadata | gh run view; gh api .../jobs; gh run view --log | 0 | conclusion success, workflow "Regression gate", job evol-x-p3-l3-qualification success; step "Checkout TEST_EXECUTION_SHA" success, its script asserts git rev-parse HEAD equals TEST_EXECUTION_SHA = 3c4a641b... |
| C5b | Retained CI artifact (not expired) evol-x-p3-l3-3c4a641b3e48..., junit.xml | gh run download 41337077334 -n ... | 0 | 290 tests, 0 failures/errors/skipped; sha256sum junit.xml = 69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4 = canonical hash. MATCH |
| C5c | Mode B replay of the retained CI JUnit @ TES, certification mode | python scripts/qualification/evol_x_p03_evidence.py verify-test-execution --evidence-ref F --certification --replay-junit ci_art/junit.xml (HEAD=T) | 0 | {"status":"EVIDENCE_INTEGRITY_OK","mode":"B","test_execution_sha":"3c4a641b..."}: CI outcomes equal bundle evol_junit_outcomes (290 passed); all 290 manifest nodes present and passed |
| C5d | Negative control: same replay on a COPY of the junit with node anti_vacuity[S1] removed | same, junit_dropped.xml | 1 | Fails closed (KeyError traceback, exit 1; not a clean EVIDENCE_INTEGRITY_FAIL message). Not a false pass. Observation only |
| C5e | Local re-execution of the 290-test pytest suite | not run | - | Not done by this session (see section 3); the CI run's own JUnit was replayed instead |
| C6 | Publication chain ancestry on origin/development | git rev-parse --verify; git merge-base --is-ancestor prev next | 0 each | see table 3 |
| C7a | Closure preflight @ origin/development cc86e4d80 | PYTHONPATH=<clone> v/Scripts/python.exe scripts/qualification/evol_x_closure_publication_preflight.py --gate formal-closure | 0 | "EVOL-X closure preflight: PASS"; Gate A PASS; Gate B PASS; policy mode IIAC; repository canonical |
| C7b | IIAC policy activation | PYTHONPATH=<clone> v/Scripts/python.exe scripts/governance/verify_iiac_policy_activation.py --require-in-force | 0 | "IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d" |

### Table 3 - publication chain (C6)

| Step | SHA (resolved full) | Ancestor of origin/development | Strict ancestor of next | Recorded in |
|------|---------------------|-------------------------------|--------------------------|-------------|
| P2 certification publication | ad0877715ed079ce920ff9d925fda03555b2c04e | yes | yes (of 1dd76933) | certification record JSON certification_publication_sha; closure evidence index line 7 |
| B6 phase 1 (B6_ACCEPTANCE_PUBLICATION_SHA) | 1dd769331506149a1ee2f411c2bc62e85e5bbf25 | yes | yes (of 109029b2c; phase-2 commit b288ba756 lies between) | SHA ladder; EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json; EVOL_X_CLOSURE_EXECUTION_STATUS.json; certification record JSON b6_acceptance_publication_sha |
| W7-03 | 109029b2ca29fe0ba1b95a0f66ae4c243fdf9e5b | yes | yes (of 5f9727032) | closure evidence index line 9 (short SHA); OD-EVOLX-08 delegation doc (short SHA); short SHA resolves uniquely to this full SHA |
| W7-04 | 5f972703237268ef22a477c8954842508301dd51 | yes | yes (of 50ddc04fb) | closure evidence index (short); certification record JSON blocker B-5 closure_publication_sha (full) |
| B-5 resolution | 50ddc04fbabd24ac062267d5d33f14ccef89840c | yes | yes (of be77b3a5f) | closure evidence index (short SHA only) |
| FORMAL_CLOSURE_PUBLICATION_SHA | be77b3a5fb97f10a3bd2df6ee038cb92aad77eca | yes | n/a (tip of chain; 65ccf681a and cc86e4d80 descend from it) | SHA ladder; EVOL_X_CLOSURE_EXECUTION_STATUS.json formal_closure_publication_sha; certification record JSON formal_closure_publication_sha |

Every SHA in the task resolved uniquely (short ones included); the ladder, status JSON and certification JSON
values equal the SHAs given in the task. git log of the range shows the six commits in this order with subjects
matching their roles.

## 3. Items NOT verified, and why

1. Full local re-execution of the 290-node pytest suite: not done (about 29 min per the earlier-session log; the
   task asked to replay the retained CI JUnit). Mode B parity was established against the CI JUnit only.
2. Independence of the earlier sessions and of the signers: not provable from git (operator signatures are
   process-enforced text lines, per the preflight contract). This session verifies bytes, SHAs and gates, not
   identities.
3. The CI runner environment and job contents were taken from GitHub's API/log as served; only the artifact bytes
   were hashed locally (hash matched).
4. Handoff section 2 rows other than TES/FINAL/CI/plan publication (R2 reconciliation 167583ad..., local B6
   unattended 42363732...) were outside this D10 task and were not checked.
5. The full SHAs of W7-03 (109029b2c) and B-5 (50ddc04fb) are not recorded in machine-readable form anywhere in
   the repo; only short SHAs in prose (index, delegation doc). W7-04 appears in full only as the B-5
   closure_publication_sha in the certification JSON. Observation, not a defect of the chain.

## 4. Observations (non-blocking)

- origin/development had moved past the W7-05 merge (cc86e4d80, PR #396) when verified; the preflight PASS is for
  that tip. The EVOL-X chain, the evidence bundle and the VL3 manifest are byte-identical at that tip to the
  pinned versions.
- Mode B replay fails closed with an unhandled KeyError (exit 1) instead of a clean EVIDENCE_INTEGRITY_FAIL when a
  manifest node is missing from the replay: detection is correct, error reporting is rough.
- Between TES and FINAL two evidence tooling scripts changed (C4i); no test file, testing_support or product module
  differs between TES and origin/development within tests/qualification/evol_x and testing_support.

## 5. Verdict

D10 reconciliation: PASS

Reason: the reconciliation rule holds (TES is an ancestor of FINAL, and both are ancestors of origin/development);
the evidence MANIFEST at FINAL carries test_execution_sha and plan_publication_sha equal to the ladder; all 27
ARTIFACT_DIGESTS were recomputed independently from git blobs and match, and the repository's own Mode A returned
EVIDENCE_INTEGRITY_OK at FINAL; the VL3 manifest at the TES tree (290 nodes, identical blob through
origin/development), the FRZ bundle (FRZ-EVOL-01..10 passed), the metrics (M-09A MEASURED PASS, M-09H NOT_MEASURED)
and the 16 S1..S8 experiment verdicts all bind to TES; the retained artifact of CI run 41337077334 hashes to the
canonical 69c6a9ad...15ea4 and its replay passes Mode B in certification mode (a negative control fails closed);
the six-step EVOL-X publication chain is a strict ancestor chain on origin/development and equals the recorded
SHAs; and the formal-closure preflight and the IIAC in-force check both exit 0. The only gap (no local re-run of
the full 290-test suite) is covered by the CI-artifact replay and does not change the result.
