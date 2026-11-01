# EVOL-X — SHA ladder, W6-05 review worksheet (non-certifying) and W7 blockers

**Status:** implementation and evidence PREPARED. **No B6 PASS. No FRZ-EVOL row promoted. Roadmap not closed.**
Authority: `../EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md` (Part V-0).

## 1. SHA ladder (git facts only)

| Role | SHA | Note |
|------|-----|------|
| PLAN_PUBLICATION_SHA | `b4fca4bc413a650b2e5a8dd8826669a9025864c5` | master plan v3 |
| Batch B / C / D / E implementation | `506a2edd7` (C), `fd7daa6d3` (D), `bb4d1dec1` (E) | B before C on `development` |
| IMPLEMENTATION_SHA (last code commit) | `283fb0f4829de9146700d926986bb874ed827805` | Batch F: metrics + P03 verifier |
| TEST_EXECUTION_SHA | `283fb0f4829de9146700d926986bb874ed827805` | 224 passed, one VL3 pytest run, tracked tree clean |
| EVIDENCE_PUBLICATION_SHA | `git rev-parse` of the commit that adds `evol_x_p03_independent_replay/**` (child of TEST_EXECUTION_SHA, subject `evidence(EVOL-X): W6-06 …`) | **never stored in MANIFEST** (Part V-A) |
| B6_ACCEPTANCE_PUBLICATION_SHA | — | requires an independent auditor (see §3) |
| FORMAL_CLOSURE_PUBLICATION_SHA | — | depends on B6 |

Verifier: `uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish` (Mode A, run at the evidence commit) and
`verify-test-execution --evidence-ref <EVIDENCE_PUBLICATION_SHA> [--replay-junit <xml>]` (Mode B, run at TEST_EXECUTION_SHA).
Both exit 2 on any integrity failure; 8 negative cases are in `tests/qualification/evol_x/gates/test_evol_x_p03_integrity.py`.

## 2. W6-05 review worksheet (NON-CERTIFYING — for the independent auditor)

The auditor, not the implementer, marks each line. Nothing here is a verdict.

| # | Check | Evidence | Auditor mark |
|---|-------|----------|--------------|
| 1 | Mode A at evidence commit exits 0 | verifier output | |
| 2 | Mode B at TEST_EXECUTION_SHA exits 0; replayed canonical outcomes equal archived | `evol_junit_outcomes.v1.json` | |
| 3 | 16 experiment verdicts pass; negative arms are non-vacuous (naive baseline fails) | `experiments/*.json`, `tests/qualification/evol_x/experiments/` | |
| 4 | FRZ-EVOL-01..10 behavioural tests pass; footnotes 07/08/10 reviewed | `evol_frz_evidence.v1.json` | |
| 5 | Metrics M-01..11 measured on real populations at canonical thresholds (no lowering) | `evol_metrics.v1.json`, `tests/qualification/evol_x/data/metric_populations.v1.json` | |
| 6 | WM-01..15 / DM-01..07 matrices are implementation-defined (W1 docs missing) — review rows | `EVOL_X_WORKFLOW_DEPLOY_MATRICES_W4.md` | |
| 7 | M-09 operator attribution drill (median ≤ 15 min, ≥ 3 seeds) performed by a human | not automatable | |

Known honest limits for the auditor: M-01 history window is pinned to `anchor_sha` `bb4d1dec1` and classifies paths removed later by owner rule;
M-03 oracle R* is an independent ast reverse-import closure plus policy (mandatory, self-surface), not a human labelling;
M-09 tool part (determinism, ≤ 120 s) is measured, the human attribution part is not.

## 3. W7 / Batch G blockers (exact)

| Step | Blocker | Needed from |
|------|---------|-------------|
| W6-04 CI dispatch replay | workflow must be on `main` before a run id can be recorded (OD-EVOL-10); agent must not touch `main` | operator |
| W7-01 / W7-02 independent audit, B6 record | a person other than the implementer; OD-EVOL-11 auditor name not supplied; Mode B needs a detached checkout of TEST_EXECUTION_SHA (forbidden for the agent by branch-safety rule) | operator + auditor |
| M-09 attribution drill | human timing of the operator attribution | auditor |
| W7-03 FRZ-EVOL promotion | auditor must confirm FRZ-EVOL-07/08/10 scope wording | auditor |
| W7-04 | OD-EVOLX-08 signature (PENDING); OD-EVOL-12 PENDING | operator |
| W7-05 / W7-06 formal closure, roadmap §3/§5 | depend on B6 PASS record | operator + auditor |

Agent did **not**: publish a B6 PASS, edit canonical FRZ checklist rows, close the roadmap, change production config, migrate to Docker, or touch `main`.
