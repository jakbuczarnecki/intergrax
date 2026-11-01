# EVOL-X — SHA ladder (SSOT, C-LADDER-01)

**Authority:** [EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md](../EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md) Part V-0.  
Root stub: [../EVOL_X_SHA_LADDER.md](../EVOL_X_SHA_LADDER.md) (pointer only).

| Role | SHA | Notes |
|------|-----|-------|
| PLAN_PUBLICATION_SHA | `b4fca4bc413a650b2e5a8dd8826669a9025864c5` | master plan v3 |
| W7 baseline (blockers doc) | `4393692a2e7447797c5e26408811e4b335c642f5` | pre-W7 A–D remediation |
| TEST_EXECUTION_SHA | `cb407403c686f464a17d8f97e1ba0177c15c5e3b` | one VL3 @ TES (243/243 passed) |
| EVIDENCE_TECH_PUBLICATION_SHA | `20a3bc63f3d34560e999905e16496a02087b75df` | Mode A `--technical`; M-09H NOT_MEASURED |
| EVIDENCE_FINAL_PUBLICATION_SHA | — | Batch E supplement + M-09H PASS |
| B6_ACCEPTANCE_PUBLICATION_SHA | — | independent auditor |

## W7 disposition

- **Technical evidence** never carries full certification PASS (no composite `EVOL-M-09`, no M-09H PASS).
- **Final evidence** adds `evol_m09_human_drill.v1.json` @ same `test_execution_sha`.
- **Mode B certification** @ TES: `--certification --replay-junit` mandatory.

## CI (W6-04)

See [EVOL_X_CI_QUALIFICATION_RECORD.md](./EVOL_X_CI_QUALIFICATION_RECORD.md) — **dispatch blocker OPEN**.
