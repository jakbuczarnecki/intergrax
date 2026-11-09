# P4 — Independent product batch audit handoff

| Field | Value |
|---|---|
| Status | **READY** (auditor is a separate session) |
| Prerequisite | `hist-cert/product-batch` pushed; auditor frozen at `INTEGRATION_SHA` |
| INTEGRATION_SHA | `3a6ed857329177cc8e3b7669dbda53be1800c3df` |
| START_HEAD | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` |
| Worktree (planned) | `D:/Projekty/_wt/intergrax/hist-aud-prod` detached at `INTEGRATION_SHA` |
| Implementer session | `e48820fd-2405-4ac5-84f7-6e089f5c44ab` (must not audit) |

## Auditor mandate (P4-AUD)

Fresh session only. Re-execute the focused Phase 3 pytest commands on the integration SHA (one process at a time, `D:/Projekty/intergrax/.tmp/session/hist-cert-campaign/locks/heavy.lock`). Plus:

- `ruff check` / `ruff format --check` on touched product paths per lane
- `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force` — **expected red** (ACT-N11); record output, do not fix
- Per-lane `git diff --name-only fe0937517..<lane-tip>` ⊆ ticket allowed files
- Negative probes from master plan where applicable (scratch worktrees only)

## Lane tips (immutable audit subjects)

| Lane | Ticket | Branch | SHA |
|---|---|---|---|
| C1 | T-IGX-030-B | `hist-cert/c1-030b` | `088c9c5db` |
| C2 | T-IGX-030-C | `hist-cert/c2-030c` | `30f67c94a` |
| C3 | T-IGX-032-A | `hist-cert/c3-032a` | `b6a056e7d` |
| C4 | T-IGX-033-A | `hist-cert/c4-033a` | `8bfee643f` |

## Implementer verification summary (non-binding)

From P3 resume slice @ `INTEGRATION_SHA`:

- C1–C3 focused tests: exit 0
- C4 architecture/EBH-5 gates: exit 0
- C4 `test_cli_verification_hardening.py`: exit 1 (`canonical_governance_identity_required`); **same on clean `fe0937517`** at `hist-base` — document as environment/pre-existing, not 033-A regression unless auditor proves otherwise

Record `HIST-F-DUR-01` and `HIST-F-BND-03B` as open residuals (not implemented in batch).

## Deliverables (auditor owns)

Write under `historical_certification_campaign/audits/aud-product/`:

- `AUDIT_REPORT.md` — per-lane verdict: `PASS`, `PASS_SCOPED`, `PARTIAL`, or `FAIL`
- `findings.json` — exit codes, counts, command lines, session id

Do not merge to `development`. Do not sign IIAC activation.

## Authority

- Master plan Phase 4: [`HISTORICAL_CERTIFICATION_FINAL_MASTER_PLAN.md`](../../HISTORICAL_CERTIFICATION_FINAL_MASTER_PLAN.md)
- SHA ladder: [`../../HISTORICAL_CORRECTIVE_EXECUTION_AND_SHA_LADDER.md`](../../HISTORICAL_CORRECTIVE_EXECUTION_AND_SHA_LADDER.md)
- P3 evidence: [`../impl-campaign/P3_PRODUCT_IMPL_HANDOFF.md`](../impl-campaign/P3_PRODUCT_IMPL_HANDOFF.md)
