# OD-AF-RC-KR08 — merge mandate for red `governance-fast` on the RW train

**Status:** `GRANTED` (narrow scope: **PR #680 only**).

**Operator decision:** TAK — recorded **2026-11-21T05:47:00+01:00** (session OD-AF-RC-KR08).

**Question (plan section F):** May the orchestrator merge train PRs while `governance-fast` is red, when the failure is exactly the ACT-N11 stale-audit class (KR-08), before FCS?

**Granted scope (mandate):**

- Merge **only** PR #680 into `arch-freeze/rw-train`.
- Preconditions: train pin at merge time `28c27cb896d02e0185b9fa02047411fa152f6469`; `governance-fast` on #680 is **expected-red** with signature **ACT-N11** (stale audit after material governance / product-path delta).
- If `governance-fast` is red for any other class, or the failure log lacks ACT-N11 stale-audit dominance → **STOP**, no merge, report.
- Does **not** grant merge for #684, #682, or other train PRs with red `governance-fast`.
- Does **not** grant FCS / G2 / M3 / CLOSED; closure remains PT-Q03 / RB-2.

**Pre-merge CI proof (#680, head `a6999e97bb46561e1ba578c9364c13ed310f1d54`):**

| Field | Value |
| --- | --- |
| Workflow run | `44690933116` |
| URL | https://github.com/jakbuczarnecki/intergrax/actions/runs/44690933116 |
| Check `governance-fast` | **FAILURE** (completed `2026-11-20T07:02:01Z`) |
| Dominant signature | **ACT-N11** stale audit after material governance change; product-path deltas vs audited SHA (KR-08 class) |
| Other PR checks | `ci-smoke`, `ticket-governance`, `ent-audit-fast`, `evol-x-b4-preflight`, `governance-policy-pin` — SUCCESS |
| Classification | Expected-red / KR-08 cluster; see `eqo_closure/pr680_governance_fast.md` |

Orchestrator verified run logs: failures in `test_evol_x_real_repo_fresh_clone.py`, `test_iiac_activation_gate.py`, `test_iiac_negative_guards.py` cite ACT-N11 stale audit lines, not unrelated gate regressions.

**Historical (RC-1):** Status was `PENDING_OPERATOR`; no merge until this grant.

**Consequence under grant:** merge #680 after this record and delegated ticket rules; post-merge train refresh and RC-2 dossier update; **no** RC-3+ start before RC-2 exit criteria from plan.