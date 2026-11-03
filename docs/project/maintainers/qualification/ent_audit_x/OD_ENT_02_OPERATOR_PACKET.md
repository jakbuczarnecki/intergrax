# OD-ENT-02 — Branch protection operator packet (ENT-AUDIT-X)

**Purpose:** External gate `BLK-CI-ENFORCE` / FRZ-AUD-10. Agent prepares; **operator + GitHub admin** execute.

**Repo:** `jakbuczarnecki/intergrax` · **Branch:** `development`  
**Reference runbook:** [ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md](./ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md)

## Required status checks (pin these exact names after green CI @ TEST_EXECUTION_SHA)

| Priority | GitHub check name (as shown on PR) | Workflow file | Job ID |
| --- | --- | --- | --- |
| P0 | `Enterprise audit (ENT-AUDIT-X) / ent-audit-fast` | `.github/workflows/enterprise-audit.yml` | `ent-audit-fast` |
| P0 | `Enterprise audit (ENT-AUDIT-X) / ent-audit-pr-gate` | `.github/workflows/enterprise-audit.yml` | `ent-audit-pr-gate` |
| P1 | Nightly enterprise audit (verify label @ Actions UI post-dispatch) | `.github/workflows/enterprise-audit-nightly.yml` | (nightly FULL profile) |
| P1 | Unit tests (existing org gate) | `.github/workflows/unit-tests.yml` | — |

## Operator checklist

1. Confirm latest **TEST_EXECUTION_SHA** is ancestor of `development` HEAD (`git merge-base --is-ancestor <TES> HEAD`).
2. Open GitHub → **Settings → Branches → Branch protection rules → `development`**.
3. Enable **Require status checks to pass before merging**.
4. Add checks from table above (FAST + PR gate minimum before B6 handoff).
5. Capture screenshot or `gh api` JSON of protection rule → store under operator-controlled evidence (not repo root).
6. Append operator record referencing OD-ENT-02 completion date + check names (do **not** mark FRZ-AUD-10 PASS until independent B6 PASS).

## Verification commands (operator machine)

```powershell
gh api repos/jakbuczarnecki/intergrax/branches/development/protection
gh run list --workflow=enterprise-audit.yml --branch development -L 5
gh run list --workflow=enterprise-audit-nightly.yml --branch development -L 3
```

## Agent status

- Workflows present @ HEAD; **branch protection HTTP 404** until operator completes steps above.
- Agent **does not** declare OD-ENT-02 approved or FRZ-AUD-10 CLOSED.
