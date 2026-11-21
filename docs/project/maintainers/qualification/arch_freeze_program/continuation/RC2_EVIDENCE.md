# RC-2 evidence dossier (in progress)

Worktree `D:/Projekty/_wt/intergrax/af-rc2`, branch `af2/rc2-continuation`. Base includes RC-1 checkpoint `36c470249`.

`OD-AF-RC-KR08` = **GRANTED** (2026-11-21, narrow #680). No G2 declaration.

## OD-AF-RC-KR08 pre-merge verification (2026-11-21)

- Command: `gh pr checks 680`, `gh run view 44690933116 --log-failed`
- Train pin at verification: `28c27cb896d02e0185b9fa02047411fa152f6469`
- PR head: `a6999e97bb46561e1ba578c9364c13ed310f1d54` (`af2/pt-fc1`)
- `governance-fast`: **FAILURE** — signature **ACT-N11** stale audit (expected-red / KR-08)
- Verdict: conditions for operator grant satisfied → authorized merge #680 only

## Train review pulls (no merge)

| PR | Branch | Head | Train base | governance-fast | Action |
| --- | --- | --- | --- | --- | --- |
| #680 | `af2/pt-fc1` | `a6999e97` | merged into branch | **FAILURE** (KR-08 ACT-N11) | **merge authorized** under OD-AF-RC-KR08 |
| #683 | `af2/rc1-continuation` | `36c470249` | atop `28c27cb` | not required for docs | open |
| 051-N PR | `af2/rw-supply-ab` | `df476721` | merged `28c27cb` | await CI | opened this package; merge gated like #680 |

## #680 digest review (read-only)

- Ticket: `T-IGX-043-A` (PT-FC1). Approval and plan digest present on branch.
- Product scope: Nexus routing posture, fail-closed persistence, FC-1 tests (NT-FC1-*).
- Red check: `governance-fast` — KR-08 ACT-N11 stale audit class (plan section F).
- Plan does not authorize merge while KR-08 is open unless operator grants `OD-AF-RC-KR08`.

## 051-N

- Implementation on `af2/rw-supply-ab` was ahead of train; merged `origin/arch-freeze/rw-train` → `df476721`.
- Does not patch `harness_host_runtime.py` (serial with #680 TENANT/WF per plan B.4).

## RC-2 exit (plan J)

Not met yet: `RW` milestone, full B.2 product backlog, ENT manifest closure (`043-M`), zero new unregistered reds.

## STOP (unchanged for out-of-scope)

Merge other red-governance train PRs without their own operator mandate. FCS / G2 / M3 / CLOSED.

## Post-merge (#680)

_(filled after merge commit below)_