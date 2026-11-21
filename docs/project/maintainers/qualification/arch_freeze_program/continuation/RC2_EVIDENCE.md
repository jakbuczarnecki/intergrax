# RC-2 evidence dossier (in progress)

Worktree `D:/Projekty/_wt/intergrax/af-rc2`, branch `af2/rc2-continuation`. Base includes RC-1 checkpoint `36c470249`.

No merge of PR #680 or #682 without `OD-AF-RC-KR08 = TAK`. No G2 declaration.

## Train review pulls (no merge)

| PR | Branch | Head | Train base | governance-fast | Action |
| --- | --- | --- | --- | --- | --- |
| #680 | `af2/pt-fc1` | `a6999e97` | merged into branch | **FAILURE** (KR-08) | digest review only; **merge blocked** on KR-08 |
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

## STOP

Merge #680 with red governance-fast without operator KR-08 answer. FCS / G2 / M3 / CLOSED.