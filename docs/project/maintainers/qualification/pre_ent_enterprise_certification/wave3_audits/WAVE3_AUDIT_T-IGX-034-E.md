# WAVE3 independent audit — T-IGX-034-E (ROADMAP-REPLAY-X)

| Field | Value |
|-------|-------|
| Session slug | `replay` |
| Worktree | `w3-replay` @ `D:/Projekty/_wt/intergrax/w3-replay` |
| Role | `independent-quality-auditor` |
| AUDIT_HEAD | `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` |
| Replay UTC | `2026-11-10T14:37:54Z` |
| Ticket | [`governance/tickets/T-IGX-034-E.md`](../../../../../../governance/tickets/T-IGX-034-E.md) |
| Ref (campaign) | `governance/tickets/T-IGX-034-A.md` |

## Blocking pytest

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q
```

**Summary:** 16 passed in 8.17s, **exit 0** (auditor replay `w3-replay` @ AUDIT_HEAD).  
**Log:** [`.tmp/session/pre-ent-wave3-audit/replay/pytest.log`](../../../../../../.tmp/session/pre-ent-wave3-audit/replay/pytest.log) (`# EXIT_CODE: 0`)

## Evidence reviewed

- [`PRE_ENT_REPLAY_L3_PROD_DELTA.md`](../PRE_ENT_REPLAY_L3_PROD_DELTA.md) — scoped prod-Δ SSOT and batch dispositions.
- [`WAVE1_REPLAY_STATUS.md`](../WAVE1_REPLAY_STATUS.md) — reconciliation narrative @ prior qual HEAD.

## Scope findings (ticket only)

- Mechanical exit gate module **green** @ AUDIT_HEAD (historical-evidence and VBN replay counters per ticket intent).
- L3 prod-Δ disposition SSOT documents batches including BND-03B, 030-B/C, 032-A without FRZ promotion claims.
- **Governance lag (observation):** ticket lifecycle `in_progress`; Approved line still `(pending — PRE-ENT campaign)` — does not invalidate mechanical gate evidence @ AUDIT_HEAD.
- **Not claimed:** FRZ PASS, roadmap §5 publication, G6/IIAC closure.

## Verdict (T-IGX-034-E scope only)

**PASS**
