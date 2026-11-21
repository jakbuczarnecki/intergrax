# EQO-W3-WB10-AUD — session E packet (wave-3)

**Packet status:** `OPEN`
**Verdict:** _reserved — independent session E only; implementer records no PASS/FAIL._
**Land SHA (post PR #690):** `8631b1e8459e096b74d1c80247d3e432b44fff30`
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**DOC pin:** `8cee3c5a913f234e8fb65053b6ab4f34ed78249b`

## Evidence refs

| Artefact | Path |
|----------|------|
| RUN_MANIFEST | `eqo_closure/runs/wb10/RUN_MANIFEST.json` |
| AUD prep | `eqo_closure/runs/wb10/AUD_PREP_CHECKLIST.md` |
| DIS/EVD | `eqo_closure/runs/wb10/TRACE_STATE_REPLAY_EVIDENCE.md` |
| Pytest log | `eqo_closure/runs/wb10/pytest_wb10_exe.log` |

## EXE summary

| Field | Value |
|-------|-------|
| pytest_exit_code | `1` |
| pytest_summary | 6 failed, 66 passed, 1 skipped in 80.56s (0:01:20) |
| executed_utc | 2026-11-21T06:24:57Z |
| blockers | PYTEST_SLICE_NON_ZERO_EXIT |

## Finding IDs in scope

- `EQOA-A2-019`
- `EQOA-A2-030`
- `EQOA-A5-012`

## Checklist answers (AUD prep → packet)

| Item | Answer |
|------|--------|
| TRAIN_PIN matches manifest | YES |
| Pytest log on disk | YES |
| Independent session (not EXE agent) | PENDING — requires separate session E |
| Matrix AUD column updated | DEFERRED to integrator (wave-3 matrix §) |

## Open items for session E

- EXE slice **BLOCKED** (non-zero pytest); AUD may scope to DIS-only or wait for product fix.

**Ref:** `governance/tickets/T-IGX-046-F.md`
