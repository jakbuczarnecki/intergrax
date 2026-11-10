# WAVE4 certifier — T-IGX-034-E (ROADMAP-REPLAY-X mechanical exit gates)

| Field | Value |
|-------|-------|
| Session slug | `replay` |
| Role | `certification-governance-agent` |
| AUDIT_HEAD | `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` |
| CERT_DOCS_HEAD (handoff SSOT read) | `d00df592037404cd6c2b1bc95c00757544d551d8` |
| Worktree | `w4-replay` @ detached `1e83ade3` |
| Replay UTC | `2026-11-10T15:07:05Z` |
| Ticket | [`governance/tickets/T-IGX-034-E.md`](../../../../../../governance/tickets/T-IGX-034-E.md) |
| Ref (campaign) | `governance/tickets/T-IGX-034-A.md` |
| Wave 3 auditor | [`WAVE3_AUDIT_T-IGX-034-E.md`](../wave3_audits/WAVE3_AUDIT_T-IGX-034-E.md) — **PASS** |

## Blocking pytest (certifier replay)

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q
```

**Summary:** 16 passed in 18.00s, **exit 0** (parity with Wave 3 auditor: 16 passed @ same `AUDIT_HEAD`; longer wall time includes fresh `.venv` in certifier worktree).  
**Log:** [`.tmp/session/pre-ent-wave4-cert/replay/pytest.log`](../../../../../../.tmp/session/pre-ent-wave4-cert/replay/pytest.log)

## Evidence reviewed

- Wave 3 independent audit report and consolidated register row for `replay`.
- [`PRE_ENT_REPLAY_L3_PROD_DELTA.md`](../PRE_ENT_REPLAY_L3_PROD_DELTA.md) — scoped prod-Δ SSOT and batch dispositions (incl. BND-03B, 030-B/C, 032-A).
- [`WAVE1_REPLAY_STATUS.md`](../WAVE1_REPLAY_STATUS.md) — reconciliation narrative @ prior qual HEAD.
- Product @ `AUDIT_HEAD`: qualification SSOT modules `_pre_ent_l3_prod_delta.py`, `_prod_delta_disposition.py`, mechanical gate tests under `tests/qualification/roadmap_replay_x/`.

## Scope findings (ticket only)

- **Mechanical exit gates:** all blocking qualification tests green @ certifier replay; `historical_evidence_invalidated_by_later_changes` and VBN replay counters consistent with ticket intent (scoped prod-Δ, unscoped delta empty @ disposition SSOT).
- **L3 prod-Δ honesty:** disposition narrative matches Wave 3 auditor findings — batches documented without FRZ promotion or parent closure language.
- **Wave 3 → Wave 4:** auditor minimum pytest **replayed successfully**; no regression vs Wave 3 register (16 passed, exit 0).
- **Governance lag (observation):** ticket lifecycle `in_progress`; Approved line `(pending — PRE-ENT campaign)` — no material contradiction with mechanical evidence @ `AUDIT_HEAD`; certifier does not advance lifecycle or FRZ.

## Not claimed

- G6 / IIAC / PRE-ENT campaign **CLOSED**
- FRZ promotion or roadmap §5 publication
- `ALL_PRE_ENT_PREDECESSORS_INDEPENDENTLY_CERTIFIED`
- Ticket lifecycle `closed` without operator sign-off

## Certifier disposition (T-IGX-034-E scope only)

**ACCEPT SCOPED**

Scoped to ROADMAP-REPLAY-X mechanical exit gates and L3 prod-Δ disposition honesty @ `AUDIT_HEAD`.
