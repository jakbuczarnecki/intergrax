# T-IGX-022-B — Independent audit of the B-4 delta introduced by T-IGX-022-A

## Scope

`T-IGX-022-A` (merged in PR #396, commit `00cc594a9`, merge `cc86e4d80`) edited two state-pinned tests, one of them in the pinned B-4 corrective test set (`tests/unit/qualification/test_evol_x_closure_preflight_b4.py`). Under OD-3 condition 3 (operator reading: preflight script, its P2 constant, preflight tests) that edit voids the B-4 acceptance until an independent audit of the delta is done.

This batch is the audit of that delta by a separate, independent session that is NOT the implementer of `T-IGX-022-A`. The audit answers:

1. Is the delta exactly the two test functions and the ticket (no other byte of the preflight script, its P2 constant, `.gitattributes` or the preflight tests changed since the pinned B-4 implementation)?
2. Does the edited preflight test still discriminate (pending or scope-limited rows still classify as not recorded; signed rows as recorded)?
3. Does the edited attestation test still catch a premature CLOSED claim (mutation probes)?
4. Are there findings that should block relying on the B-4 corrective again?

The implementer (this ticket's author) does not set the verdict.

## Plan digest

(none - audit batch; operator instruction in chat on 2026-11-08: "Tak, wykonaj" after the proposal to open T-IGX-022-B)

## Approved

(not written by the implementer; the batch was started on the operator's explicit chat instruction of 2026-11-08)

## Test contract

```powershell
git diff --name-only 65ccf681a cc86e4d80
uv run python scripts/qualification/evol_x_closure_publication_preflight.py --gate formal-closure
```

The auditor runs only the targeted tests and mutation probes in a scratch clone (one pytest at a time). The full 520-test run of 022-A is not repeated.

## Out of scope

Editing the preflight script or constants, any material governance path, IIAC activation, FRZ rows, OD records, the three `roadmap_replay_x` failures, `ent-audit-fast`, a new IIAC audit manifest and certifier disposition (a formal re-certification of B-4 is a separate operator-ordered step), merging.

## Audit result

Independent audit completed 2026-11-08 by a separate session: `Verdict: PASS_WITH_FINDINGS` - see [`T-IGX-022-B_AUDIT_REPORT.md`](./T-IGX-022-B_AUDIT_REPORT.md). Preflight script, P2 pin, `.gitattributes` and the six material governance paths are unchanged since the accepted B-4 implementation `fdb1e7a62`; the preflight test file differs by one hunk (the register assertion); 12/12 mutants of the register test and 14/16 of the attestation test are detected. Four LOW findings (F-1 ladder SHA checked by shape only, covered by preflight B-08/B-10; F-2 pre-existing substring check on OD-EVOL-14; F-3 governance SHA not compared with the ladder in the test, covered by B-02; F-4 session independence is process_enforced). Session independence is not provable from git. Whether the certifier must re-disposition B-4 formally is the operator's decision and is not done here. The implementer does not set the lifecycle beyond `ready_for_audit`.

## Lifecycle state

ready_for_audit
