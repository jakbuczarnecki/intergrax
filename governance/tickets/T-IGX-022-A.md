# T-IGX-022-A — Update the two state-pinned EVOL-X tests after formal closure

## Scope

After the EVOL-X formal closure (PR #395, merge `65ccf681a`; OD-EVOL-12 and OD-EVOLX-08 signed, `evol_x_closed: true`) two unit tests still assert the pre-closure repository state and fail by design. CI check `evol-x-b4-preflight` is red on `development` because of the first one.

1. `tests/unit/qualification/test_evol_x_closure_preflight_b4.py::test_ra_b4_03_real_register_rows_still_classify_correctly` asserts that the real register rows OD-EVOL-12 and OD-EVOLX-08 are NOT recorded. Change: they must now classify as recorded decisions (the classifier behaviour under test is unchanged; scope-limited approvals OD-EVOL-14/15/17 must still not classify as closing decisions).
2. `tests/unit/qualification/test_evol_x_b6_attestation_contract.py::test_published_records_keep_formal_b6_pending` asserts `b6_acceptance_publication_sha is None`, `evol_x_closed is False` and PENDING register rows. Change: it keeps the ECAP-amendment assertions (`PENDING_REQUIRED_APPROVALS`, `NOT_IN_FORCE`) and now asserts that a published CLOSED state is evidence-bound: `evol_x_closed` implies a 40-hex `b6_acceptance_publication_sha`, a 40-hex `FORMAL_CLOSURE_PUBLICATION_SHA` row in the SHA ladder, and OPERATOR APPROVED register rows for OD-EVOL-12 and OD-EVOLX-08. The test name is kept because `EVOL_X_FINAL_ACCEPTANCE_EVIDENCE_MATRIX.md` cites it.

No production code, no preflight script, no constant, no document is changed. Only the two test functions.

## Constraints (must be honoured)

- The preflight tests belong to the pinned B-4 corrective (OD-3 condition 3, operator reading: preflight script, its P2 constant and the preflight tests). Editing `test_evol_x_closure_preflight_b4.py` therefore voids the B-4 acceptance until an independent audit of this delta is done. That audit (T-IGX-022-B) is NOT performed by this batch and the implementer must not self-certify it.
- `scripts/qualification/evol_x_closure_publication_preflight.py` and the IIAC material governance paths are not touched; `verify_iiac_policy_activation --require-in-force` must stay IN_FORCE.
- `development` is protected: the change goes through a pull request; the implementer does not merge.

## Plan digest

(none - small bounded test-only change; operator instruction in chat on 2026-11-07: "Zaloz ticket na naprawe dwoch testow zaleznych od stanu, a nastepnie od razu zrob test i wprowadz poprawki")

## Approved

(not written by the implementer. The batch was started on the operator's explicit chat instruction of 2026-11-07; the operator may add the `Approved:` line.)

## Test contract

```powershell
uv run pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b6_attestation_contract.py -q
uv run python scripts/qualification/evol_x_closure_publication_preflight.py --gate formal-closure
git diff --name-only origin/development
```

The last command must list only `governance/tickets/T-IGX-022-A.md` and the two test files. One pytest at a time.

## Follow-ups (not part of this batch)

- `T-IGX-022-B`: independent audit of the B-4 delta (this test change), separate session, never the implementer.

## Out of scope

Preflight script, P2 constant, IIAC protocol and verifiers, FRZ rows, OD records, `roadmap_replay_x` failures (production delta since DUP-X), `ent-audit-fast` failure, Docker, vendor, production configuration, merging the PR.

## Lifecycle state

quality_gate

## Checkpoint

Serial pytest on both files: 520 passed, 2 skipped (the skips are pre-existing canonical pre-closure tests). Preflight Gate A and Gate B PASS, IIAC IN_FORCE. Independent audit of the B-4 delta (T-IGX-022-B) not performed.
