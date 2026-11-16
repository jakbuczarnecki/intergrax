# T-IGX-023-A — EVOL-X W7-06: formal ENT-AUDIT-X handoff pin and D10 digest-chain attestation

## Scope

EVOL-X is formally closed (`FORMAL_CLOSURE_PUBLICATION_SHA` = `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca`, merged in PR #395). W7-06 of `evol_x/EVOL_X_B6_TO_E4_EXECUTION_SEQUENCE.md` is the next step:

1. Finalize `docs/project/maintainers/qualification/evol_x/EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md` as a descendant of the closure SHA: formal closure, B6 and operator decisions are no longer "unset/OPEN"; formal checklist ticked with references.
2. D10 digest chain attested by an independent verifier session (the "B6 party"): report `docs/project/maintainers/qualification/evol_x/EVOL_X_D10_DIGEST_CHAIN_ATTESTATION.md`.
3. Two-phase pin (non-self-referential): phase 1 commit publishes the record and the attestation; phase 2 commit binds `W7_06_HANDOFF_PUBLICATION_SHA` (new ladder row) and `w7_06_handoff_publication_sha` (execution status) to the phase-1 commit.

No ENT-AUDIT-X code is implemented in this wave.

## Constraints (must be honoured)

- No change to the preflight script, its tests, constants, the IIAC material governance paths or the activation record; `verify_iiac_policy_activation --require-in-force` stays IN_FORCE and `evol_x_closure_publication_preflight.py --gate formal-closure` stays PASS.
- The handoff record must not claim ENT-AUDIT-X CLOSED, ARCH-FREEZE, PROD-Q, or production rollback hardening; exclusions of section 8 stay.
- The operator signatures behind the closure were written by operator-designated signing sessions (Amendment 01, process_enforced); the handoff record states this.
- The PR must be merged with a merge commit (not squash or rebase), otherwise the phase-1 SHA pin breaks.

## Plan digest

(none - docs-only step; operator instruction in chat on 2026-11-08: "Wykonaj kolejne zadanie", option "W7-06: przekazanie do ENT-AUDIT-X")

## Approved

(not written by the implementer; the batch was started on the operator's explicit chat instruction of 2026-11-08)

## Test contract

```powershell
uv run python scripts/qualification/evol_x_closure_publication_preflight.py --gate formal-closure
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
git diff --name-only origin/development
```

The last command lists only docs and this ticket. Test suites are not affected (no code or test change); one pytest at a time if a run is needed.

## Out of scope

ENT-AUDIT-X implementation, ARCH-FREEZE, PROD-Q, Docker, vendor, production configuration, edits of the three closure-state tests, IIAC Amendment 01 hardening (021-B/C), merging the PR.

## Lifecycle state

quality_gate
