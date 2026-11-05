# IIAC certification agent publication contract

**Role:** Certification / governance agent (not independent auditor).

## May publish

- Validation stdout bundles for manifest + `verify_iiac_*` (after auditor PASS)
- `IIAC_*_CERTIFICATION_RECORD.json` with `certification_publication_sha`
- Ladder fields: `B6_ACCEPTANCE_PUBLICATION_SHA`, `FORMAL_CLOSURE_PUBLICATION_SHA` in **dedicated publication commits**

## Must not

- Replace independent audit with checklist-only approval
- Override fail-closed preflights
- Certify governance validators authored in Session A without Session B independent review
- Activate `IN_FORCE` without `--require-in-force` exit 0, a published PASS manifest, and an authorized `IIAC_OD_ACTIVATION_*.md` decision. Session A approval is not that decision.
- Treat the ACT-01 corrective batch as certification of itself.

## Closure commit rule (EBH-7)

Do not introduce formal CLOSED claim in the same commit that first creates the pass record artifact.

## Ticket

Ref: [`governance/tickets/T-IGX-016-A.md`](../../../../governance/tickets/T-IGX-016-A.md) (Session A); ACT-01 gate correction is [`governance/tickets/T-IGX-016-B.md`](../../../../governance/tickets/T-IGX-016-B.md).
