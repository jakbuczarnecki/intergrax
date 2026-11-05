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
- Activate `IN_FORCE` without operator activation line + Session B digest pinned in [`IIAC_POLICY_ACTIVATION_RECORD.json`](./IIAC_POLICY_ACTIVATION_RECORD.json)

## Closure commit rule (EBH-7)

Do not introduce formal CLOSED claim in the same commit that first creates the pass record artifact.

## Ticket

Ref: [`governance/tickets/T-IGX-016-A.md`](../../../../governance/tickets/T-IGX-016-A.md) (Session A); stage certs may use T-IGX-016-B/C rows when scaffolded.
