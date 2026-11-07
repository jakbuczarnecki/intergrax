# IIAC operator signing delegation record (template)

**Status of the governing rule:** `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md` is `PROPOSED - NOT IN FORCE`. This template creates no authority and must not be filled in or used until that amendment is activated by the operator. Safeguard S-1 (one delegation record per delegation) is agent-proposed and may be struck by the operator.

**Delegation ID:** `________________`
**Policy version:** `________________`
**Date (ISO-8601):** `________________`

## Delegated act

- Stage: `________________`
- Decision ID (decision to be signed): `________________`
- Record path (repo-relative path of the record to be signed): `________________`
- SHA-256 of the text to be signed: `________________`
- Delegated act: write the delegated signature line `Approved <Decision ID>: operator-delegate @ <ISO-8601 timestamp>` plus the line `Delegation: <this record path>@<commit sha>` in the record above, and nothing else.

## Designated signing session

- Designated session id: `________________`
- Designated agent id: `________________`
- Dedication statement (operator): the designated session is a NEW session, dedicated solely to the certification task named above, it is NOT a continuation of any other session and it is NOT an implementation session. `[ ] confirmed by the operator`
- Dedication statement (designated session, to be recorded in its first message): the same four points, restated by the session itself. `[ ] recorded; message reference: ________________`
- Implementer session id(s) of this stage (must differ from the designated session id): `________________`

## Limits

- Expires (ISO-8601): `________________`
- Not delegable (S-3): this delegation record, IIAC policy activation, any amendment of the signing rule. `[ ] the decision named above is none of these`
- Single use: the delegation covers only the Decision ID and the SHA-256 above. `[ ] confirmed`

## Operator signature line

To be written by the operator personally, once per delegation. Do not pre-fill. No agent writes this line.

`Approved delegation ________________: operator @ ________________`
