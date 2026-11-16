# IIAC operator signing delegation record (template)

**Status of the governing rule:** `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md` is IN FORCE as operator-adopted policy text (revised 2026-11-08). A delegation record is a transcription of a commission given in chat, not a condition of validity (S-1 struck).

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
- Single use: the delegation covers only the Decision ID and the SHA-256 above. `[ ] confirmed`

## Operator signature line

Optional under Amendment 01 (S-1 struck). Required only by validators that read it literally (currently the OD-ENT-01 baseline promotion, see Amendment 01 section 3). Keep the line shape below; do not fill it unless a validator requires it.

`Approved delegation ________________: operator @ ________________`
