# EVOL-X W4-F08 — operator attestation of session identities (TEMPLATE, UNSIGNED)

**Status:** `UNVERIFIED / OPERATOR_ATTESTATION_REQUIRED` until the operator completes and commits this record.
**Assurance level:** `process_enforced` (unchanged). This file is not a signature mechanism; it records a statement the operator makes in their own name.
**Prepared by the implementer session** (`local_4b071fe5-82d0-48bc-b524-146a3daf5c45`). The implementer filled in **only** the one field it can verify from the desktop session API. Every other field is left blank for the operator. Nothing here is signed, attested or approved.

Ticket: `governance/tickets/T-IGX-019-A.md` · Audit baseline: `5afaad9eadfc1655f4ef1ab59eb022960a924648` · Code under audit: `9f8b04d2b8b670077b1e78ce812bb34b680f3968`

## 1. Session identities

For each row the operator ticks exactly one of: `ID` (and writes the id) or `UNRECOVERABLE`.

| Commit | Role | Verified by the implementer | Operator statement |
|---|---|---|---|
| `23feb2bbb017b1beb63f9368b5978384093ff924` (original B4) | implementer | no (correlation only, see report §H) | `[ ] ID: ____________  [ ] UNRECOVERABLE` |
| `23feb2bbb017b1beb63f9368b5978384093ff924` | independent auditor | no | `[ ] ID: ____________  [ ] UNRECOVERABLE` |
| `23feb2bbb017b1beb63f9368b5978384093ff924` | certifier | no | `[ ] ID: ____________  [ ] UNRECOVERABLE` |
| `6cf8c5a2247465a1321c239d0559d751383daec2` (RA-B4 corrective) | implementer | no (correlation only, see report §H) | `[ ] ID: ____________  [ ] UNRECOVERABLE` |
| `9f8b04d2b8b670077b1e78ce812bb34b680f3968` (W4-01 consolidated) | implementer | **yes**: `local_4b071fe5-82d0-48bc-b524-146a3daf5c45` | `[ ] confirmed  [ ] corrected: ____________` |
| `9f8b04d2b8b670077b1e78ce812bb34b680f3968` | independent auditor (new session) | not yet exists | `ID: ____________` |
| `9f8b04d2b8b670077b1e78ce812bb34b680f3968` | certifier (third, different session) | not yet exists | `ID: ____________` |

Rule to confirm: the auditor and the certifier of `9f8b04d2b` are different sessions from each other and from the implementer.

## 2. Operator statement (to be written by the operator)

> I, ______________________, state in my own name that the session identities above are correct to the best of my knowledge, or are marked UNRECOVERABLE where I cannot establish them. I understand that this is a process-enforced statement and not a cryptographic signature.
>
> Date: ______________   Operator signature line: ______________________

## 3. Effect

* Completed with `UNRECOVERABLE` entries: W4-F08 becomes a documented, accepted limitation (not a closed defect); the auditor treats it as known.
* Not completed: W4-F08 stays `UNVERIFIED / OPERATOR_ATTESTATION_REQUIRED`.
* This record does not accept the corrective, does not close B-4, authorises no B6 step and promotes no FRZ status.
