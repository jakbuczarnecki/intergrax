# IIAC operator delegation — EVOL-X B-4 original independent audit evidence publication

Decision ID: OD-EVOLX-B4-PUB-01
Policy version: IIAC-2026-11-05.1
Decision scope: EVOL-X B-4 original independent audit evidence publication
Decision status: AUTHORIZED
Decision kind: publication delegation (not a certification, not a PASS, not an IIAC activation)

Approved EVOL-X B-4 evidence publication delegation OD-EVOLX-B4-PUB-01: operator @ 2026-11-06T07:49:13+01:00

> The timestamp above is the time this record was written to the repository, taken from the publication session's system clock. It is not the time of the operator's original message and no signature or authenticated transcript reference is claimed.

## Provenance

### Source of authorization

1. **Mission brief.** The publication session received the operator's mission "EVOL-X B-4 ORIGINAL AUDIT EVIDENCE PUBLICATION" as a pasted message. The brief states that the authorization was supplied through ChatGPT and describes the scope below as "proposed". The ChatGPT conversation is not stored in git and is not reproduced here.
2. **Express approval.** After the publication session stopped at the approval gate and showed the draft scope, the operator replied in the Claude Code chat with the words `Approve scope`. That reply is the express approval of the scope below.

### Role-registry limitation (not waived, recorded)

`governance/agent-control-plane/role_registry.yaml` (v1.0) has no `certification-governance-agent` role, and no existing delegation procedure was found. This record is the operator's direct delegation for this one publication. It does not add a role, change the registry, or establish a general delegation mechanism.

## Authorized scope (and nothing more)

1. Publish, byte-preserved, the 10 artifacts indexed by `iiac_evidence/EVOL-X/b4_reaudit/EVIDENCE_INDEX.json`, plus that index.
2. This delegation record.
3. One publication commit pushed to `origin/development` without force, then a separate non-self-referential SHA-binding commit.

## Prohibitions in force

- No change to the auditor's findings, dossier, manifest draft or unsigned handoff; the verdict stays `EVOL-X B-4 — PARTIAL / EXACT BLOCKERS`.
- No PASS promotion; RA-B4-01..04 stay open; no B6 acceptance; no FRZ-EVOL promotion.
- No OD-EVOL-12 or OD-EVOLX-08 decision; no roadmap CLOSED.
- No product or corrective implementation code change.
- `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` is untouched and unstaged.
- The publication session does not self-certify and does not substitute its identity for the auditor's.

## Session separation

| Role | Session id | Source |
|---|---|---|
| Publication session | `9e3b9293-fa2a-4176-ab50-1f11470bebb8` | Claude Code session metadata (scratchpad path, project transcript name) |
| Independent auditor | `cf31a7e9-08ee-47f4-84ca-5b420d9a377d` | Recorded in the auditor's manifest draft as `IIAC-C-EVOL-X-B4-REAUDIT-2026-11-05__claude-code-cf31a7e9-08ee-47f4-84ca-5b420d9a377d`; a local transcript with that id exists |
| T-IGX-017-A implementer | UNKNOWN | Not present in the repository; not invented. The auditor's handoff record leaves operator attestation open and this record does not complete it |

The three identities are not shown to be equal. Assurance level remains `process_enforced`; there is no cryptographic session proof.

## Evidence verification (before publication)

All 10 indexed artifacts match `EVIDENCE_INDEX.json` (sha256 and byte count) in the working tree and in the custody archive `D:\Projekty\_intergrax_evidence_archive\EVOL-X_b4_reaudit_20261106\b4_reaudit`. Digests are recomputed from staged blobs at commit time; the result is in the publication commit message.

## Remaining governance limitations

- The auditor handoff record is still an unsigned operator attestation; `implementation_session_id` is unattested.
- The manifest draft keeps `certifier_session_id`, `evidence_publication_sha` and `certification_publication_sha` as `PENDING_CERTIFIER`; this publication does not fill them.
- This publication is not a certification and cannot close B-4.
