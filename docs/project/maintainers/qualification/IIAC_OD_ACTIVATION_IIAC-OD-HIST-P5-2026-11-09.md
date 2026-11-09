# IIAC operator decision — HIST P5 shared re-activation (product batch)

Decision ID: IIAC-OD-HIST-P5-2026-11-09
Policy version: IIAC-2026-11-05.1
Decision scope: IIAC policy activation
Decision status: AUTHORIZED
Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e

Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-09T07:00:00+01:00

## Provenance

### Operator authorization (this session)

Operator message (2026-11-09, chat): **„Udzielam zgody, wykonaj”** — authorization to publish this activation record and execute P5-SIGN-ACT on `development` @ product merge SHA `0946b11576d2ee617321fb099ed794f8b21bf5ba`.

P2-WAIT sequencing consent is recorded separately in `docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md` on `hist-cert/final-docs`; that delegation does **not** substitute this `operator @` activation line.

### Scope

Re-bind IN_FORCE policy `IIAC-2026-11-05.1` after the HIST C1–C4 product merge (T-IGX-030-B/C, 032-A, 033-A). Session B governance certification evidence (`IIAC-GOV-1`) is unchanged; `audit_execution_sha` advances to the post-merge audit tree. This decision does not close EVOL-X, ENT-AUDIT-X, ARCH-FREEZE, or the historical certification campaign.

### Evidence references

| Item | Value |
|---|---|
| Governance implementation SHA | `2ae1014116d353bc7742985e62f2bccc2bfac798` |
| ACT-01 corrective SHA | `2a6545f1d551814a0ba8896be79576c580353771` |
| Prior activation phase-1 SHA | `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` |
| HIST product merge SHA | `3c91694c645ff0d97d104adf0c5a564b9a4c5b3c` |
| HIST P5 audit / merge HEAD | `0946b11576d2ee617321fb099ed794f8b21bf5ba` |
| P5 pre-activation IIAC evidence | `docs/project/maintainers/qualification/historical_certification_campaign/iiac_evidence/hist-p5-2026-11-09/` |
| Active certification manifest | `docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/certification/iiac_audit_session_manifest.IIAC-GOV-1.json` |

### Decision limitations

- Technical gates (`verify_iiac_policy_activation.py --require-in-force`) must pass after two-phase publication.
- ENT-AUDIT-X counter-signature on shared re-activation remains absent unless separately recorded.
- Prior operator decision `IIAC-OD-01-2026-11-05` remains in git history; this decision supersedes only the active binding SHAs.
