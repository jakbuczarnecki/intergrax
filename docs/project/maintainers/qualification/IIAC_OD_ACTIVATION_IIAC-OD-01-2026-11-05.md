# IIAC operator decision — OD-01 policy activation

Decision ID: IIAC-OD-01-2026-11-05
Policy version: IIAC-2026-11-05.1
Decision scope: IIAC policy activation
Decision status: AUTHORIZED
Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e

Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-05T18:14:38+01:00

## Provenance

### Verbatim operator authorization (supplied for this session)

> Jako operator INTEGRAX jednoznacznie autoryzuję aktywację polityki IIAC-2026-11-05.1 na podstawie certyfikacji IIAC-GOV-1. Przygotuj finalną instrukcję dla odrębnej sesji Cursor, która zapisze autentyczną decyzję OD-01, przeprowadzi dwufazową publikację oraz zweryfikuje --require-in-force. Nie fabrykuj podpisu ani czasu; decyzję powiąż z moją obecną autoryzacją.

### Source description

Operator statement relayed from the operator’s ChatGPT conversation into the Cursor OD-01 activation session prompt. This repository file is the durable decision record; no separate ChatGPT transcript is stored in git.

### Recording / publication timestamp

`2026-11-05T18:14:38+01:00` (Europe/Warsaw, `+01:00`) — time this decision file was written to the repository. It is **not** claimed to be the original ChatGPT message timestamp.

### Scope

Internal independent agent certification authority for pre-production stages `EVOL-X` and `ENT-AUDIT-X` under policy `IIAC-2026-11-05.1`. This decision does not close `EVOL-X`, `ENT-AUDIT-X`, or `ARCH-FREEZE`.

### Evidence references

| Item | Value |
|---|---|
| Governance implementation SHA | `2ae1014116d353bc7742985e62f2bccc2bfac798` |
| ACT-01 corrective SHA | `2a6545f1d551814a0ba8896be79576c580353771` |
| ACT-01 audit execution SHA | `ae467c8c4dd9c87521c1c60a7e5e75224d3214a0` |
| ACT-01 report digest (accepted audit) | `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e` |
| F01–F03 re-audit digest | `62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde` |
| Certification publication SHA | `a5fb6cb51ecd982284fd53ebc41e284e18b5523d` |
| Active certification manifest | `docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/certification/iiac_audit_session_manifest.IIAC-GOV-1.json` |

### Decision limitations

- Technical activation gates (`verify_iiac_policy_activation.py --require-in-force`) must still pass after two-phase publication.
- Session A and ACT-01 implementation approvals are not substitutes for this decision.
- Historical Session B `CONDITIONAL` evidence remains preserved; it is not rewritten.
