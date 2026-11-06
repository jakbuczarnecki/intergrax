# IIAC-GOV-1 — governance certification report

**Role:** `certification-governance-agent`  
**Policy:** `IIAC-2026-11-05.1`  
**Certification replay SHA:** `ae467c8c4dd9c87521c1c60a7e5e75224d3214a0`  
**Corrective implementation SHA:** `2a6545f1d551814a0ba8896be79576c580353771`  
**Policy status after this publication:** `NOT_IN_FORCE`

## Verdict

`PASS — IIAC GOVERNANCE CERTIFICATION`

Independent acceptance verified for:

| Audit | Verdict | Report digest |
|---|---|---|
| F01–F03 re-audit | `PASS` | `62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde` |
| ACT-01 independent re-audit | `PASS — ACT-01 INDEPENDENTLY VERIFIED` | `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e` |

Historical Session B evidence (`CONDITIONAL` / `PARTIAL`) is preserved under `historical_session_b/` and was not rewritten.

## Certification checks

- ACT-01 dossier bytes published under `iiac_evidence/IIAC-GOV-1/certification/` (not `.tmp/` alone).
- Reaudit manifest and role separation companions: PASS @ replay SHA.
- Canonical activation gate: `--expect-not-in-force` exit 0; `--require-in-force` exit 1 (`ACT-N01`).
- G0 completeness: PASS @ `afe9e8abeabdbab410bf562ebed6f4a309a91057`.
- No `IIAC_OD_ACTIVATION_*.md` with `Decision status: AUTHORIZED`.
- EVOL-X / ENT-AUDIT-X / ARCH-FREEZE closure preflights remain fail-closed (unchanged).

## Activation boundary

This report certifies governance eligibility. It does **not** activate the policy. Operator authorization remains the exact gate documented in `IIAC_ACT01_OPERATOR_ACTIVATION_PACKET.md`.
