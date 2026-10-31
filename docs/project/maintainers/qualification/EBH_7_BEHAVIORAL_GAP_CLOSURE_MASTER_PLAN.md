# EBH-7 — Behavioral gap closure master plan (operator mirror)

| Field | Value |
| --- | --- |
| Source plan | Cursor plan `ebh-7_gap_closure_program` (qualification-corrected) — **do not edit plan file** |
| Baseline | `development` @ gap-closure implementation wave |
| SSOT disposition | `tests/qualification/ebh_7/_semantic_disposition_registry.py` |
| OD-EBH7-01 | **APPROVED** @ `81679e672` — operator record [`EBH_7_OD_EBH7_01_ENTRY_READINESS_RECORD.md`](EBH_7_OD_EBH7_01_ENTRY_READINESS_RECORD.md) |

## W0 — operator gate

- **W0-RC04-D:** route **integrate-test-first** (ST-01 adversarial on `RuntimeToolInvoker` / `ToolRuntime.invoke` before optional ADR).
- Agent may materialize tests, disposition updates, and dossier prep — not OD-EBH7-01 PASS, not B6, not EBH-7 CLOSED.

## Waves (executable)

| Wave | Task | Exit |
| --- | --- | --- |
| W1 | ST-01..04 integrated proofs + minimal product delta if ST-01 fails | TEN-03/04/05/13 disposition honest |
| W2 | RPL Package C v2 dual-register @ `resolve_from_profile` | 51 semantic QUALIFIED |
| W3 | ST-05..08 platform/cross_domain integrated tests | 60/60 EXACT |
| W4 | 8 cert gates, ECAP C4/C5, PROHIBITED-* | 0 B6 xfail |
| W5–W6 | Protocol 03 B3–B5 dossier; external B6; OD-EBH7-08 | CLOSED SCOPED (operator) |

## Evidence capture

Session logs under `.tmp/session/ebh-7-gap-closure/<wave>/` only (see intergrax-session-artifacts rule).
