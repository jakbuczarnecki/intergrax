# OD-SFX-R2-GOV-01 — F-GOV-01 publication sequencing disposition (T-IGX-070-C)

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-R2-GOV-01 |
| **Ticket** | `T-IGX-070-C` |
| **Campaign** | `IGX-SFX-R2-CANONICAL-ADOPTION-001` |
| **Finding** | F-GOV-01 |
| **Disposition** | **ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING** |
| **Signature status** | **RECORDED** — operator commission relayed; not signed by Composer/Cursor as operator |
| **Related OD** | [`OD-SFX-R2-CANONICAL-ADOPTION.md`](OD-SFX-R2-CANONICAL-ADOPTION.md) (unchanged) |

## Operator decision source

The Integrax operator accepted this disposition in **ChatGPT on 2026-11-23** (message fragment `ACCEPT_SCOPED_WITH_GOVERNANCE_FINDIN`, completed per the operator’s same-thread confirmation as **`ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING`**). The acceptance was relayed to the closure implementer in the Cursor mission **T-IGX-070-C / FINAL GOVERNANCE DISPOSITION & TICKET CLOSURE** (2026-11-23).

This file **records** that commission. It does **not** claim in-person operator cryptographic signing in this repository unless a separate delegated signing session adds an `Approved:` line under an existing delegation path.

## Question

Does the operator accept **documentation-only** SFX R2 Option B overlay adoption on `development` **while preserving on record** that overlay commits reached `development` before independent G6, with `G6_GOVERNANCE_VERDICT = NONCOMPLIANT`, and **without** retroactive approval of pre-G6 publication or any implementation authorization?

## Disposition (accepted scope)

1. **ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING** — documentation adoption stands; F-GOV-01 remains a **historical procedure violation**, not evidence that publication sequencing was compliant.
2. **`G6_GOVERNANCE_VERDICT = NONCOMPLIANT`** — preserved; not upgraded to COMPLIANT.
3. **No retroactive** approval of publication before G6.
4. **G6 content / delta:** `G6_CONTENT_VERDICT = PASS_WITH_FINDINGS`; `G6_DELTA_VERDICT = ACCEPT` (corrective docs @ `7c16d9b5488234475e80a2e758158decca5e5ac5`).
5. **Conditional** formal ticket closure only after lifecycle, gates, and independent closeout package review — per operator closure mission.
6. **`IMPLEMENTATION_AUTHORIZED = NO`**; **`ENTERPRISE_PRODUCT_CERTIFIED = NO`**.
7. **No grant:** autonomous multi-submit, production deployment, ARCH-FREEZE clearance, or GATE-SFX-ENTRY bypass.

## Formal finding state

```text
F-GOV-01 = OPERATOR_ACCEPTED_WITH_FINDING
```

This disposes the **historical** governance gap; it is **not** proof that pre-G6 publication followed procedure.

## Evidence pins

| Event | SHA / identity |
| --- | --- |
| G2 plan approval | `f33501780a62ae82a718834b6521adc856ec9bfb` |
| G3 operator adoption (OD) | `81c51263ea672e0f4d886dd0dc85bcf646849108` |
| First overlay package on `development` | `1cd4c621ad7eee35526fb24eaedbfdc265a20404` |
| G6 audit candidate | `9e4eb2d07df8621378830efd26e6c9db6a0f337a` |
| G6 audit host | `local_039b59d4-5f1c-4360-8fd7-28315b244478` |
| G6 corrective publication (`development`) | `7c16d9b5488234475e80a2e758158decca5e5ac5` |

## G6 chronology (post-publication)

| Stage | Result |
| --- | --- |
| Initial G6 (content) | `PASS_WITH_FINDINGS` @ `9e4eb2d07` |
| Initial G6 (governance) | `NONCOMPLIANT` (F-GOV-01) |
| Corrective delta | `ACCEPT` @ `ac2cb4ea` → published `7c16d9b54` |

## Explicit non-grants

| State | Value |
| --- | --- |
| `IMPLEMENTATION_AUTHORIZED` | **NO** |
| `ENTERPRISE_PRODUCT_CERTIFIED` | **NO** |
| `G6_GOVERNANCE_VERDICT` | **NONCOMPLIANT** (preserved) |
| Pre-G6 publication retro-approval | **NOT GRANTED** |
| SEF CON-N-01 / delegated multi-submit | **NOT AUTHORIZED** |