# Case C3 — Evaluator diagnostic packet (blind)

**Case ID:** C3  
**Policy:** OD-EVOL-08-M1 · **TES:** `dcd7ae6e264d435994c61162e0d41aba25f71283`

## Seed (input identity)

| Field | Value |
|-------|--------|
| Canonical changed path | `testing_support/change_safety/contracts/enums.py` |
| Input binding | Seed file bytes @ TES (operator verifies `input_digest` pre-flight) |

## Preconditions

- Designated human evaluator; no registry / oracle / honeytokens during diagnosis.
- Operator @ TES; tool output for C3 attached.

## Attached tool output

`.tmp/session/evol-x-e2-m09h/tool-output/C3.json` — `impact` + `dossier` (Q1–Q6).

## Task

Attribute root-cause identity (64-char hex list) from packet only; sealed submission before oracle.

## Timing

Attribution window per OD-EVOL-08-M1 (packet complete → submission).

## Submission format

Same as C1 with `case_id: C3`.

## Failures / ambiguity / surprises / sealing

Same as C1.
