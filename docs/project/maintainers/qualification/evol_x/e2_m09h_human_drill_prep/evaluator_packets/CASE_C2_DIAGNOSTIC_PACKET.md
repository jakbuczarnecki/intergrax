# Case C2 — Evaluator diagnostic packet (blind)

**Case ID:** C2  
**Policy:** OD-EVOL-08-M1 · **TES:** `dcd7ae6e264d435994c61162e0d41aba25f71283`

## Seed (input identity)

| Field | Value |
|-------|--------|
| Canonical changed path | `testing_support/change_safety/graph/impact.py` |
| Input binding | Seed file bytes @ TES (operator verifies `input_digest` pre-flight) |

## Preconditions

- Designated human evaluator; no access to drill registry / oracle / honeytokens during diagnosis.
- Operator @ TES; tool output for C2 attached.

## Attached tool output

`.tmp/session/evol-x-e2-m09h/tool-output/C2.json` — `impact` + `dossier` (Q1–Q6).

## Task

Attribute the root-cause identity (64-char hex SHA-256 list) for the seeded path from the packet alone. Submit on sealed worksheet before oracle disclosure.

## Timing

Attribution starts when this packet + JSON are complete; stops at diagnosis submission. Exclude tool time.

## Submission format

Same structure as C1 with `case_id: C2`.

## Failures / ambiguity / surprises

Same rules as C1 cover sheet.

## Sealing

Sealed submission before operator compares to auditor oracle.
