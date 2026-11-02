# Case C1 — Evaluator diagnostic packet (blind)

**Case ID:** C1  
**Policy:** OD-EVOL-08-M1 · **TES:** `dcd7ae6e264d435994c61162e0d41aba25f71283`

## Seed (input identity)

| Field | Value |
|-------|--------|
| Canonical changed path | `intergrax/contracts/execution_identity.py` |
| Input binding | Seed file bytes @ TES (operator verifies `input_digest` pre-flight) |

## Preconditions

- You are the designated human evaluator (not the implementer / not an agent).
- You have **not** opened the P03 drill registry, oracle answer key, or honeytoken sheet.
- Operator has checked out TES and run the qualified tool on the seed path.

## Required tool output (attached by operator)

Operator attaches `.tmp/session/evol-x-e2-m09h/tool-output/C1.json` containing:

- `impact` (`evol_impact_report.v1` shape): blast radius for the seed path only
- `dossier` (`evol_change_dossier.v1` shape): sections Q1–Q6

## Your task

From the **complete packet** (cover + JSON), identify the **DIRECT** semantic-graph node for the seeded module (see `attribution_evidence` or `impact.affected`) and compute the cause identity per [`EVOL_X_M09H_CAUSE_IDENTITY.md`](../../EVOL_X_M09H_CAUSE_IDENTITY.md): `stable_digest_hex([kind, node_id])`.

Record your answer on the sealed worksheet **before** any expected answer is shown.

## Timing (human attribution)

| Event | Record |
|-------|--------|
| Packet complete — start attribution | UTC timestamp |
| Diagnosis submitted — stop attribution | UTC timestamp |
| `attribution_seconds` | Stop − start (seconds, ≥ 0) |

Tool runtime is **excluded** from attribution.

## Output format

```text
case_id: C1
attributed_cause_identities:
  - <64-char hex>
attribution_seconds: <number>
evaluator_notes: <optional>
```

## Tool failure

If JSON is missing or tool exceeded 120 s, write `TOOL_FAILURE` on the worksheet; do not invent timings. Operator decides abort vs retry.

## Ambiguous evidence

If Q2/Q3 are `NOT_ASSESSED`, base your diagnosis on assessed sections and impact graph facts present in the packet; note ambiguity in `evaluator_notes`.

## Unexpected findings

Document any anomaly (empty impact, integrity failure, stale flags) in `evaluator_notes` and on the worksheet; do not alter tool output.

## Evidence references

Operator stores: tool JSON path, scan/photo of sealed submission, UTC timestamps.

## Sealing

Sign/date the submission block; operator witnesses seal **before** opening [AUDITOR_ONLY/EVOL_X_E2_ORACLE_ANSWER_KEY.md](../AUDITOR_ONLY/EVOL_X_E2_ORACLE_ANSWER_KEY.md).
