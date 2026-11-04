# Independent review handoff (G6)

## Separation of duties

| Session type | May declare G6 PASS? |
|--------------|----------------------|
| Implementer / tester / delivery-closure | **No** |
| Cursor `bugbot` / `security-review` subagent | **No** |
| `independent-quality-auditor` in **new** operator-authorized chat | Findings only per AUDIT_PROTOCOL |

Cursor subagents do **not** provide organizational independence (EVOL-X B6).

## Implementer stop line

1. Lifecycle → `quality_gate`.
2. Checkpoint: stdout from `uv run python -m scripts.governance.igx_gov ticket` + test list from ticket **Test contract**.
3. Operator moves ticket to `ready_for_audit` and starts auditor session with `@intergrax-independent-audit` and `@docs/audit_results/AUDIT_PROTOCOL.md`.

## Ticket fields

- `ready_for_audit` — eligible for external review.
- Auditor updates ticket; **closed** only after operator sign-off.
