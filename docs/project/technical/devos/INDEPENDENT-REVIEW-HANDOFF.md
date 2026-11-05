# Independent review handoff (G6) and IIAC

## Separation of duties

| Session type | May declare G6 PASS? | May declare IIAC PASS / IN_FORCE? |
|--------------|----------------------|-----------------------------------|
| Implementer / tester / delivery-closure | **No** | **No** |
| Cursor `bugbot` / `security-review` subagent | **No** | **No** |
| `independent-quality-auditor` in **new** operator-authorized chat | Findings + scoped verdict per protocol | **No** activation — certifier/operator only |
| Certification / governance agent | **No** G6 alone | Manifest validation + publication **after** auditor PASS |

Cursor subagents do **not** provide organizational independence (EVOL-X B6 or IIAC).

## IIAC sessions (post Session A)

| Session | Role | Outcome |
|---------|------|---------|
| A | Implementer | G0+G1+G3 @ `GOVERNANCE_IMPLEMENTATION_SHA`; policy stays `NOT_IN_FORCE` |
| B | Independent auditor | `IIAC-GOV-1` manifest; gates activation |
| C | Independent auditor + certifier | EVOL IIAC CLOSED publication |
| D | Independent auditor + certifier | ENT IIAC CLOSED + FRZ-AUD + OD-ENT-02 disposition |

Norm: [`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`](../../maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md)

## Implementer stop line

1. Lifecycle → `quality_gate`.
2. Checkpoint: stdout from `uv run python -m scripts.governance.igx_gov ticket` + test list from ticket **Test contract**.
3. Operator moves ticket to `ready_for_audit` and starts auditor session with `@intergrax-independent-audit`.

## Ticket fields

- `ready_for_audit` — eligible for external review (Session B for governance batch).
- Auditor updates ticket; **closed** only after operator sign-off.
