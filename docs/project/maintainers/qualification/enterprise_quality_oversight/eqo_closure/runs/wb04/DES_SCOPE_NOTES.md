# EQO-W3-WB04-DES — scope notes (Campaign B-1)

**Source:** `EQO_B1_CRITICAL_INVARIANTS_PLAN.md` · **Pin:** `TRAIN_PIN` (`28c27cb`) · **Parallel group:** S1 · **Milestone:** B (`DES` only).

| Field | Value |
|-------|-------|
| Workstream | WB04 |
| Domain | Security |
| Finding IDs | EQOA-A2-047, EQOA-A7-013, EQOA-A5-015 |
| Ticket refs | T-IGX-051-B |
| Primary matrix | `EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl` |
| Write set | `eqo_closure/runs/wb04/` only |

Probe bundle spans CTRL-X qualification replay, security composition (A2-047), and MCP-SERVER-X P5 batch (A7-013 MCP exposure). ENT-AUDIT FULL staleness (A5-015) is dispositioned after EXE evidence, not by DES alone.

**Acceptance (DES):** preregistered SHA256 + hypotheses on `TRAIN_PIN`.