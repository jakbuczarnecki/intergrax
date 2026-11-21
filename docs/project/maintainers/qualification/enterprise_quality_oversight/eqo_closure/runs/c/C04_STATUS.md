# EQO-C-04 — status refresh (wave-7, log-derived)

**Parent:** `EQO-C-04` · **Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Source log:** `eqo_closure/runs/c/c04_targeted_replay.log` (wave-3 execute; no re-run in wave-7)

## Pytest summary (from log tail)

| Field | Value |
|-------|-------|
| Exit code | `1` |
| Failed | `47` |
| Passed | `395` |
| Duration | `70.90s` |

## Failure bands (representative)

| Band | Examples in log |
|------|-----------------|
| TRACE-X closed-world | `test_trace_x_p6_closed_world_gates.py` (G-class / impact_map) |
| MCP P5 batch | `test_mcp_server_x_p5_qualification_batch.py` (trust principal binding exit 2) |
| Public docs contracts | `tests/unit/docs/test_*_contract.py` (LKW, partners, roadmap, README brevity, link integrity) |

## EQO status

`BLOCKED` — targeted replay executed once under pytest lock; product/train drift at TRAIN_PIN dominates failures. Wave-7 records custody only; no pytest re-execution in this lane.

**Matrix:** row C04 remains `PARTIALLY_MET` (replay attempted, non-zero exit).

**Ref:** `governance/tickets/T-IGX-046-F.md`