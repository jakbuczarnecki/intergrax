# DIST-SCALE-X — Dimension I Evidence (P5)

| Field | Value |
| --- | --- |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P5-v1` |
| **EXECUTION_SHA** | `ddb18890a216b0c51abc9e463efc31f73dcfe359` |
| **Disposition** | **PASS** (scoped P5 physical) — L9 parent audit pending |

## N/A with evidence

| ID | Claim | Evidence |
| --- | --- | --- |
| **I-NA-01** | No Raft/consensus partition tolerance | P0 inventory; roadmap §3.0.6 I |

## Executed / cited

| ID | Scenario | Status |
| --- | --- | --- |
| **I-01** | Rabbit stop/start | `test_p5_i_rabbit_outage_bounded_recovery` |
| **I-02** | PG pause mid-write | `test_p5_i_pg_pause_bounded_errors_recover` |
| **I-05** | Event store deny | **CITED** P4 F @ `f0c49fd3cff8c00fef096e4db96f7ddcf07c9845` |

## Residual

**I-06** Redis KV partition — **N/A WITH EVIDENCE** (LKW qual topology).
