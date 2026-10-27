# DIST-SCALE-X — Dimension N Evidence (P6)

| Field | Value |
| --- | --- |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P6-v1` |

## Chaos matrix

| Seed | Fault | Delegate |
| --- | --- | --- |
| N-01 | worker SIGKILL | P4 G-01 crash |
| N-02 | PG pause | P5 I PG |
| N-03 | Rabbit stop | P5 I Rabbit |
| N-05 | event deny | P4 F-EVENT |

Test module: `p6_chaos_matrix.py` — manifest per seed @ EXECUTION_SHA.
