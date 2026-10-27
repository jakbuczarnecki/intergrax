# DIST-SCALE-X — Dimension J Evidence (P5)

| Field | Value |
| --- | --- |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P5-v1` |

## Reuse (no rerun)

| Case | EXECUTION_SHA | Anchor |
| --- | --- | --- |
| J-01 competing writers | `8656fb7d8c42e24e1448c4b7e97f18a15c1d765c` | WL-3 |
| J-02 duplicate delivery | `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` | WL-6 Rabbit |
| J-04 lost ACK | `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` | WL-4 lost_ack |
| J-07 retry after fail | `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` | WL-6 |
| J-08 slow/timeout | `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` | WL-4 slow |

## P5 gap coverage

| Case | Test |
| --- | --- |
| J-06 stale replay | `test_p5_j_gap_stale_replay_rejected_by_idempotency_ledger` |
