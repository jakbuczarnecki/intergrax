# DIST-SCALE-X — Failure envelope (P7)

| Field | Value |
| --- | --- |
| **Mode** | `ENVELOPE_ONLY` |
| **SAC** | `SAC-DIST-SCALE-X-v1` §3.1 automatic FAIL invariants |

## In-scope failure classes (closed-world)

| Class | Qual path | Expected behavior |
| --- | --- | --- |
| Broker unavailable | P5 I-01 | Bounded outage; recovery after `docker start` |
| PG unavailable | P5 I-02, P4 WL-5 | Typed errors / fail-closed; no tenant bleed |
| Worker crash | P4 G, P6 N-01 | Durable state; reclaim work |
| Event store deny | P4 F, P6 N-05 | Fail-closed append |
| Admission saturation | P5 L | `DependencyConcurrencyExceededError` (REJECT) |
| Retry storm | P3 WL-6 | Bounded amplification (cited SHA) |

## Explicit N/A (with evidence)

| ID | Mechanism |
| --- | --- |
| I-NA-01 | Consensus / WAN partition tolerance |
| I-06 | Redis KV partition @ default LKW qual topology |

## Out of scope / forbidden chaos (G-04)

Volume delete, shared production stop, host-wide network cut, live vendor sandboxes without OD.

## WL-7 soak failure signals (M)

Monitor: RSS slope samples, p95 latency samples, error_count, `soak_complete_envelope` in manifest. **Exit 11** requires 14400 s run with `soak_complete_envelope=1`.
