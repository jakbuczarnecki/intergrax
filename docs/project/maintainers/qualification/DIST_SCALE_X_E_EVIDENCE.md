# DIST-SCALE-X — Dimension E Evidence (P3)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** (WL-2/3/4 + HTTP WL-6) | `5bc896e2797708f9b71efd1b6611bc910168e4be` |
| **EXECUTION_SHA** (WL-6 Celery @ RabbitMQ AMQP, audit B01) | `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` |
| **EXECUTION_SHA** (WL-4 strict-host @ audit B03) | `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P3-v1` |

## WL-4 strict-host HTTP

| run_id | Notes |
| --- | --- |
| `wl04-ok-540cfdfaef29` | Success @ 180s |
| `wl04-http-503-5a8a1f22bf5e` | Transient 503 |
| `wl04-slow-deadline-4c2150a5ac05` | Slow beyond deadline |
| `wl04-fault-mix-0da741ea0740` | Envelope fault mix |
| `wl04-lost-ack-045b1c18abb9` | Lost response after effect |

Detail: [DIST_SCALE_X_WL4_HTTP_LOOPBACK_PHYSICAL_EVIDENCE.md](DIST_SCALE_X_WL4_HTTP_LOOPBACK_PHYSICAL_EVIDENCE.md)

## WL-6 retry storm

| run_id | Path | Notes |
| --- | --- | --- |
| `wl06-rabbit-df8123a9c607` | **RabbitMQ M + Celery dispatcher** (authoritative) | 60 logical; 74 physical attempts; 14 dispatcher retries; amp **1.23**; `RetryPolicy` owner |
| `wl06-storm-dda074987eb6` | HTTP invoker (supplementary @ `5bc896e27`) | amp ≈ **1.07** |
| `wl06-rabbit-61f709aa5dd6` | superseded diagnostic | test-loop redrive — not canonical |

Detail: [DIST_SCALE_X_P3_AUDIT_CORRECTIVE_CLOSURE.md](DIST_SCALE_X_P3_AUDIT_CORRECTIVE_CLOSURE.md)

Provenance: [DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md)
