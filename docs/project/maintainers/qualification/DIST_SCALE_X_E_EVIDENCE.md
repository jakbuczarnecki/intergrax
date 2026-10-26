# DIST-SCALE-X — Dimension E Evidence (P3)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** (WL-2/3/4 + HTTP WL-6) | `5bc896e2797708f9b71efd1b6611bc910168e4be` |
| **EXECUTION_SHA** (RabbitMQ M WL-6) | `fc5960a15b02c98e46abbce0534b63ac1322864a` |
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
| `wl06-rabbit-61f709aa5dd6` | **RabbitMQ M** (mandatory) | 198 logical submissions; **971** physical handler attempts; amp ≈ **4.90**; semantic effects **198** (no dup); 120s fault @ 20% `RetryableHandlerError`; drain to terminal success |
| `wl06-storm-dda074987eb6` | HTTP invoker (supplementary @ `5bc896e27`) | amp ≈ **1.07**; strict-host slice |

Rabbit path: `create_rabbitmq_integration` + `create_rabbitmq_worker` + `execute_logical_task` handler; qual orchestration re-enqueues on broker `FAILED` during fault/drain (consumer `auto_ack` — no broker redelivery).

Provenance: [DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md)
