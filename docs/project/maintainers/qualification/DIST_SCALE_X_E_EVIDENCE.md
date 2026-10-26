# DIST-SCALE-X — Dimension E Evidence (P3)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** | `5bc896e2797708f9b71efd1b6611bc910168e4be` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P3-v1` |

## Authoritative runs

| run_id | Workload | Retry / fault notes |
| --- | --- | --- |
| `wl04-ok-540cfdfaef29` | WL-4 | Strict-host HTTP success @ 180s |
| `wl04-http-503-5a8a1f22bf5e` | WL-4 | Transient 503 class |
| `wl04-slow-deadline-4c2150a5ac05` | WL-4 | Slow beyond tool deadline |
| `wl04-fault-mix-0da741ea0740` | WL-4 | 10% 503 / 5% slow / 2% lost-ack mix parameters |
| `wl04-lost-ack-045b1c18abb9` | WL-4 | Single logical effect under lost-response-after-effect |
| `wl06-storm-dda074987eb6` | WL-6 | 120s fault window; amp ≈ **1.07**; drain slice |

HTTP path: `build_declarative_invoker_from_tool_wiring` → `RuntimeToolInvoker` → loopback oracle (see [DIST_SCALE_X_WL4_HTTP_LOOPBACK_PHYSICAL_EVIDENCE.md](DIST_SCALE_X_WL4_HTTP_LOOPBACK_PHYSICAL_EVIDENCE.md)).

## Audit note (WL-6 broker path)

P3 WL-6 physical run exercises **HTTP invoker retry** under fault mix (infra check includes RabbitMQ; broker task-queue storm not replayed in this module). Independent audit should confirm SAC slice vs. envelope § WL-6 RabbitMQ M obligation.

Provenance: [DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md)
