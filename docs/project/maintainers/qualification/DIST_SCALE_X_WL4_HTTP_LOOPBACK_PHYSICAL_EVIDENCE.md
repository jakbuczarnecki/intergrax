# DIST-SCALE-X — WL-4 HTTP loopback physical evidence (P3)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** | `5bc896e2797708f9b71efd1b6611bc910168e4be` |
| **Scenario duration** | 180 s per class (envelope default) |
| **Transport** | `127.0.0.1` loopback qualification server only |

## Execution stack (canonical)

1. `build_declarative_invoker_from_tool_wiring` (strict application host)
2. Production `RuntimeToolInvoker` + `RegistryToolExecutor`
3. Registered HTTP qualification tool → loopback handler (`p3_strict_http_stack`)
4. Durable external-operation wiring per TOOL-LIFE-X qual patterns (`INTERGRAX_TOOL_LIFE_DURABLE_WIRING_ENABLED`)

Loopback is **transport + fault injection** only; not an execution authority substitute.

## Run matrix @ EXECUTION_SHA

| run_id | fault_mode | logical_success | logical_failure | physical_attempts | amp (approx.) |
| --- | --- | ---: | ---: | ---: | ---: |
| `wl04-ok-540cfdfaef29` | ok | (see manifest) | — | — | — |
| `wl04-http-503-5a8a1f22bf5e` | 503 | — | — | — | — |
| `wl04-slow-deadline-4c2150a5ac05` | slow | — | — | — | — |
| `wl04-fault-mix-0da741ea0740` | mix | 1521 | 122 | 1643 | 1.08 |
| `wl04-lost-ack-045b1c18abb9` | lost_ack | single-effect assert | — | — | — |

Full metrics: `.tmp/proof/dist-scale-x/<run_id>/dist_scale_x_evidence_manifest.json`

Session log: `.tmp/session/dist-scale-x-p3-envelope/p3-physical-full.log`
