# TOOL-LIFE-X R4 — evidence-gap closure (F3 composition + F6 honesty)

| Field | Value |
| --- | --- |
| **START_HEAD** | `c9fad321a7dc26195237da05a9e6a5c09b159f2c` |
| **IMPLEMENTATION_SHA** | `5843bfcda2a06b0ba0a37d1b43b0d4f640c9af7f` |
| **PUBLICATION_HEAD** | `5843bfcda2a06b0ba0a37d1b43b0d4f640c9af7f` |
| **Disposition** | `TOOL-LIFE-X R4 — EVIDENCE GAP CHECKPOINT PUBLISHED — PARENT OPEN` |

## F3 — physical HTTP + canonical composition

| Item | Result |
| --- | --- |
| `object.__setattr__` removed | **PASS** — `root_child_work_ingress` on `build_host_task_execution` / `nexus_host_execution.py` |
| uvicorn + Streamable HTTP L3 | **PASS** — 3 tests in `test_mcp_catalog_f3_streamable_http_physical_l3.py` |
| OWN-A2 task ownership | **PASS** — `task_id` on `run_agent` result matches binding |
| Transport disconnect (non-cooperative) | **PASS** — `test_f3_streamable_http_transport_disconnect_during_catalog_child` |
| Production Option C | **unchanged** — default-off |

## F6 — scenario 24

| Step | Status |
| --- | --- |
| 1–4 RUNNING → revoke → UNKNOWN | **PASS qual** |
| 5–8 positive re-drive → terminal | **BLOCKED** — `test_scenario_24_positive_redrive_blocked_until_redrive_authority_contract` |
| 9 stale / forged negatives | **PASS** — existing tests |

## G-A02

See [`TOOL_LIFE_X_R4_G_A02_MANDATORY_CONTRACT_DECISION.md`](TOOL_LIFE_X_R4_G_A02_MANDATORY_CONTRACT_DECISION.md).

## G-A07

Wiring evidence unchanged; **no production activation**. Qualification ≠ deployment.

## Consolidated verification

| Command | Result | Log |
| --- | --- | --- |
| R4 deduplicated L2 (ext ops + batch A–D + R1/R2 + F3 + wiring) | **204 passed** | `.tmp/session/TOOL-LIFE-X-R4/pytest-r4-evidence-gap-consolidated.log` |
| Ruff (touched tier-0/1 + qual tests) | **PASS** | same session |

## EXT-L4 (independent parent audit)

Mandatory external audit; operator must not treat this document as EXT-L4 closure.
