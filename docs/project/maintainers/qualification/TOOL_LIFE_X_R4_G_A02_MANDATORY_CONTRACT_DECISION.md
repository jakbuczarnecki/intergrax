# TOOL-LIFE-X R4 — G-A02 mandatory contract decision (evidence-gap closure)

| Field | Value |
| --- | --- |
| **START_HEAD** | `c9fad321a7dc26195237da05a9e6a5c09b159f2c` |
| **Disposition** | Per-sink honesty; no waiver |

## DB (durable external-operation store)

| Guarantee | Required | Evidence | Limitation |
| --- | --- | --- | --- |
| Store / authorization fencing | Yes | R8, Gate A CAS, SQLite `BEGIN IMMEDIATE` | — |
| Physical effect initiation fence | Yes | `mark_physical_running` + revoked epoch | In-process only unless shared store |
| Atomic provider-side fence | N/A | — | Store is authoritative for platform state, not remote DB rows |
| In-flight UNKNOWN | Yes | Certified loss → UNKNOWN | — |
| Idempotency / reconciliation | Yes | Stage/apply + consumed evidence ids | — |

## HTTP (provider resource lifecycle)

| Guarantee | Required | Evidence | Limitation |
| --- | --- | --- | --- |
| Authorization fencing | Yes | Batch D L3 | — |
| Physical send initiation | Class-specific | Batch D admission + revoke deny | **No universal atomic remote fence** |
| Atomic provider-side fence | Only where provider contract supports it | Batch D scoped rows | **ARCHITECTURE DECISION REQUIRED** for effects needing remote CAS beyond current HTTP adapter |
| In-flight UNKNOWN | Yes | Batch B/D UNKNOWN paths | — |
| Idempotency / reconciliation | Yes | Idempotency coordinator + tool contracts | Non-idempotent effects remain BLOCKED for blind retry |

## MCP (subprocess / streamable transport)

| Guarantee | Required | Evidence | Limitation |
| --- | --- | --- | --- |
| Authorization fencing | Yes | R2 + F3 streamable HTTP L3 | — |
| `tools/call` admission | Yes | Ingress + cross-loop qual | — |
| Atomic provider-side fence | Not claimed | — | Local admission ≠ remote tool atomicity |
| In-flight UNKNOWN | Yes | Cancel / disconnect qual (F3) | Late provider result → bridge cancel parity |
| Idempotency / reconciliation | Class-specific | Catalog child + side_effect flags | — |

## Scenario 24 (TL8 positive re-drive)

| Prerequisite | Status |
| --- | --- |
| RUNNING → UNKNOWN after certified loss + revocation | **PROVEN** — `test_scenario_24_certified_loss_transitions_running_to_unknown` |
| Reconciliation admission on UNKNOWN | **PROVEN** — gate validates physical UNKNOWN only |
| Fresh physical RUNNING after admission | **BLOCKED** — revoked `worker_ownership_epoch` remains on durable record; `mark_physical_running` rejects (`ExternalOperationFencedCommitRejectedError`) |
| **ARCHITECTURE DECISION REQUIRED** | Sanctioned **redrive authority** contract: clear or supersede bound worker epoch after reconciliation admission (or equivalent single CAS) before eligible idempotent re-drive |

## Summary

G-A02 **cannot** be marked universally PASS for atomic provider-side fencing across all effect classes. Platform evidence supports store-level and authorization-level guarantees; HTTP/MCP remote atomicity remains **class-specific** and **documented**. TL8 **positive** re-drive remains **BLOCKED** until redrive-authority contract above is ratified and qualified.
