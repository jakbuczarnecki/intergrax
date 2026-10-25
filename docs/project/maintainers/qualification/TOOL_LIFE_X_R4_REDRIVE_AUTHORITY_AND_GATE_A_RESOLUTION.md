# TOOL-LIFE-X R4 — redrive authority & Gate A resolution

| Field | Value |
| --- | --- |
| **START_HEAD** | `d21e1811b5333726dccbd55c989c09b29fcba79c` |
| **IMPLEMENTATION_SHA** | *set at publication commit* |
| **Disposition** | `TOOL-LIFE-X R4 — PARTIAL / EXACT BLOCKERS` |

## OD-R4-REDRIVE-DIRECTION — implementation

**Contract:** `transfer_redrive_worker_execution_authority` (store CAS) + `transfer_redrive_authority_after_reconciliation` (reconciliation admission + Class B gate).

| Guarantee | Mechanism |
| --- | --- |
| Old epoch stays revoked | Revocation rows immutable; transfer requires `bound_worker_execution_epoch_is_revoked` |
| No reuse of revoked authority | `worker_ownership_epoch` must strictly increase; new worker binding |
| Atomic durable transfer | Single `_compare_and_set_locked` / SQLite `BEGIN IMMEDIATE` |
| Certified-loss audit | Requires `consumed_certified_loss_evidence_ids` |
| Competing recovery denied | `consumed_redrive_admission_ids` single-shot |
| Eligible class only | `ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED` |
| Class C fail-closed | `CLASS_C_FAIL_CLOSED` rejected at orchestration |

**Qual:** `test_tool_life_x_r4_f6_scenario_24_recovery.py` — positive re-drive + negatives.

## OD-R4-G-A02-SAFETY-DIRECTION — class map (honest)

| Sink | Class A | Class B | Class C |
| --- | --- | --- | --- |
| DB (platform store) | Durable CAS / revocation | Idempotent store-scoped redrive (scenario 24) | Arbitrary multi-row effects without reconciliation |
| HTTP | Provider atomic fence when contract supports | Batch D idempotency + reconciliation plugins | Blind POST retry |
| MCP | Not claimed | Catalog child with side_effect flags + qual | Non-idempotent `tools/call` UNKNOWN |

No universal remote atomic fencing claimed.

## Consolidated verification

| Suite | Result | Log |
| --- | --- | --- |
| R4 deduplicated L2 | **207 passed** | `.tmp/session/TOOL-LIFE-X-R4/pytest-r4-redrive-consolidated.log` |
| Scenario 24 | **7 passed** | same module |

## Remaining (not CLOSED)

- **OD-R4-G-A07-PROD-ACTIVATE** — production wiring/recovery activation.
- **HTTP/MCP Class B** — per-tool physical qual beyond existing Batch D / F3 subsets.
- **EXT-L4** independent parent audit.
- **TOOL-LIFE-X parent** — OPEN.
