# TOOL-LIFE-X R4 — EXT-L4 final corrective evidence bundle

| Field | Value |
| --- | --- |
| **AUDIT_START_HEAD** | `0c1b3119b78caeac24ddcca1daa0cdeac00ebb3f` |
| **FROZEN_IMPLEMENTATION_SHA** | See `git rev-parse origin/development` after push (this document is not amended for publication SHA recursion). |
| **Disposition** | `TOOL-LIFE-X R4 — READY FOR FINAL INDEPENDENT EXT-L4` |

## B1 — trusted-contract binding guarantee

Durable operation records store immutable `bound_redrive_effect_*` fields. `bind_authoritative_redrive_effect_to_operation` resolves `contract_id` only via `resolve_trusted_redrive_effect_contract` and, when `provider_id` + `tool_id` are supplied, enforces sanctioned execution context via `assert_trusted_redrive_binding_matches_execution_context` (`intergrax/contracts/external_operation_redrive_trusted_effect_contracts.py`). `transfer_redrive_authority_after_reconciliation` requires durable binding match before Class B CAS.

**Call sites (production):** `bind_authoritative_redrive_effect_to_operation` is defined in `redrive_external_operation_authority.py`; qualification and adversarial tests exercise the sanctioned path. Untrusted Class B contract_id with Class C tool context is rejected at bind time.

| Case | Test | Result |
| --- | --- | --- |
| Class B contract for Class C tool at bind | `test_class_b_contract_bind_rejected_for_class_c_tool_execution_context` | PASS |
| Class B contract for Class C binding at transfer | `test_substitute_class_b_contract_for_class_c_bound_operation_rejected` | PASS |
| Forged operation_key | `test_mismatch_contract_operation_key_rejected` | PASS |
| Cross-tenant | `test_cross_tenant_redrive_substitution_rejected` | PASS |
| Stale CAS revision | `test_stale_store_revision_transfer_rejected` | PASS |
| Competing authority | `test_competing_redrive_authority_still_rejected` | PASS |
| Forged reconciliation | `test_forged_reconciliation_without_unknown_admission_rejected` | PASS |

## Integrated HTTP Class B — physical lost-ack L3 (B2+F6)

| Step | Evidence |
| --- | --- |
| 1–14 lost-ack → UNKNOWN → redrive → single effect | `test_integrated_http_class_b_lost_ack_redrive_single_physical_effect` — provider commits then `socket.SHUT_RDWR` before response; store stays non-SUCCESS until sanctioned redrive |
| Class C fail-closed | `test_integrated_http_class_c_bound_operation_rejects_class_b_redrive` |
| Substitution | `test_integrated_http_class_b_substitution_after_class_c_bind_rejected` |

Log: `.tmp/session/TOOL-LIFE-X-R4/pytest-ext-l4-corrective.log`

## Verification commands

| Suite | Result | Log |
| --- | --- | --- |
| EXT-L4 corrective + R6/R7/R8 + Gate A spot | **60 passed** | `pytest-ext-l4-corrective.log` |
| R4 consolidated L2 (23 modules) | **215 passed** | `pytest-ext-l4-l2-consolidated.log` |
| Ruff (touched tier-0/1 + qual tests) | PASS | session |

## G-A07 vs production activation

| Requirement | Status | Evidence |
| --- | --- | --- |
| Composition qualification (default-off, topology, materialization) | PASS scoped | `test_tool_life_host_bootstrap_wiring.py` |
| Live production recovery activation before formal CLOSED | **NOT REQUIRED for this evidence batch** | Qualification ≠ deployment |
| Production recovery activation | **NOT AUTHORIZED** | Operator gate `OD-R4-G-A07-PROD-ACTIVATE` — remains blocker for parent CLOSED |

## PostgreSQL Docker L3 @ corrective SHA

**REUSE** — not closure-critical for EC10/EC13 at this SHA; prior evidence `test_tool_life_x_l3_physical_postgresql_qualification.py` (Docker-gated).

---

## EC1–EC15 (individual)

| ID | REQUIREMENT | STATUS | OWNER | TEST | PHYSICAL | RESIDUAL |
| --- | --- | --- | --- | --- | --- | --- |
| EC1 | Tenant-safe external operation foundation | PASS scoped | `external_operation_state_store` | R6 suite | SQLite L3 | — |
| EC2 | Deadline hierarchy | PASS scoped | nexus tools batch A | `test_tool_life_x_batch_a_deadline_hierarchy` | in-process | — |
| EC3 | Retry/idempotency/UNKNOWN | PASS scoped | idempotency coordinator | batch B | in-process | — |
| EC4 | Governance authorization | PASS scoped | batch B real gov | batch B | in-process | — |
| EC5 | Stalled/abandoned recovery | PASS scoped | batch C | batch C | in-process | — |
| EC6 | Provider resource lifecycle | PASS scoped | batch D | batch D | HTTP/SQL bounds | — |
| EC7 | OD record unchanged | PASS scoped | qual record | prior doc | — | — |
| EC8 | MCP host lifecycle | PASS scoped | R2 qual | R2 L3 | subprocess | — |
| EC9 | F3 streamable HTTP catalog | PASS scoped | applications F3 | F3 physical 3 tests | uvicorn loopback | — |
| EC10 | Durable single-host persistence | PASS scoped | sqlite durable stores | R8 + scenario 24 | SQLite file | PG reuse |
| EC11 | Certified owner loss | PASS scoped | R7 issuer | R7 | SQLite | — |
| EC12 | Revocation supervision | PASS scoped | R9 | R9 | SQLite | — |
| EC13 | Commit authority loss | PASS scoped | R10 | R10 | SQLite | prod activate for CLOSED |
| EC14 | Redrive authority + binding | PASS scoped | redrive authority | EXT-L4 lost-ack HTTP + bind context + adversarial | loopback HTTP transport loss | per-effect qual beyond registry |
| EC15 | Independent EXT-L4 audit | PARTIAL | qualification bundle | this document | — | **mandatory independent EXT-L4** |

## Scenarios 1–24 (individual)

| # | STATUS | TEST / EVIDENCE | RESIDUAL |
| --- | --- | --- | --- |
| 1 | PASS scoped | batch A deadline | — |
| 2 | PASS scoped | batch A | — |
| 3 | PASS scoped | batch B idempotency | — |
| 4 | PASS scoped | batch B | — |
| 5 | PASS scoped | batch B governance | — |
| 6 | PASS scoped | batch C | — |
| 7 | PASS scoped | batch C | — |
| 8 | PASS scoped | batch D SQL | — |
| 9 | PASS scoped | batch D HTTP | — |
| 10 | PASS scoped | batch D browser | — |
| 11 | PASS scoped | batch D MCP bounds | — |
| 12 | PASS scoped | batch E adversarial | — |
| 13 | PASS scoped | R1 consolidated | — |
| 14 | PASS scoped | R2 MCP host | — |
| 15 | PASS scoped | R6 tenant | — |
| 16 | PASS scoped | R7 certified loss | — |
| 17 | PASS scoped | R8 revocation CAS | — |
| 18 | PASS scoped | R9 auth revoke | — |
| 19 | PASS scoped | R10 commit authority | — |
| 20 | PASS scoped | F3 cancel cross-loop | — |
| 21 | PASS scoped | F3 streamable HTTP | — |
| 22 | PASS scoped | G-A07 wiring | — |
| 23 | PASS scoped | R4 remediation L2 reuse | — |
| 24 | PASS scoped | scenario 24 + EXT-L4 lost-ack HTTP integrated | — |

## G-A01–G-A09 (individual)

| ID | STATUS | OWNER | TEST | RESIDUAL |
| --- | --- | --- | --- | --- |
| G-A01 | PASS scoped | durable composition | R6–R10 | — |
| G-A02 | PASS scoped | trusted contract + binding + execution context | EXT-L4 B1/B2 | per-effect HTTP/MCP |
| G-A03 | PASS scoped | UNKNOWN reconcile gate | batch B + recovery gate | — |
| G-A04 | PASS scoped | batch A | batch A | — |
| G-A05 | PASS scoped | batch B | batch B | — |
| G-A06 | PASS scoped | batch D | batch D | — |
| G-A07 | PASS scoped wiring | host bootstrap | wiring tests | **OD-R4-G-A07-PROD-ACTIVATE** for CLOSED |
| G-A08 | PASS scoped | regression L2 | 215/215 | — |
| G-A09 | PASS scoped | scenario 24 + adversarial + lost-ack | EXT-L4 | — |

## FRZ / ECAP

| ID | STATUS | RESIDUAL |
| --- | --- | --- |
| FRZ-CTR-* | PASS scoped — no new semantic owners | — |
| FRZ-OWN-* | PASS scoped — redrive remains store CAS | — |
| FRZ-BND-* | PASS scoped — tier boundaries preserved | — |
| ECAP P1 | PASS — evidence @ frozen SHA | — |
| ECAP P2 | PASS — physical L3 where claimed (lost-ack HTTP) | — |
| ECAP P3 | PASS — no mock-only closure for EXT-L4 HTTP row | — |
| ECAP P4 | PARTIAL — independent EXT-L4 audit pending | external sign-off |

## Remaining mandatory blockers (not self-CLOSED)

1. Independent **EXT-L4** parent audit / sign-off.
2. **OD-R4-G-A07-PROD-ACTIVATE** — production recovery activation (qualification complete; deployment not authorized).
3. Additional per-tool Class B HTTP/MCP effects require new trusted registry entries + L3.

## Formal recommendation

`TOOL-LIFE-X R4 — READY FOR FINAL INDEPENDENT EXT-L4` — parent **OPEN**; not CLOSED.
