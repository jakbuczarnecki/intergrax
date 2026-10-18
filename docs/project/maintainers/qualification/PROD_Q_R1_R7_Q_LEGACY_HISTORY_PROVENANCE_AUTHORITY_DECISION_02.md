# PROD-Q-R1-R7-Q — Legacy history provenance authority — Decision 02

| Field | Decision |
|---|---|
| Status | **RATIFIED CONTRACT DIRECTION — NO IMPLEMENTATION AUTHORIZATION WITHOUT PROVEN ANCHOR** |
| Operator ratification | 2026-10-18 |
| Baseline | `329255d908a35b8bbb2f66b938fd0a4f76517315` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — BLOCKED |
| Prior decisions | Legacy CREATE Compatibility Decision 01; Stable CREATE Identity Decision 01; SQLite GCA Decision 01; F02 Option B; R2 composition decision; PIS ADR-03 |
| Production delta | **0** |

## 1. Threat model and safety invariant

Legacy random `inv-*` CREATE intents in durable PIS may have caused a physical provider effect but lack canonical execution, tenant and principal identity. If the corresponding GCA GER/F02 disappears, newly derived `gca-pinv:*` is distinct and ordinary PIS INSERTED would redispatch. Historic `inv-*` cannot be matched by heuristics. A blank PIS/GCA pair, schema version, disk timestamp, or operator toggle is not evidence of a safely fresh environment.

**Invariant:** Never physically dispatch `create_work` unless an authoritative, current, durable, scope-bound proof of CREATE history compatibility has been validated at the *existing* dispatch admission, including across process restarts, media restore/replacement, concurrent process access and history rollback.

## 2. Single ownership and evidence semantics

- The existing production process lifecycle, operating under `ProductionControlPlaneGovernance`, is the sole admissibility owner of CREATE **history provenance**. It may not supersede PIS as provider intent/outcome owner, GCA as execution/GER/F02 owner, or the sole `GovernedProviderInvocationDispatchGate` as dispatch enforcer.
- Require one strongly typed, read-only-to-dispatch evidence verdict representing **FRESH_INITIALIZATION** or **AUTHORIZED_RECONCILIATION**, together with environment/process ownership scope, identity epoch (`gca-pinv` generation), current PIS and GCA history identifiers/generations, proof issuer/authorization, continuity and revocation status.
- **FRESH_INITIALIZATION** is established at trusted provisioning of both history media before any CREATE can execute, never inferred later merely from empty files or new schema metadata.
- **AUTHORIZED_RECONCILIATION** requires independently authorized, complete resolution of affected historic CREATE invocations or explicit quarantine of unresolved scope without provider replay; GER/F02 remains the only sanctioned per-execution legacy ID association.
- Default admission scope is the process/environment where tenant segregation is not proven; no scan of other tenants' PIS payloads and no ungrounded tenant inference.

## 3. Independent anti-rollback anchor — mandatory

The authoritative *current* generation/continuity anchor MUST be trustworthy and independent of BOTH PIS and GCA SQLite history files and their backups. A marker, hash or epoch only inside either/both files cannot detect restoration of an old mutually consistent database pair.

The anchor must prevent stale evidence and old database snapshots being accepted as the current history; binding must include current PIS/GCA history identities and generations and guard independent replacement, reset, backup restore, rollback and restoration of both histories together. A simple static file hash is not sufficient for normal mutable databases: design must distinguish authorized advancing writes from lost or reverted history, with atomic/consistent validation or a safe recovery protocol for partial transitions and crashes.

This decision **does not claim** that `ProductionControlPlaneGovernance` already has an external durable, monotonic trust anchor. Inventory its real authority and persistent backing first. If there is no existing approved anti-rollback/continuity capability, STOP with the *smallest* required owner/port and durable trust-anchor decision. **Do not create a new service, database, shadow registry, epoch store or operator-editable acceptance flag without separate authorization.**

## 4. Enforcement and lifetime

- Enforce the verified verdict in the **existing** `GovernedProviderInvocationDispatchGate.dispatch_after_intent_persisted` for `operation == "create_work"`, before physical `execute()`; deny even where PIS returns INSERTED if compatible history is unproven.
- A startup preflight alone is insufficient; each CREATE dispatch must be guarded against stale provenance or file swap. No bypass via strict HTTP, process lifecycle, alternate entry points, different process or restart.
- Protect against time-of-check/time-of-use changes between proof validation, PIS intent insert and dispatch, including concurrent writers; semantics must fail closed when proof and media state cease to be bound.
- `submit_quote_acceptance` and `cancel_work` keep their separate existing admissibility and are not incidentally blocked or silently weakened by this CREATE-specific check.
- Unknown/missing PIS outcome for any already admitted invocation continues to deny ordinary physical re-dispatch.
- Failure (no proof, bad scope/issuer, stale proof, unreadable provenance, generation mismatch, rollback, partial media replacement, failed persistence) means zero physical CREATE and no implicit migration. Any administrative reconciliation is separate, authorized and effects-free.

## 5. Delivery gates and acceptance

Before implementation prove the concrete existing anchor, owner and persistence; monotonic advancement/recovery semantics; environment and media binding; authoritative provisioning or reconciliation protocol; and exact producer/consumer contracts. If these cannot be met, **STOP — ARCHITECTURE DECISION REQUIRED — TRUSTED HISTORY ANCHOR**. Do not bypass this gate by hardcoding fixture data.

Required tests (source may be prepared without execution): fresh provenance positive; unproven empty pair negative; authorized resolved history and unresolved legacy negative; missing/UNKNOWN provider outcome; PIS-only orphan; one/both databases replaced, restored or rolled back; rollback of entire pair plus stale proof; independently advancing writes and interrupted partial commits; concurrent/TOCTOU admission; strict serving and process opener; tenant/environment isolation; accept/cancel unchanged; actual provider fake call count zero on every denied CREATE. Where appropriate real SQLite GCA+PIS and fresh instances.

`pytest=0` and `uv run=0` by default pending explicit operator approval. No production code may be implemented without proof of the sanctioned anchor, nor any global FRZ marked PASS from this document.

## 6. Status and boundaries

**Decision 02 RATIFIED**; specific trust anchor and implementation **UNPROVEN**, no production delta, no test execution. Legacy admission remains BLOCKED; `IMPL-01`, `R7-Q`, `PROD-Q` and all applicable `FRZ-*` remain OPEN / NOT QUALIFIED. This decision does not authorize a new persistence owner or fabricated continuity evidence.
