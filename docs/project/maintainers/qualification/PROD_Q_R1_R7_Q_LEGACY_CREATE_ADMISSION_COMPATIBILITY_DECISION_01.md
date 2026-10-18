# PROD-Q-R1-R7-Q — Legacy CREATE history compatibility admission decision 01

| Field | Value |
|---|---|
| Status | **RATIFIED — SCOPED DIRECTION; MECHANISM NOT YET VERIFIED OR IMPLEMENTED** |
| Operator approval | 2026-10-18 |
| Evidence baseline | `1dd00c5afdbff35de0cdc8d546aafc2e0f6376a5` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — NOT CLOSED |
| Precedents | Stable CREATE Invocation Identity Decision 01, GCA SQLite Durable Recovery Decision 01, F02 Option B, R2 composition decision, PIS ADR-03 |
| Production delta | **0** |

## 1. Safety problem

The old PIS schema keys physical provider invocations by random `inv-*` `invocation_id` without canonical `execution_id`, tenant or principal fields. The current stable `gca-pinv:*` identity is deterministic per authorized CREATE. If previously executed legacy CREATE remains in PIS but GER/F02 correlation is lost, the current PIS insert can treat the new stable identity as new and physically redispatch CREATE. There is no sanctioned query to reliably correlate PIS-only legacy intent to an execution. A missing legacy row or clean PIS file is not by itself proof of complete, safe upgrade history.

## 2. Ratified direction: trusted history compatibility evidence at existing production admission boundary

Require trustworthy, explicit **history provenance and compatibility admission evidence** before accepting physical CREATE in a deployment that may predate stable CREATE identity. Prefer existing platform process activation, migration-provenance, reconciliation or qualification authority over new contract/storage.

Admit eligible CREATE only when one of the following is **affirmatively established by sanctioned authority**:
1. **Provably fresh deployment:** trusted first initialization and binding of the relevant durable PIS/GCA history, epoch/provenance, process environment and data scope establish that no legacy CREATE can be orphaned.
2. **Upgraded deployment with resolved history:** complete, authorized compatibility/reconciliation evidence establishes the relevant historical CREATE invocations are correctly bound to surviving GER/F02 (or explicitly quarantined with continued deny) without physical re-dispatch.

Otherwise **deny physical CREATE** for the entire affected scope, fail closed. Default to conservative process/environment scope when per-tenant or per-operation completeness cannot be proven. Other operations are not automatically blocked if their distinct admission guarantees are independently proven.

A startup/activation check alone is insufficient where a later change, reopened store, backup restore, reset, file replacement or independent serving path could bypass the evidence: enforce at the existing sanctioned production admission point and bind evidence to the actual backing history, identity epoch and lifecycle, without any alternative provider dispatch.

## 3. Hard architecture and privacy bounds

- Existing `ProviderInvocationStore` / SQLite PIS is the sole provider intent/outcome authority; GCA orchestrator remains sole execution/follow-on semantic owner; exactly one preexisting provider dispatch gate.
- Do not invent shadow registry, new PIS, generic migration framework, duplicate dispatch gate, or operator-editable allow flag.
- Do not heuristically match `inv-*` to `gca-pinv:*` by task/run/idempotency.
- Do not scan multi-tenant PIS payloads outside an authorized information boundary or infer tenant from fields that are not present.
- Do not silently mark history compatible by presence/absence of specific rows, file creation date, schema_version=1, or missing proof.
- Treat historical intent with UNKNOWN/missing provider outcome as potentially externally effected; no ordinary re-dispatch.
- Evidence must be owned by an existing sanctioned activation/provenance authority; identify source, binding scope, signer/admitter, durability, replay/freshness, rollback/restore implications and failure behavior before implementation.
- Any necessary NEW persistent evidence store, contract with new authority, new identity epoch semantics or incompatible migration requires **STOP — ARCHITECTURE DECISION REQUIRED** with exact smallest decision before implementation. Operator approval here is a direction, not permission to fabricate proof.

## 4. Mandatory pre-implementation inventory

Closed-world review of existing process lifecycle and strict serving opener, activation/reconciliation gates, existing durable migration/qualification records, PIS bootstrap/schema and provenance, GCA GER/F02 binding, context tenant/environment authorization and any existing trust anchor. Determine whether existing evidence reliably distinguishes fresh history from legacy upgrade, including restored/copied databases and simultaneous versions. For each candidate record explicit authority, lifecycle, trust, completeness and falsification risks. If no suitable sanctioned evidence exists, **STOP** with minimal missing contract decision.

## 5. Mandatory negative and positive tests

- Demonstrably fresh, correctly bound PIS+GCA data → allow.
- Unproven fresh or empty database, copied/replaced/rolled back backup, missing proof → deny.
- Unresolved legacy PIS-only random-ID CREATE (with or without outcome) → deny without provider call.
- Known legacy GER/F02 binding → permit only legitimate operations, preserve original invocation identity; no unauthorized new physical effect.
- Authorized reconciled/quarantined upgrade → scope-specific allow/deny, no physical redispatch during reconciliation.
- History epoch mismatch, failed evidence read/signature/authorization, stale proof, database corruption → deny.
- Cross-tenant and environment isolation; no enumeration of other tenants' legacy payloads.
- Alternate strict process/HTTP entry paths cannot bypass evidence.
- Restart with new PIS and GCA instances and intentional history loss/replacement.
- Positive authorized new CREATE works after proven-safe admission.
- No new registry/store/gate or silent auto-migration.

Use real SQLite PIS/GCA and a fake provider counting physical dispatch for relevant scenarios. Source tests first; execution only with explicit operator approval.

## 6. Status

Decision ratified; implementation and qualification **NOT PROVEN**. `IMPL-01`, `R7-Q`, `PROD-Q`, all global `FRZ-*` remain OPEN or NOT QUALIFIED. Applicable: `FRZ-REL-03/05`, `FRZ-PRD-08`, `FRZ-SEC-01/04`, relevant OWN/BND/CTR/TYP/RPL and `FRZ-REG-02/03`.

Do not run `pytest` or `uv run` on this architectural inventory without explicit additional operator permission. Max child status READY FOR AUDIT following independent exact-SHA evidence. No historical backfill inferred from post-upgrade state.
