# TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1 — Marketplace Binding Identity, Typed Target & Intent Provenance Reconciliation

| Field | Value |
|---|---|
| **Task** | `TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1` |
| **Parent** | `TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1` → `TRACE-X-P5-R2-P3-R1-R1-R1-R1` → `TRACE-X-P5-R2-P3-R1-R1-R1` → `TRACE-X-P5-R2-P3-R1-R1` → `TRACE-X-P5-R2-P3-R1` → `TRACE-X-P5-R2-P3` → `TRACE-X-P5-R2` → `TRACE-X-P5` → `TRACE-X` |
| **START_HEAD** | `dc8795043e08ff57af51fac7b88177a31d8f9194` |
| **Disposition** | **READY FOR AUDIT** (not CLOSED) |
| **Production delta** | **0** |
| **FRZ-TRC-11** | **OPEN** |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |
| **Supersedes (partial)** | [`TRACE_X_P5_R2_P3_R1_R1_R1_R1_R1_CANONICAL_TOOL_EXECUTION_CONVERGENCE_LOCK.md`](TRACE_X_P5_R2_P3_R1_R1_R1_R1_R1_CANONICAL_TOOL_EXECUTION_CONVERGENCE_LOCK.md) §8–12 (target `binding_provider_id` routing, `catalog-tool-capability:` reference, nullable intent extension); **preserves** accepted activation, ingress, Pattern A, convergence level D, and shared Tool execution core direction from that lock |

## 1 — Canonical state (@ START_HEAD)

| Stage | Status |
|---|---|
| TRACE-X | CURRENT |
| TRACE-X-P5 | CURRENT / BLOCKED ON R2 |
| TRACE-X-P5-R2 | CURRENT / P3 BLOCKED |
| TRACE-X-P5-R2-P3 | BLOCKED |
| TRACE-X-P5-R2-P3-R1-R1-R1-R1 | BLOCKED / PARTIALLY SUPERSEDED |
| TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1 | BLOCKED / SUPERSEDED BY CHILD (this lock) |
| TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1 | **CURRENT** (this lock) |
| P4 | NOT ENTERED |
| DUP-X | FINAL / MANDATORY (roadmap §3.0.3) |

---

## 2 — Independent audit @ `dc8795043e08ff57af51fac7b88177a31d8f9194`

**Accepted (unchanged — do not reopen):**

- convergence level **D**;
- `CapabilityIdentityKey` + `CapabilityReleaseIdentity` semantics;
- one `ToolHostActivationPort`; `ToolRegistryRead` activation truth;
- activation inside canonical Execution;
- one intent store direction; one Tool invocation core;
- zero configured-specific activation service;
- Pattern A unchanged;
- shared bound-execution ingress; one `ExecutionRuntime`; one handler registry **mechanism**.

**Rejected — blockers 18 / 19 / 20** (parent lock partial proposals).

---

## 3 — Blockers 18 / 19 / 20

### R2-P3-BINDING-PROVIDER-IDENTITY-CONFLATION-18

@ HEAD `QualifiedCapabilityExecutionTarget.binding_provider_id` is used simultaneously as:

1. **binding provenance** (who produced the binding result), and  
2. **execution-handler registry key** (`QualifiedCapabilityExecutionBindingHandlerRegistry.resolve(target.binding_provider_id)`).

The superseded child proposed CONFIGURE_EXISTING targets with `binding_provider_id = marketplace.tool.qualified_binding.v1` without running `MarketplaceToolQualifiedCapabilityBindingProvider` — **false provenance** and **forbidden impersonation** of `MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID`.

**Resolution:** introduce a distinct **execution-handler routing identity** on the execution target (and handler protocol), while `binding_provider_id` remains **truthful binding-provider provenance only**.

### R2-P3-EXECUTION-TARGET-STRING-CONTRACT-19

Rejected: `catalog-tool-capability:v1:{serialized CapabilityIdentityKey}` when execution later **parses** that string back into business identity.

**Resolution:** `execution_target_reference` is an **opaque correlation handle** only; `CapabilityIdentityKey` crosses boundaries as **`CapabilityIdentityKey`** via typed configured subject and/or typed intent provenance — never via reparsing the target reference.

### R2-P3-TOOL-INTENT-PROVENANCE-CONFLATION-20

Rejected: extending `QualifiedMarketplaceToolExecutionIntent` with nullable UCA/configured field soup.

**Resolution:** one source-neutral **`MarketplaceToolExecutionIntent`** (contract rename on migration) with a **closed typed provenance union**; impossible combinations unrepresentable at construction time.

---

## 4 — `binding_provider_id` use inventory (@ HEAD)

| Location | Role @ HEAD | Semantic meaning |
|---|---|---|
| `QualifiedCapabilityExecutionTarget.binding_provider_id` | DTO field | Intended provenance; **also** used as registry key (conflation) |
| `QualifiedCapabilityBindingResult.provider_id` | result | Binding provenance (**truthful**) |
| `QualifiedCapabilityExecutionBindingHandler.binding_provider_id` | handler property | Registry key; equals provider id for 1:1 families |
| `QualifiedCapabilityExecutionBindingHandlerRegistry.resolve(binding_provider_id)` | routing | **Execution routing** (misnamed parameter) |
| `QualifiedCapabilityExecutionRuntimeDelegate` | delegate | `resolve(request.execution_target.binding_provider_id)` |
| `ExecutionBoundCapabilityExecutionRuntimeDelegate` | delegate | same |
| `MarketplaceToolQualifiedCapabilityBindingProvider` | producer | Sets `binding_provider_id=self.provider_id` (`marketplace.tool.qualified_binding.v1`) |
| `MarketplaceToolQualifiedCapabilityExecutionHandler` | handler | `binding_provider_id` property; validates `target.binding_provider_id == self.binding_provider_id` |
| `HostAvailableToolCapabilityBindingProvider` | producer | `host_available.tool_capability_binding.v1` — provenance + routing 1:1 |
| `CodeCraftQualifiedCapabilityBindingProvider` / handler | producer + handler | `codecraft.qualified_capability_binding.v1` — 1:1 |
| Tests / composition | fixtures | Mirror production conflation |

**Verdict:** @ HEAD the field means **both** provenance and routing **accidentally** for all families; Marketplace CONFIGURE_EXISTING requires **split** so multiple truthful binding providers can share one Marketplace Tool execution handler.

---

## 5 — Binding provenance vs handler routing (decision)

| Field | Responsibility | Owner |
|---|---|---|
| `binding_provider_id` (target + binding result) | **Who produced** the binding | Binding adapter (`QualifiedCapabilityBindingProvider` family) |
| `execution_handler_id` (new, target v2) | **Which execution-handler family** receives dispatch | Tool / Execution contract (`intergrax/contracts` + Tools handler constants) |

**Forbidden:** alias binding provider IDs whose sole purpose is registry compatibility; CONFIGURE_EXISTING must not set `binding_provider_id` to `marketplace.tool.qualified_binding.v1`.

---

## 6 — Handler identity owner

| Identity | Stable value (canonical write) | Owner |
|---|---|---|
| Marketplace Tool execution handler | `marketplace.tool.execution.v1` | Tools / Marketplace Tool execution contract module (same ownership lane as `MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID` today) |
| UCA Marketplace binding provider | `marketplace.tool.qualified_binding.v1` | `MarketplaceToolQualifiedCapabilityBindingProvider` |
| Configured Marketplace binding provider | `marketplace.tool.configured_capability_binding.v1` (new thin adapter constant) | Tools configured binding adapter |
| CodeCraft handler/binding | existing `codecraft.qualified_capability_binding.v1` | CodeCraft runtime (1:1 — handler id may equal binding provider id until a second binding source shares CodeCraft execution) |
| Host-available tool | `host_available.tool_capability_binding.v1` | AW host-available binding (1:1) |

Applications **compose** handler instances only; they do not own handler id strings.

---

## 7 — Execution target v1 / v2

| Version | Policy |
|---|---|
| **v1** @ HEAD | `qualified_capability_execution_target.v1` — retain **read** compatibility for historical records |
| **v2** (canonical write) | `qualified_capability_execution_target.v2` with required `execution_handler_id` + truthful `binding_provider_id` |

**Forbidden:** nullable `execution_handler_id` on v1 with fallback routing through `binding_provider_id` (permanent dual resolution).

**Migration:** bounded read projection v1→v2 only where historical targets had 1:1 provider/handler mapping; Marketplace UCA v1 targets project `execution_handler_id = marketplace.tool.execution.v1` while preserving truthful UCA `binding_provider_id`. CONFIGURE_EXISTING writes **only** v2.

---

## 8 — `execution_target_reference` responsibility

| Encoding @ HEAD | Classification |
|---|---|
| `marketplace-qualified-tool:v1:{handoff_id}` | **Legacy source-specific protocol** — UCA correlation; handler may parse **only** for UCA provenance adapter (existing); not business `CapabilityIdentityKey` |
| `host-available-tool:{kind}:…` | **Legacy source-specific protocol** — identity sort-key embedding; **DUP-X / COMPAT-X debt** for global neutralization; not CONFIGURE_EXISTING template |
| CodeCraft craft reference | **Opaque / craft-scoped correlation** via existing parser |
| `catalog-tool-capability:v1:*` | **FORBIDDEN** (never implemented @ HEAD) |

CONFIGURE_EXISTING: generate **opaque** instance handles (UUID or binding-operation-derived), e.g. `marketplace-configured-tool:v1:{opaque_correlation_id}` — **no** capability identity payload in the string.

---

## 9 — Typed capability identity carriage

`CapabilityIdentityKey` **REUSE** — carried in:

- `ConfiguredCapabilityExecutionSubject` / configured binding adapter inputs (existing subject lock direction);
- `ConfiguredMarketplaceToolExecutionProvenance.capability_identity` on intent;
- activation/package resolution ports inside handler core.

**Forbidden:** serialize identity into `execution_target_reference` then parse; derive identity from `capability_ref`, `configuration_ref`, provider id, or operation name.

---

## 10 — Intent common truth (canonical v2 write)

Source-neutral core (all provenance variants):

| Field | Notes |
|---|---|
| `execution_request_id` | durable idempotency key |
| `tenant_id` | explicit tenant hop |
| `task_id` | execution correlation |
| `worker_need_id` | fulfillment correlation |
| `selected_operation` | post shared 0/1/N selector |
| `binding_operation_id` | binding correlation |
| `subject_reference` | source-neutral subject correlation (legacy name `qualified_subject_reference` **LEGACY NAMING / COMPATIBILITY** on read v1) |
| `capability_identity` | `CapabilityIdentityKey` when Tool logical identity required for execution |
| `execution_target_correlation` | optional opaque link to `execution_target_reference` |

No UCA handoff, resume, or configuration decision fields in the common core.

---

## 11 — Typed provenance union

```text
MarketplaceToolExecutionIntent
  … common fields …
  provenance: MarketplaceToolExecutionProvenance
```

```text
MarketplaceToolExecutionProvenance =
    UcaMarketplaceToolExecutionProvenance
  | ConfiguredMarketplaceToolExecutionProvenance
```

Closed discriminant (e.g. `provenance_kind` enum + typed variant models). **Forbidden:** parallel optional fields whose validity depends on implicit combinations.

---

## 12 — UCA provenance variant

`UcaMarketplaceToolExecutionProvenance` — truthful UCA facts only:

- `handoff_id`;
- `resume_operation_id`;
- UCA qualified subject reference (if not lifted solely to common `subject_reference`).

No configured decision/configuration ids.

---

## 13 — Configured provenance variant

`ConfiguredMarketplaceToolExecutionProvenance` — truthful Variant B facts only:

- `capability_identity: CapabilityIdentityKey`;
- configuration decision / binding correlation ids (refs, not full provider objects);
- configured subject reference;
- adoption reference **only** if persisted intent truly requires correlation (not execution permission).

`ConfiguredCapabilityExecutionBindingResult.provider_id` = configured binding provider id, **not** execution handler id.

---

## 14 — Repository migration

| Layer | Direction |
|---|---|
| Contract | `QualifiedMarketplaceToolExecutionIntentRepository` → **`MarketplaceToolExecutionIntentRepository`** (source-neutral name); same semantic owner |
| Implementation | **REUSE** `DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository` + `ConditionalDocumentStore` |
| Partition | retain `intergrax.qualified_marketplace_tool_execution_intent.v1` for historical rows |
| Persistence | v1 payload = historical UCA-only shape; **v2 canonical write** = common core + provenance union JSON; explicit `persistence_schema_version` |
| Read | bounded one-way projection: v1 rows → `UcaMarketplaceToolExecutionProvenance`; no silent fallback |

**Forbidden:** `ConfiguredMarketplaceToolExecutionIntentRepository`; second partition; permanent dual truth.

---

## 15 — Compatibility policy

- v1 read support: **bounded**, explicit, non-authoritative, removable after migration window;
- v2 write: **canonical**;
- no `try v2 else v1` authority in handler — repository decode owns version dispatch;
- unbounded legacy → **STOP — ARCHITECTURE DECISION REQUIRED** → COMPAT-X.

---

## 16 — Handler routing (@ target migration)

```text
registry.resolve(target.execution_handler_id) → handler
```

Handler protocol: `execution_handler_id` property (**MODIFY** `QualifiedCapabilityExecutionBindingHandler`; rename semantics from misleading `binding_provider_id`).

`MarketplaceToolQualifiedCapabilityExecutionHandler.execution_handler_id` = `marketplace.tool.execution.v1`.

Handler selects thin provenance adapter from intent `provenance` discriminant — **not** from target string prefix, missing fields, or binding provider impersonation.

---

## 17 — One Marketplace handler proof

| Concern | Count |
|---|---|
| `MarketplaceToolQualifiedCapabilityExecutionHandler` | **1** shared core |
| `QualifiedCapabilityExecutionBindingHandlerRegistry` instances per host composition | **1** |
| Marketplace execution handler registrations | **1** (`marketplace.tool.execution.v1`) |
| Binding adapters (UCA vs configured) | **2** thin adapters → **1** handler id |

---

## 18 — One repository proof

| Store | Count |
|---|---|
| Semantic intent repository contract | **1** (renamed) |
| DocumentStore-backed implementation | **1** |
| Configured-only repository | **0** (forbidden) |

---

## 19 — Duplicate classification matrix

| Pair | Classification |
|---|---|
| UCA vs configured binding adapters | **THIN SOURCE ADAPTER** ×2 |
| Shared target construction core | **FACTOR SHARED CORE** |
| `binding_provider_id` vs `execution_handler_id` | **DISTINCT RESPONSIBILITY** (after v2) |
| Registry resolve key | **SHARED CORE** (evolved key) |
| Intent v1 vs v2 payload | **SCHEMA EVOLUTION** |
| Nullable universal intent DTO | **DUPLICATE / BLOCKER** (rejected) |
| `catalog-tool-capability:` reference | **DUPLICATE / BLOCKER** (rejected) |
| Second handler registry | **FORBIDDEN** |
| `HostAvailableToolCapabilityBindingProvider` precedent | **EVIDENCE** — source-neutral target envelope |

---

## 20 — Ownership matrix

| Concern | Exactly-one owner |
|---|---|
| Capability identity | Capability Catalog (`CapabilityIdentityKey`) |
| Binding provenance | respective binding adapter `provider_id` |
| Handler routing identity | Tools / Execution handler contract |
| Handler registry | `QualifiedCapabilityExecutionBindingHandlerRegistry` (single instance) |
| Intent semantic truth | Tool-domain intent repository |
| Intent persistence | same DocumentStore partition + versioned decode |
| Activation mutation | `ToolHostActivationPort` |
| Activation read | `ToolRegistryRead` |

---

## 21 — Tenant continuity

Unchanged from parent lock: `CapabilityIdentityKey` is **not** tenant-bound; `tenant_id` and scope evidence enforced at opportunity, binding, intent, intake, registry read, and provider pin hops.

Typed provenance must not weaken tenant checks. **PASS — architecture proof complete** (implementation must wire Variant B hops).

---

## 22 — Governance invariants (do not reopen)

CONFIGURE_EXISTING Pattern A; `WorkerExecutionAdmissionPort`; ToolRuntime Governance; no activation during binding-only adapters; configuration authorization ≠ Tool activation ≠ Execution admission.

---

## 23 — Future implementation map

| Artifact | Action |
|---|---|
| `qualified_capability_binding.py` (`QualifiedCapabilityExecutionTarget`) | **SCHEMA EVOLUTION** v2 + provenance fields |
| `qualified_capability_execution_handlers.py` | **MODIFY** resolve by `execution_handler_id` |
| `marketplace_qualified_capability_execution_handler.py` | **MODIFY** handler id + provenance adapters |
| `marketplace_qualified_capability_binding_provider.py` | **REUSE** UCA adapter |
| configured Marketplace binding adapter module | **THIN ADAPTER — NEW** |
| `qualified_marketplace_tool_execution_intent.py` | **SCHEMA EVOLUTION** + provenance union |
| `qualified_marketplace_tool_execution_intent_repository.py` | **MODIFY** v1/v2 decode |
| `marketplace_tool_execution_target_resolution` (shared core) | **FACTOR SHARED CORE** |
| `catalog-tool-capability:` encodings | **DELETE / ELIMINATE** (never ship) |
| Fake UCA `binding_provider_id` on configured path | **DELETE / ELIMINATE** |
| Second registry / handler / intent store | **FORBIDDEN** |

**NEW SEMANTIC MECHANISM = 0** (contract normalization only: one new routing field, typed provenance variants, thin adapter).

---

## 24 — STOP conditions

Return **STOP — ARCHITECTURE DECISION REQUIRED** if implementation requires: binding provider impersonation; second handler registry; second Marketplace handler; second intent repository; identity parse from target strings; nullable provenance soup; permanent dual registry resolution; provider configuration as Tool identity; UCA handoff on pure Variant B; changing Execution or activation authority.

---

## 25 — Implementation exit criteria

1. Blockers 18–20 resolved per this lock.  
2. Truthful `binding_provider_id` + distinct `execution_handler_id` on v2 targets.  
3. CONFIGURE_EXISTING uses `marketplace.tool.configured_capability_binding.v1` (or equivalent) — never UCA binding provider id.  
4. Registry resolves **only** `execution_handler_id`.  
5. `CapabilityIdentityKey` typed end-to-end.  
6. `MarketplaceToolExecutionIntent` + closed provenance union.  
7. One DocumentStore repository with explicit v1→v2 migration.  
8. One Marketplace handler core.  
9. Qualification gates green.  
10. Independent audit @ GitHub commit — not Cursor report alone.

---

## Expected after-graph

```text
UCA binding adapter (binding_provider_id = marketplace.tool.qualified_binding.v1)
configured adapter (binding_provider_id = marketplace.tool.configured_capability_binding.v1)
        \ both set execution_handler_id = marketplace.tool.execution.v1
         → QualifiedCapabilityExecutionTarget v2 (opaque target_reference)
                → ONE QualifiedCapabilityExecutionBindingHandlerRegistry
                → ONE MarketplaceToolQualifiedCapabilityExecutionHandler
                → load MarketplaceToolExecutionIntent (provenance: UCA | CONFIGURED)
                → common Tool execution core → canonical Tool invocation
```

**FRZ-TRC-11 = OPEN** · **P4 = NOT ENTERED**
