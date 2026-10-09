# TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1-R1 — Target Compatibility & Intent Identity Final Reconciliation

| Field | Value |
|---|---|
| **Task** | `TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1-R1` |
| **Parent** | `TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1` |
| **START_HEAD** | `5f348257e7ff506f57a2b8f381c131ee0e62599f` |
| **Disposition** | **READY FOR AUDIT** (not CLOSED) |
| **Production delta** | **0** |
| **FRZ-TRC-11** | **OPEN** |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |
| **Supersedes (partial)** | [`TRACE_X_P5_R2_P3_R1_R1_R1_R1_R1_R1_MARKETPLACE_BINDING_TARGET_INTENT_RECONCILIATION_LOCK.md`](TRACE_X_P5_R2_P3_R1_R1_R1_R1_R1_R1_R1_MARKETPLACE_BINDING_TARGET_INTENT_RECONCILIATION_LOCK.md) §7 runtime target v1→v2 compatibility; §9/§13 duplicate `CapabilityIdentityKey` in configured provenance; §8 UUID-vs-deterministic ambiguity for configured target handles. **Preserves** all other accepted decisions from that lock and ancestors (binding vs handler split, one handler registry, one Marketplace handler, opaque target reference, typed provenance union, one intent repository, convergence D, Pattern A). |

## 1 — Canonical state (@ START_HEAD)

| Stage | Status |
|---|---|
| TRACE-X | CURRENT |
| TRACE-X-P5 | CURRENT / BLOCKED ON R2 |
| TRACE-X-P5-R2 | CURRENT / P3 BLOCKED |
| TRACE-X-P5-R2-P3 | BLOCKED |
| TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1 | BLOCKED / SUPERSEDED BY CHILD (this lock) |
| TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1-R1 | **CURRENT** (this lock) |
| P4 | NOT ENTERED |
| DUP-X | FINAL / MANDATORY |
| **NEW SEMANTIC MECHANISM** | **0** |

---

## 2 — Independent audit rejection @ `5f348257e7ff506f57a2b8f381c131ee0e62599f`

**Accepted from parent lock (do not reopen):** truthful `binding_provider_id` vs `execution_handler_id`; one handler registry; one Marketplace Tool execution handler; opaque `execution_target_reference`; typed `CapabilityIdentityKey`; one Tool execution intent direction; typed closed UCA/configured provenance union; one intent repository; no configured-specific activation/registry/executor; convergence level D; Pattern A unchanged.

**Rejected — remaining blockers from parent qualification:**

- **R2-P3-EXECUTION-TARGET-COMPATIBILITY-WITHOUT-EVIDENCE-21** — proposed runtime `QualifiedCapabilityExecutionTarget` v1 read projection → v2 without durable target persistence evidence.
- **R2-P3-INTENT-CAPABILITY-IDENTITY-DUPLICATION-22** — `CapabilityIdentityKey` in common intent truth **and** `ConfiguredMarketplaceToolExecutionProvenance`.

---

## 3 — Blocker 21 (target compatibility without evidence)

Parent §7 assumed bounded **runtime** v1→v2 target migration. Closed-world audit @ START_HEAD finds **no** dedicated target persistence path (no repository `model_dump` / `model_validate`, no DocumentStore/KV partition, no event durability for `QualifiedCapabilityExecutionTarget`).

**Resolution:** eliminate runtime target compatibility machinery. Future implementation performs **one atomic contract migration** for all producers, delegates, handler protocol, registry resolution, and tests.

---

## 4 — Target persistence inventory (@ START_HEAD)

| Surface | Role |
|---|---|
| `intergrax/contracts/capability_qualification/qualified_capability_binding.py` | Defines `QualifiedCapabilityExecutionTarget` (`qualified_capability_execution_target.v1` only in production Python) |
| Binding providers (Marketplace, CodeCraft, host-available) | Construct target at bind time |
| Intake/dispatch/resume contracts | Embed target on runtime envelopes |
| Runtime delegates + handler registry | Consume target in-process |
| Unit/qualification tests | Fixtures mirror runtime handoff |

**Not found:** target repository; target `model_dump` to durable store; external/public API serialization of target as persisted artifact.

**Verdict:** **runtime handoff only** — no justified runtime v1→v2 compatibility obligation.

---

## 5 — Target migration decision

| Policy | |
|---|---|
| **Canonical write** | `qualified_capability_execution_target.v2` (schema identifier only) with required `execution_handler_id`, truthful `binding_provider_id`, opaque `execution_target_reference`, `qualified_subject_reference` |
| **Migration** | Single implementation wave — **no** v1 reader, mapper, fallback routing, or dual target models |
| **Routing** | `registry.resolve(target.execution_handler_id)` **only** |
| **Forbidden** | `if target.v2 … else …`; `try v2 fallback v1`; dual registries |

Historical v1 remains **historical source evidence** only, not a runtime compatibility branch.

---

## 6 — Deterministic target handle (CONFIGURE_EXISTING)

Configured opaque `execution_target_reference` is **deterministic** from immutable binding identity, e.g.:

```text
derive_marketplace_configured_tool_execution_target_reference(binding_operation_id)
→ marketplace-configured-tool:v1:{deterministic_correlation}
```

- Opaque, correlation-only, non-semantic.
- **No** `CapabilityIdentityKey` encoding in the string.
- **No** random UUID unless a proven platform invariant requires it (@ START_HEAD none found).

UCA legacy `marketplace-qualified-tool:v1:{handoff_id}` remains UCA adapter correlation (not CONFIGURE_EXISTING template).

---

## 7 — Blocker 22 (duplicate capability identity)

Placing `CapabilityIdentityKey` on both common intent and configured provenance creates **two copies** of the same semantic truth.

**Resolution:** store capability identity **exactly once** on common intent. Provenance carries **source-specific lineage only**.

**Forbidden:** reconciliation validator `assert common.capability_identity == provenance.capability_identity`.

---

## 8 — Exactly-one capability identity

Canonical owner:

```text
MarketplaceToolExecutionIntent.capability_identity: CapabilityIdentityKey
```

Single semantic intent; single provenance variant per record; identity is **common execution truth** (UCA and CONFIGURE_EXISTING execute one known logical Tool capability).

---

## 9 — UCA identity source

```text
MarketplaceQualifiedToolStage.selected_release
  → canonical discovery / logical identity
  → CapabilityIdentityKey
  → MarketplaceToolExecutionIntent.capability_identity
```

No second UCA capability identity type; not encoded in handoff/target reference strings.

---

## 10 — Configured identity source

```text
ConfiguredCapabilityExecutionSubject.capability_identity
  → MarketplaceToolExecutionIntent.capability_identity
```

No parsing from `configuration_ref`, `provider_id`, target reference, or operation string.

---

## 11 — Final provenance variants (thin, no identity copy)

**UCA** — `UcaMarketplaceToolExecutionProvenance`:

- `handoff_id`
- `resume_operation_id`
- qualified-subject correlation facts as needed (not duplicating `capability_identity`)

**CONFIGURE_EXISTING** — `ConfiguredMarketplaceToolExecutionProvenance`:

- configured decision correlation
- configuration binding correlation
- configured subject correlation
- adoption correlation only if durable intent truly requires it

**Must not contain:** `CapabilityIdentityKey`; provider objects; permission/authority claims.

---

## 12 — Durable intent vs execution target (compatibility distinction)

| Object | @ START_HEAD | Compatibility policy |
|---|---|---|
| **Execution target** | Not independently persisted | **No** runtime schema compatibility branch |
| **Marketplace Tool intent** | `DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository` + `ConditionalDocumentStore` | Legitimate **bounded v1 read → canonical v2 in-memory** (UCA historical rows); v2 canonical write; configured path writes v2 only |

Intent migration: read-only, bounded, explicit, UCA-only historical projection, non-authoritative, one-way. **One** `MarketplaceToolExecutionIntentRepository` semantic owner; **no** second partition unless mechanically proven required.

---

## 13 — Duplicate audit matrix

| Concern | Classification |
|---|---|
| Target v1→v2 runtime adapter | **ELIMINATE** (no evidence) |
| Target schema id `qualified_capability_execution_target.v2` | **SCHEMA EVOLUTION**, not mechanism |
| Common `capability_identity` | **CANONICAL OWNER** |
| Provenance copy of capability identity | **DUPLICATE / ELIMINATE** |
| UCA provenance | **THIN TYPED SOURCE PROVENANCE** |
| Configured provenance | **THIN TYPED SOURCE PROVENANCE** |
| Intent repository | **ONE CANONICAL OWNER** |
| Handler registry | **ONE CANONICAL OWNER** |

---

## 14 — Future implementation delta (post-audit)

```text
MODIFY QualifiedCapabilityExecutionTarget
  → required execution_handler_id
  → truthful binding_provider_id
  → deterministic opaque target ref (configured)

ATOMICALLY UPDATE producers, delegates, handler protocol, registry, tests

MODIFY MarketplaceToolExecutionIntent
  → capability_identity exactly once
  → typed provenance union (no identity in variants)

MODIFY durable intent repository
  → canonical v2 writes
  → bounded v1 UCA read projection

NO target compatibility runtime
NO duplicate capability identity
```

---

## 15 — STOP conditions

Return **STOP — ARCHITECTURE DECISION REQUIRED** if evidence shows:

- execution target is durably persisted or externally stable requiring compatibility;
- removing runtime target v1 support breaks a real persisted consumer;
- UCA vs configured paths require materially different capability identities in one intent;
- intent migration requires a second repository/partition authority;
- accepted handler/registry/activation architecture must change.

---

## 16 — Implementation exit criteria

Independent audit may accept implementation when: blockers **21** and **22** resolved per this lock; target inventory complete; atomic target migration specified; deterministic configured target handle locked; single `capability_identity` owner; provenance thin; intent durability policy distinct from target; **NEW SEMANTIC MECHANISM = 0**; prior invariants preserved.

**FRZ evidence (scoped, no PASS promotion):** **FRZ-TRC-11** OPEN; **FRZ-OWN-01/03/04/05**; **FRZ-CTR-01/02**; **FRZ-TYP-01**; relevant PLG/RPL; **FRZ-EXE-01/02**; relevant TEN; **FRZ-REG-01/02/03**; **FRZ-CMP-02/03/06/08** (intent schema distinction only).
