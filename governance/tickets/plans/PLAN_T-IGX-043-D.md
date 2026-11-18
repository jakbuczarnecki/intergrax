# PLAN T-IGX-043-D — EQO-A code wave FC-4: skill execution pin durability and KV durability attestation (A4-002)

Status: plan text; drafted 2026-11-16, amended 2026-11-17 (state-neutral wording; stale references to session artefacts replaced by the published records; facts re-checked at the train head `T`, section 15). Read-only planning (nothing implemented, no branch, no test run). The plan is not an implementation go: implementation waits for the ARCH-FREEZE gate `SCOPE` and the Operator's product-path go. Nothing here is applied or audited. It certifies nothing, promotes no FRZ criterion and changes no stage status.

Reading notation: `H` = `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (`origin/development` at drafting, merge of PR #604); `B` = `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (Campaign A baseline). Code lines below are at `H`; every FC-4 source file is byte-identical between `B` and `H` (`git diff --stat B H` over the six product files and the touched tests is empty: UNCHANGED). Full-SHA evidence is in `FACT_CHECK.md`. `T` = `05699b633cdb1318c215f6529e449420efc1382c` (`origin/arch-freeze/rw-train` at the amendment, contains PT-WA, merge of PR #642); the six product files and the touched tests are byte-identical between `H` and `T` (section 15).

## 1. Purpose and finding

| Field | Value |
|---|---|
| Finding | EQOA-A4-002 (MEDIUM, `CONFIRMED_DEFECT`, ARCH-FREEZE impact `CONDITIONS`), owner SKILL-X / durability owner; related HIST-F-DUR-01 (open residual), FRZ-PRD-07 (qualification honesty) |
| Defect | `KvSkillExecutionPinningStore.is_durable` and `DocumentStoreSkillExecutionPinningStore.is_durable` return a literal `True` (`intergrax/skills/execution_pinning_persistence.py@H:290-292` and `:387-389`). The production guard reads only that flag (`execution_pinning_durability.py@H:29`). So any `DistributedKVStore` or `ConditionalDocumentStore`, including an in-memory one, passes the guard in `production_mode=True` and the pins are lost at restart. |
| Why it is not allowed | The protocol docstring says `is_durable` means "pinning state survives process restart" (`execution_pinning_durability.py@H:17-19`); HIST-F-DUR-01 is a recorded residual, not a permission. C3 / OD-REM-03: PRODUCT/STRICT must be fail-closed. |
| Decisions that fix the design | DP-31 (default-False capability; Sqlite declares True; Redis declares True only through an explicit typed attestation), DP-24 (hybrid: the wiring layer fails closed on STRICT/PRODUCT, leaves default-deny), DP-11 (one wave), DP-44 (SKILL-X custody; see section 9). The Operator accepted the recommended answer to every draft question (OQ-1..OQ-6), as relayed to this session by the orchestrating session on 2026-11-16; the transcription of those answers is published as `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md` (an orchestrator transcription, not a signature; the question list `OPEN_QUESTIONS.md` stays a session artefact under `.tmp`). The design below is the decided one. |
| Master-plan node | `PT-FC4` (`WP-12/TICKETS_DRAFT.md:29`, `WP-12/DAG.md:97`); wave package W3 (AR-41). Hard prerequisite `SCOPE`; the node gates `RW` (`DAG.md:106`) and is audited by `PT-Q02`. |

## 2. Design (smallest design that lets the guard fail a non-durable store)

Principle: the durability claim moves from the pin store (a constant) to the backing store (a capability that defaults to False). The guard keeps its shape; it becomes strict about what it accepts.

### 2.1 Contract — `intergrax/distributed/contracts/kv_store.py`

Current code: `DistributedKVStore` is an ABC with four abstract methods and no durability member (`kv_store.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:11-76`); `DistributedKVStoreProvider` follows at `:79-89`.

Add (all additive; no abstract member is added, so the 38 existing subclasses, 36 of them test doubles, keep working unchanged):

```text
class KVDurabilityMechanism(str, Enum):            # closed set; extension needs a contract change
    REDIS_AOF_FSYNC_ALWAYS   = "redis_aof_fsync_always"
    REDIS_AOF_FSYNC_EVERYSEC = "redis_aof_fsync_everysec"      # decided (OQ-2a): both AOF modes are attestable

@dataclass(frozen=True, slots=True)
class KVDurabilityAttestation:                     # typed, explicit, operator-stated claim
    mechanism: KVDurabilityMechanism
    attested_by: str        # non-empty: reference of the party that states persistence is configured
    procedure_ref: str      # non-empty: repo path or ticket id of the documented attestation procedure
    # __post_init__: isinstance(mechanism, KVDurabilityMechanism); attested_by/procedure_ref are str and non-empty after strip -> ValueError otherwise

class DistributedKVStore(ABC):
    @property
    def survives_process_restart(self) -> bool:    # NON-abstract, default False
        return False

def kv_store_survives_process_restart(store: object) -> bool:
    return getattr(store, "survives_process_restart", False) is True    # duck-typed KV without the member => False; non-bool truthy values (1, "yes", a mock) => False
```

`survives_process_restart` is the capability named by DP-31; `KVDurabilityAttestation` is the typed attestation. The type lives in the contract module so that any backend (for example a Redis-compatible one) uses the same shape; only the Redis provider consumes it in this ticket. `kv_store_survives_process_restart` mirrors `document_store_survives_process_restart` (`intergrax/integrations/contracts/document_store_process_durability.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:21-26`).

### 2.2 Sqlite reference provider — `intergrax/distributed/providers/sqlite_kv_store.py`

Current code: `SqliteDistributedKVStore` opens a file database in WAL mode with `PRAGMA synchronous=FULL` on every connection (`sqlite_kv_store.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:15-32`). Change: add `survives_process_restart` returning `True` (DP-31: Sqlite declares True) with a docstring that states the claim is about the store type and a file path that is reused (decided, OQ-5: accepted as a reference-host limitation, no change to the reference persistence; the reference composition's default path is a fresh uuid file per process, `production_platform_persistence.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:47-52`). No behaviour change otherwise.

### 2.3 Redis provider and its composition root

Current code: `RedisKVStore.__init__(client, *, key_prefix="intergrax")` (`redis_kv_store.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:27-34`); the only product constructor call is `create_redis_integration` (`bundle.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:92`), and the module docstring forbids direct construction by application code (`redis_kv_store.py:8-9`).

Change:
- `RedisKVStore.__init__` gains a keyword `durability_attestation: KVDurabilityAttestation | None = None`. A non-`None` value that is not a `KVDurabilityAttestation` raises `TypeError` (a bare `True` is refused). Property `durability_attestation` returns it; `survives_process_restart` returns `self._durability_attestation is not None`. Default (no attestation) stays False, so every existing caller keeps its behaviour and is simply "not attested".
- `create_redis_integration(...)` and `create_redis_kv_store(...)` in `intergrax/integrations/providers/key_value_cache/redis/bundle.py` gain the same optional keyword and pass it to `RedisKVStore` (`bundle.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:58-105` and `:130-145`). This file is not in the master-plan row; it is the Redis composition root, without it no application can supply the attestation (a decided extension of the row, OQ-4; `FACT_CHECK.md` F-26); `bundle.py` is in the write set.
- How the attestation is checked (decided, OQ-3): it is declared, not probed. The code verifies only the type (a `KVDurabilityAttestation`, anything else is refused with `TypeError`) and that the fields are well formed (enum mechanism, non-empty `attested_by` and `procedure_ref`); it does not contact the Redis server. Correctness of the claim rests on the documented procedure and the audit review. Not added: reading an attestation from environment or config files, and any live probe (`INFO persistence`); a probe is a recorded follow-up that needs live Redis, which the C10 freeze does not allow.
- The attestation procedure is documented in the SKILL-X owner document `docs/project/architecture/SKILLS.md`, and it states: (i) the attestation shape (the typed `KVDurabilityAttestation` fields and the accepted mechanisms); (ii) that an attestation is a declared claim, not evidence of the Redis server's state (no probe, OQ-3); (iii) what evidence the attester cites in `procedure_ref`; (iv) how the audit reviews it. OPEN-ATT-1 is resolved by fallback (b) (`docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md` item 4, a published repository record): the procedure text states the attestation shape only. Who may attest stays an open item owned by the Operator; no new authority grant is recorded and the procedure names no authorised role. The procedure document (DP-31 consequence: "needs a documented attestation procedure"). The accepted mechanisms are `AOF_FSYNC_ALWAYS` and `AOF_FSYNC_EVERYSEC` (OQ-2a); the procedure states the loss window of `everysec`.

### 2.4 Pin stores — `intergrax/skills/execution_pinning_persistence.py`

Current code: both `is_durable` bodies are `return True` (`:290-292`, `:387-389`).

Change (two property bodies plus one import):
- `KvSkillExecutionPinningStore.is_durable` returns `kv_store_survives_process_restart(self._kv_store)`.
- `DocumentStoreSkillExecutionPinningStore.is_durable` returns True only when the backing store is a `ProcessRestartDurableDocumentStore` and its `survives_process_restart is True`. It does not call `document_store_survives_process_restart`, because that helper applies `bool(...)` (`document_store_process_durability.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:25`) and would accept a truthy non-bool; the shared helper is used by other owners (suspended operations) and is not edited here.
- The in-memory document store already reports False (`in_memory_document_store.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:83-85`) and the Postgres adapter True (`postgresql/adapter.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:49-51`), so the document-store half needs no provider change.

### 2.5 Guard — `intergrax/skills/execution_pinning_durability.py`

Current code: guard returns early when `production_mode` is False and raises `SkillResolutionError(DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG)` only when `not pinning_store.is_durable` (`:22-30`).

Change: raise unless `getattr(pinning_store, "is_durable", False) is True`. That refuses a store that has no `is_durable`, or returns a non-bool (this also covers an explicitly injected `pinning_store`, which today reaches the guard unchanged). The message constant stays byte-identical (existing tests match it). The `DurableSkillPinningCapability` docstring is tightened to say the value is derived from the backing store (DP-24: the wiring layer fails closed; the leaf default is deny). The LAB path (`production_mode=False`) is untouched: it remains the explicit, typed LAB posture of this wiring layer.

### 2.6 Not changed

- `intergrax/skills/skill_execution_pinning_wiring.py` (`:26-48`): KV stays authoritative when both backends are supplied (T-IGX-032-A, pinned in `tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py`); the guard call at `:44-47` already sees the derived flag. Consequence: if the KV store is non-durable and a durable document store is also supplied, STRICT is refused rather than silently falling back (decided, OQ-1). Note the master-plan glob `intergrax/skills/execution_pinning_*.py` does not match this file; no edit is planned for it.
- `intergrax/applications/_shared/harness_host_runtime.py` (`:354-356`, `:409-413`): the caller passes `production_mode`, `kv_store`, `doc_store` unchanged.
- Stored bytes, keys, partitions and the pin schema: unchanged. No migration of data.

### 2.7 Migration for existing callers

| Caller class | Effect | Action |
|---|---|---|
| LAB / `production_mode=False` | none | none |
| STRICT host with Sqlite KV (reference production persistence, `production_platform_persistence.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:53-56`) | stays accepted | none |
| STRICT host passing only an in-memory document store, or an in-memory/unattested KV | now refused with `SkillResolutionError` (intended) | host must supply a Sqlite KV, an attested Redis KV or a durable document store |
| Redis KV supplied to skill pinning in STRICT | refused until the operator passes an attestation | adopt `durability_attestation=` at the composition root |
| Test doubles that subclass `DistributedKVStore` (36 test classes at `H`) | unchanged (`survives_process_restart` is inherited False) | only tests that run `production_mode=True` against an in-memory double change (section 4.3) |
| Duck-typed KV not subclassing the ABC | treated as not durable | none (fail-closed) |

STRICT call sites outside the named tests that pass non-durable stores are not enumerated yet (the WP-2 count is UNVERIFIED; about 100 files reference `build_harness_host_runtime`). Step 0 of the order of work is a static inventory; it needs no test run.

## 3. Exact write set

Product paths (ACT-N11 / IIAC sensitive; `PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")`, `scripts/governance/iiac_common.py@B:22`):

```text
intergrax/distributed/contracts/kv_store.py
intergrax/distributed/providers/sqlite_kv_store.py
intergrax/distributed/providers/redis_kv_store.py
intergrax/integrations/providers/key_value_cache/redis/bundle.py
intergrax/skills/execution_pinning_persistence.py
intergrax/skills/execution_pinning_durability.py
```

Tests (non-product; new files marked N, edited files marked E):

```text
tests/unit/skills/test_skill_execution_pinning_durability_guard.py                    N
tests/unit/distributed/contracts/test_kv_durability_attestation.py                    N
tests/unit/distributed/test_sqlite_kv_store_durability.py                             N
tests/unit/distributed/test_redis_kv_store_durability.py                              N
tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py              E (1 test inverted)
tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py             E (5 tests moved to a durable double)
```

Docs: `docs/project/architecture/SKILLS.md` (attestation procedure section, English); `docs/project/maintainers/plans/SKILLS.md` only if `scripts/docs/check_docs_domain_pairs.py` requires the pair to move.
Governance: `governance/tickets/T-IGX-043-D.md`, `governance/tickets/plans/PLAN_T-IGX-043-D.md`.

Forbidden for this ticket: every other path under `intergrax/`, `agents/`, `applications/`, `scripts/`, `.github/`, `pyproject.toml`, `uv.lock`; the roadmap and the FRZ checklist; `docs/project/maintainers/qualification/**` (including the SKILL-X Batch C/D/E evidence files and any AUD-SKILLX custody record, DP-44); IIAC records; `skill_execution_pinning_wiring.py`; `harness_host_runtime.py`; `document_store_process_durability.py` (shared helper); the sibling KV or document-backed stores with a literal `is_durable` `True` (FACT_CHECK F-16; decided, OQ-2b: recorded separately, not fixed here).

## 4. Tests

### 4.1 Negative tests (a configuration that must now be refused). Red at `B`, green at the wave head; all refusals are `SkillResolutionError` with `DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG` unless stated

File `tests/unit/skills/test_skill_execution_pinning_durability_guard.py` (all `production_mode=True` through `resolve_skill_execution_pinning_wiring` unless stated):

| ID | Test name | Configuration | Expected |
|---|---|---|---|
| NT-1 | `test_strict_in_memory_kv_store_is_refused` | `kv_store=<in-memory DistributedKVStore double>` | refused (passes the guard at `B`) |
| NT-2 | `test_strict_kv_without_capability_member_is_refused` | `kv_store=<duck-typed object, not an ABC subclass, with the four methods>` | refused |
| NT-3 | `test_strict_in_memory_document_store_is_refused` | `document_store=<reference in-memory ConditionalDocumentStore>` (`survives_process_restart` False) | refused |
| NT-4 | `test_strict_document_store_declaring_false_is_refused` | `document_store=<double with survives_process_restart False>` | refused |
| NT-5 | `test_strict_redis_without_attestation_is_refused` | `kv_store=RedisKVStore(<fake client>)`, no attestation | refused |
| NT-6 | `test_strict_non_durable_kv_with_durable_document_store_is_refused` | `kv_store=<in-memory>`, `document_store=<durable double>` | refused; no silent fallback to the document store (decided, OQ-1) |
| NT-7 | `test_strict_explicit_non_durable_pinning_store_is_refused` | `pinning_store=KvSkillExecutionPinningStore(<in-memory>)` | refused |
| NT-8 | `test_strict_explicit_pinning_store_without_or_nonbool_is_durable_is_refused` | `pinning_store=<object with no is_durable>` and `<object whose is_durable returns "yes">` | refused (both) |

File `tests/unit/distributed/contracts/test_kv_durability_attestation.py`:

| ID | Test name | Configuration | Expected |
|---|---|---|---|
| NT-9 | `test_kv_capability_truthy_non_bool_is_not_durable` | KV double whose `survives_process_restart` returns `1` / `"yes"` / a `MagicMock` | `kv_store_survives_process_restart` is False; STRICT pinning refused |
| NT-10 | `test_document_store_capability_truthy_non_bool_is_not_durable` | document-store double whose `survives_process_restart` returns `"yes"` | `DocumentStoreSkillExecutionPinningStore.is_durable` False; STRICT refused |
| NT-11 | `test_attestation_rejects_empty_attested_by` | `KVDurabilityAttestation(mechanism=..., attested_by="  ", procedure_ref="x")` | `ValueError` |
| NT-12 | `test_attestation_rejects_empty_procedure_ref` | `procedure_ref=""` | `ValueError` |
| NT-13 | `test_attestation_rejects_non_enum_mechanism` | `mechanism="redis_aof_fsync_always"` (a string, not the enum) | `ValueError` or `TypeError` (one is fixed in the implementation; the test pins it) |

File `tests/unit/distributed/test_redis_kv_store_durability.py`:

| ID | Test name | Configuration | Expected |
|---|---|---|---|
| NT-14 | `test_redis_kv_store_rejects_untyped_attestation` | `RedisKVStore(client, durability_attestation=True)` and `=dict(...)` | `TypeError`; the store is not constructed |

Count: 14 negative tests (NT-9 and NT-8 parametrise several inputs; the count is by test function name).

Red-first expectation at `B` (the code before the change): every negative test is red. NT-1..NT-7 are red because the store is accepted by the guard; NT-8 is red because an object without `is_durable` raises `AttributeError` (not `SkillResolutionError`) and a truthy `"yes"` is accepted; NT-9..NT-14 are red because the new symbols or checks do not exist. The positive tests PT-1..PT-4 that use new symbols are red at `B` for the same reason; PT-3 and PT-4 may already pass. The mutation check for the auditor is in section 7.

### 4.2 Positive and lab tests (guard against an over-strict fix)

| ID | Test name | Expected |
|---|---|---|
| PT-1 | `test_strict_sqlite_kv_is_accepted_and_pin_survives_new_instance` (`tests/unit/distributed/test_sqlite_kv_store_durability.py`, `tmp_path`) | Sqlite KV accepted in STRICT; a pin written through one `SqliteDistributedKVStore` is read back through a second instance on the same file (restart simulation) |
| PT-2 | `test_strict_redis_with_typed_attestation_is_accepted` (fake client) | accepted; `durability_attestation` round-trips |
| PT-3 | `test_strict_durable_document_store_is_accepted` | accepted |
| PT-4 | `test_lab_mode_still_allows_in_memory_stores` | `production_mode=False` with in-memory KV or document store resolves |

### 4.3 Existing tests that pin the defect and must change (same ticket, inverted or moved to a durable double)

| Existing node | Why it changes | New form |
|---|---|---|
| `test_skill_x_batch_c_versioning_provenance_recovery.py::test_production_wiring_uses_kv_when_available` (`:440-446`) | STRICT plus in-memory KV currently expects durable | inverted: expects the refusal (this is NT-1's pin); a sibling case with a durable double expects `is_durable is True` |
| `test_skill_execution_pinning_wiring_store_precedence.py::test_both_backends_resolve_to_kv_store_t1`, `..._t2`, `..._t3`, `test_document_store_only_uses_document_pinning_t4`, `test_explicit_pinning_store_wins_t7` | STRICT plus in-memory doubles | keep their precedence assertions, run them on a test-local durable double that overrides `survives_process_restart` to True |

### 4.4 Regression scope (existing node IDs; all 31 verified to exist at `H`, 6 of them are the migrated nodes above)

Unchanged, must stay green (25):

```text
tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py::test_production_mode_requires_durable_pinning
tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py::test_lab_wiring_allows_in_memory
tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py::test_document_store_pinning_parity
tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py::test_wire_skill_execution_pinning_store_rejects_dual_backends
tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py::test_durable_pin_idempotent_and_recovery_without_resolution
tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py::test_lost_acknowledgement_idempotent_retry
tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py::test_cas_raises_before_write_fails_closed
tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py::test_cas_committed_ack_lost_idempotent_success
tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py::test_non_conditional_document_only_raises_type_error_t5
tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py::test_no_stores_in_production_raises_durable_required_t6
tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py::test_leaf_still_rejects_dual_backends_t8
tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py::test_reference_strict_composition_shape_resolves_kv
tests/unit/skills/test_p1_10_execution_binding.py::test_execution_binding_rejects_rebind_with_different_pack
tests/unit/skills/test_skill_x_batch_e_integrated_authority_tenant_qualification.py::test_ec8_same_execution_id_isolated_per_tenant
tests/unit/skills/test_skill_x_batch_e_integrated_authority_tenant_qualification.py::test_ec8_ack_loss_then_scope_verification_per_tenant
tests/unit/skills/test_skill_x_batch_d_trust_boundary_qualification.py::test_verified_historical_replay_when_catalog_matches_pin
tests/integration/marketplace/test_me15_marketplace_skill_composition_e2e.py::test_me15_bound_skill_composition_uses_canonical_execution_binding
tests/unit/distributed/contracts/test_distributed_kv_store_provider.py::test_distributed_kv_store_is_not_a_provider
tests/unit/distributed/contracts/test_distributed_kv_store_provider.py::test_distributed_kv_store_provider_matches_kv_store_property
tests/unit/distributed/test_bootstrap.py::test_bootstrap_registers_redis_provider
tests/unit/distributed/test_provider_registry.py::test_register_and_get_provider
tests/unit/integrations/providers/key_value_cache/test_redis.py::test_create_redis_integration_bundle_shares_one_client
tests/unit/integrations/providers/key_value_cache/test_redis.py::test_create_redis_key_value_cache_uses_injected_client
tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py::test_p2c_01_positive_http_admission_continuity
tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py::test_p2c_05_identity_and_scope_continuity
```

Whole-file regression (names verified at `H`): `tests/unit/skills/test_p1_10_execution_binding.py` (23 tests), the two edited skill files, `tests/unit/skills/test_skill_x_batch_d_trust_boundary_qualification.py`, `tests/unit/skills/test_skill_x_batch_e_integrated_authority_tenant_qualification.py`, `tests/unit/skills/test_version_pinning.py`, `tests/unit/distributed`. The P2-C file is the original victim of the pin wiring (T-IGX-032-A); it is expected unchanged because the strict reference host passes a Sqlite KV (what its fixtures pass is UNVERIFIED, hence the node pair above).

Out of reach without Docker (operator freeze C10, `intergrax-env-vendor-qualification.md`): `tests/integration/distributed/test_redis_kv_store.py` needs a live Redis; it is not part of this contract and no live Redis evidence is produced (see section 6).

## 5. Test commands (one process at a time; none run when drafted)

Machine budget: one `uv run` per step, under the shared lock `D:/Projekty/_wt/_locks/pytest.lock` as in the batch briefs; log under `.tmp/session/<task-id>/`.

```text
# S1 red-first, at the pre-change commit of the implementation branch (the new tests only)
uv run pytest tests/unit/skills/test_skill_execution_pinning_durability_guard.py tests/unit/distributed/contracts/test_kv_durability_attestation.py tests/unit/distributed/test_sqlite_kv_store_durability.py tests/unit/distributed/test_redis_kv_store_durability.py -p no:cacheprovider -q

# S2 green after the change: the same command

# S3 regression (one pytest command, many paths)
uv run pytest tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py tests/unit/skills/test_p1_10_execution_binding.py tests/unit/skills/test_skill_x_batch_d_trust_boundary_qualification.py tests/unit/skills/test_skill_x_batch_e_integrated_authority_tenant_qualification.py tests/unit/skills/test_version_pinning.py tests/unit/distributed tests/unit/integrations/providers/key_value_cache/test_redis.py tests/integration/marketplace/test_me15_marketplace_skill_composition_e2e.py tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py -p no:cacheprovider -q

# S4 static
uv run ruff check <the six product files and six test files>
uv run ruff format --check <same>
uv run pyright <the six product files and the new test files>
uv run python scripts/docs/check_docs_domain_pairs.py
```

Wave-level gates that FC-4 inherits and does not run on its own (they belong to the single EBH-7 / COMPAT-X re-pin and the wave audit): `tests/qualification/compat_x` (a full run took about 27 minutes in T-IGX-041-O), the EBH-7 F10 classifier, ROADMAP-REPLAY-X product-delta gates.

## 6. Order of work inside the ticket

0. Static inventory (no test run): every STRICT call site that reaches `resolve_skill_execution_pinning_wiring` or constructs the two pin stores, and which KV or document store it passes (`git grep` over `intergrax/`, `applications/`, `agents/`, `tests/`). Output: a table in the implementation checkpoint; any host that passes a non-durable store is a finding for the owner, not silently adapted. The inventory is taken on today's tree and is repeated on the integrated wave tree before the wave audit: FC-1 (its NT-FC1-20/21 and construction sweep), FC-2 and FC-3 build STRICT hosts and test fixtures that may carry an in-memory KV, and FC-4's guard will refuse them. Such fixtures are resolved by the integrated-tree census the wave adopts (delegate review item X-4); FC-4 does not edit other tickets' test files.
1. Tests first: the four new test files (S1 must show the expected red set at the pre-change commit).
2. Contract (`kv_store.py`), then Sqlite, then Redis plus `bundle.py`.
3. Pin stores and guard (`execution_pinning_persistence.py`, `execution_pinning_durability.py`).
4. Migrate the six existing nodes (section 4.3).
5. Attestation procedure in `SKILLS.md` (procedure text may not claim any live Redis qualification).
6. S2, S3, S4; fill the implementation checkpoint with real output.
7. Hand over to the wave: the ticket joins the single wave train (`RW`); it is audited at the wave head, not merged separately.

## 7. Independent audit design (C3: separate non-implementing session, exact SHA)

- Exact SHA: the wave head `RW` recorded by `PT-Q02` (`WAVE_SCHEDULE`/`DAG.md:113`), not this ticket's branch tip. The auditor did not implement or review FC-4 (`intergrax-independent-audit`).
- The auditor replays: (a) NT-1..NT-14 and PT-1..PT-4 on the wave head; (b) the same tests on `B`-equivalent code to confirm they are red there (red-first), using a detached worktree under `_wt`, not the main checkout; (c) a mutation check: restore `return True` in each of the two `is_durable` bodies, expect NT-1..NT-7 red; revert the guard to `if not pinning_store.is_durable`, expect NT-8 red; make `kv_store_survives_process_restart` or the document-store check use `bool(...)`, expect NT-9 and NT-10 red; make the Sqlite property return False, expect PT-1 red; (d) a diff reconciliation of the six product paths against this plan's write set (any further product path is a finding); (e) the migration table of section 2.7 against the inventory from step 0; (f) that the attestation procedure text exists, names its limits, and claims no live-Redis proof.
- Not claimed by this plan: independent verification, PASS, closure of A4-002, FRZ-PRD-07 promotion.

## 8. ACT-N11 / IIAC consequence

- Six product-path files change (section 3). Any product-path change after the activation commit breaks the IIAC activation until the single RB-2 re-pin (`PT-Q03` at `FCS`), per ACT-N11; FC-4 has no separate re-pin. Merging FC-4 alone would multiply re-pins (`WP-2/PLAN.md:21` and `:103`; DP-11). Therefore FC-4 merges only inside the wave train.
- `tests/` and docs are separate: six test files and one or two docs files do not count as product paths. `governance/tickets/` files are not product paths.
- EBH-7 F10 C4 classification: whether the six paths are already classified is UNVERIFIED. T-IGX-041-O shows that unclassified changed paths make `C4-UNCLASSIFIED` fail, and the delegate review found the same exposure for FC-2, which adds a new product file `intergrax/applications/_shared/wiring_posture.py`. FC-4 adds no new product file: all six paths already exist, so the only open point is whether each is classified today. That is read by the wave's classifier probe at the wave head (no run by the FC-4 implementer). If the probe reports UNKNOWN for an FC-4 path, the fix is a test-side classifier entry of the 041-O kind, carried by the wave's single EBH-7 re-pin (likewise for `wiring_posture.py`), not by FC-4's write set unless the Operator amends it. COMPAT-X persisted inventory lists `kv_store.py`, `sqlite_kv_store.py` and `redis_kv_store.py` as the `distributed.kv_entry` family (`tests/qualification/compat_x/_compat_x_persisted_inventory.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:487-501`); stored bytes do not change, so a disposition of "no persisted-shape change" is expected (UNVERIFIED until the probe).
- SKILL-X accepted certification delta is recorded as "1 file (skills/skill_execution_pinning_wiring.py)" (`scripts/qualification/pdc_x_requirements.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:93`); FC-4 widens the SKILL-X product delta by two skill files (DP-44 section 9).

## 9. DP-44 overlap (SKILL-X custody)

DP-44 settles AUD-SKILLX as a byte-preserving publication with a custody statement and G-SKILL as a scoped limitation. FC-4 changes the SKILL-X Batch C durability guard after that custody position. Rules for this ticket: (1) no edit to the AUD-SKILLX artefact, its custody statement or the SKILL-X Batch C/D/E evidence documents; (2) the wave audit states the new SKILL-X product delta in its own words and does not extend the old SKILL-X certification by reference; (3) the HIST-F-DUR-01 residual is recorded by the auditor, not closed by this ticket's text.

## 10. ARCH-FREEZE impact and what closes it

| Item | Impact | What closes it (conditions, not claims) |
|---|---|---|
| EQOA-A4-002 | `CONDITIONS` (dossier `FINDINGS.jsonl` line 11; `WP-2/PLAN.md:117`) | NT-1..NT-14 green at the wave head and red at `B`; the six migrated nodes green in their new form; independent audit at the exact wave SHA with the mutation check; the attestation procedure documented (DP-31 consequence); the decided answers OQ-1..OQ-6 followed |
| HIST-F-DUR-01 | residual of T-IGX-032-A plan, R1 | disposed by the audit record, not by this ticket |
| FRZ-PRD-07 | contribution only | this ticket adds evidence for the durability guard; it does not promote the row; the row's other gaps (A3-027, strict-path evidence) are outside FC-4 |
| Sibling stores with the same literal `True` | wider than the dossier finding | decided (OQ-2b): FC-4 fixes the two skill pin stores only and provides the reusable capability and helper. A4-002 closure condition wording (kept verbatim): skill pin stores, store capability only; the reference host does not recover pins across a restart; the siblings (27 literal-`True` `is_durable` bodies in product code per the delegate review count; this plan's own survey of seven files found 14 KV or document-backed ones, FACT_CHECK F-16) remain open. Record and owner: the siblings become rows of the Enterprise Debt & Exceptions Register (DP-46..DP-53); each row's accountable `owner_ref` is the owning stage of that store family (profile resolution, attempt lifecycle, execution terminal, delegated correlation, integration configuration), assigned under DP-47/DP-48 (no ownerless row); creation of the rows is carried by the register batch `PT-D05` (`WP-12/TICKETS_DRAFT.md`). No individual is named here. The guard class stays fail-open on the siblings and `CONDITIONS` remains for the family until those rows are disposed |
| Live Redis durability | not provable here | the attestation is a declared claim; live evidence belongs to PROD-Q (no Docker or vendor work under the C10 freeze) |

## 11. Risks

1. Hidden STRICT callers with a non-durable store will start failing (inventory in step 0; unknown count). Mitigation: inventory first; refusal is the intended behaviour; hosts that need Redis adopt the attestation.
2. The attestation is operator-stated and unchecked against the server (decided, OQ-3). It can be wrong. Mitigation: typed non-empty fields, procedure reference, audit record; a probe is a recorded follow-up.
3. `SqliteDistributedKVStore` declaring True is true of the store type only; the reference composition uses a fresh uuid file per process by default (OQ-5), so recovery across a real restart needs an explicit `db_path` (accepted limitation, OQ-5; stated in the procedure text and the property docstring).
4. `bundle.py` is outside the master-plan row; it is a necessary extension (section 2.3). The Operator accepted it (OQ-4); without it Redis could not attest through the composition root.
5. Whole-file reruns of `tests/unit/skills` take time; the commands list explicit files to stay within the machine budget.
6. The `tests/qualification/trace_x/_trace_x_p5_support.py:400-410` inventory names the two pin store classes by path; the plan changes class bodies only, not names or paths (UNVERIFIED that the gate is hash-free).

## 12. Rollback

Revert the FC-4 commits inside the wave branch before merge, or revert the wave merge after merge. No data, key or schema change was made, so no data rollback exists. A revert restores the literal-`True` defect and removes the attestation parameter, so no committed caller may use `durability_attestation=` outside the tests that are reverted with it. Reverting a merged product-path change does not restore the IIAC activation; the re-pin must be redone (WP-12 `AR-41` rollback note).

## 13. Effort estimate

| Part | Size | Note |
|---|---|---|
| Contract + Sqlite + Redis + `bundle.py` | S–M | additive; one new enum, one dataclass, three small edits |
| Pin stores + guard | S | two property bodies, one guard line |
| New tests (four files, 18 tests) | M | doubles for KV and document store, fake Redis client |
| Migration of six existing nodes | S | durable test-local doubles |
| STRICT call-site inventory (step 0) | M | unknown size; static only |
| Attestation procedure doc | S | one section; pair check |
| Independent audit (wave-level share) | M | includes the mutation check and the red-at-`B` replay |
| Total for the batch | M (consistent with `WP-2/PLAN.md:89` effort M; the sibling stores are not part of it, OQ-2b) |

## 14. Prerequisites before implementation starts

1. Milestone `M1` (PDC P9 closed) and the wave scope freeze `SCOPE` (`DAG.md:67`, `:85`); authorization AR-41 for the product paths (`WP-12/AUTHORIZATION_REQUESTS.md:83`; this directory's `AUTHORIZATION_REQUESTS.md` lists what FC-4 itself needs). Record of the Operator's go: transcription item 8 (granted, effective when `M1` holds) and addendum 1 (`M1` treated as confirmed by the Operator for this program, `PDC_INDEX.md` not edited by the program), both in `OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md`; gate `SCOPE` and the product-path go are read from their own records at implementation start.
2. Ticket approval line: the strict form of `scripts/governance/ticket_approval.py` (delegate form `Approved: operator-delegate @ <ISO>` with the `Delegation:` and `Delegate session:` lines directly below, plus an `Implementer session:` field in the ticket that differs from the delegate session) or the personal form. A plan approval is not an implementation go. Under the fallback (b) of OPEN-ATT-1 the ticket is approvable without a rule on who may attest; that item stays open with the Operator.
3. Nothing further on design: OQ-1..OQ-6 are decided (see `OPEN_QUESTIONS.md`).
4. Branch creation and the test runs of section 5 (see `AUTHORIZATION_REQUESTS.md`).

## 15. Facts at the train head (amendment of 2026-11-17)

Reading: `H` and `T` as defined in the header (`T` = `origin/arch-freeze/rw-train` `05699b633cdb1318c215f6529e449420efc1382c`; `origin/development` at the check was `2727a2c2364a97492202b931375772a400279a9b`).

| Item | Fact at `T` (and at `origin/development`) |
|---|---|
| Six product files (section 3) | byte-identical between `H` and `T` and between `H` and `origin/development` (`git diff --stat H T -- <paths>` empty); every line reference of sections 2.1 to 2.5 holds |
| Edited tests, regression files, `tests/unit/distributed`, `tests/unit/skills`, key-value-cache tests, P2-C and ME-15 files, COMPAT-X persisted inventory | identical between `H` and `T`; the 31 regression node ids of section 4.4 still exist |
| `harness_host_runtime.py` (not edited here) | PT-WA changed two hunks at lines 713 and 753 (environment spec argument and internal composition); the references `:354-356` and `:409-413` of section 2.6 are unchanged |
| `docs/project/architecture/SKILLS.md` | six lines added since `H` (new section "Failure, recovery and lifecycle ownership" before "Go deeper"); the attestation procedure section of section 3 is added beside it, no overlap |
| Other PT-WA changes under `intergrax/skills/` | only `intergrax/skills/registry/tool_requirements.py`; not in the write set, no effect on the pin stores or the guard |
| Subclasses of `DistributedKVStore` | the same count at `H` and `T` by the same regular expression (38 classes, 35 of them under `tests/`); section 2.7 figures (38 / 36) are the plan's own count and are unchanged by PT-WA |
| File overlap with PT-WA, PT-WB and the other train tickets (RW-PLAN section (b)) | none: the PT-FC4 row and column of the overlap matrix are empty; PT-FC4 sits in lane 2 after PT-WA and PT-WB. PT-WB is not merged at `T` |
| Step 0 inventory (section 6) | not taken here; it is repeated on the integrated wave tree as section 6 states |
