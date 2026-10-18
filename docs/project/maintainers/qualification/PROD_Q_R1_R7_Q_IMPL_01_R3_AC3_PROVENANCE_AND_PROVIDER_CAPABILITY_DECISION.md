# PROD-Q-R1-R7-Q-IMPL-01-R3-AC3-PROVENANCE-AND-PROVIDER-CAPABILITY-DECISION

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** |
| **Date** | 2026-10-18 |
| **Branch** | `development` |
| **START_HEAD / code evidence SHA** | `cae02d4cc6520c81422da511e4efb3b2044cc3ea` (`fix(gca): R2 strict external-work composition recovery (IMPL-01-R2)`) |
| **Parent** | **PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01** — **NOT CLOSED** (R2 recovery landed; qualification proofs deferred) |
| **Preserved (not reopened)** | [`PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_ADMISSION_ADR_01.md`](PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_ADMISSION_ADR_01.md) **A+B** · [`PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_OWNERSHIP_RECOVERY_CORRECTIVE_ADR_01.md`](PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_OWNERSHIP_RECOVERY_CORRECTIVE_ADR_01.md) · [`PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md`](PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md) **Option B** · [`PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_STRICT_COMPOSITION_AUTHORITY_DECISION_R2.md`](PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_STRICT_COMPOSITION_AUTHORITY_DECISION_R2.md) (OD-R2-01…04 ratified @ `6afff15e060cfd8247d64ab19b1415f91151a030`) |
| **Roadmap** | [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) — **PROD-Q-R1-R7-Q IN PROGRESS / NOT QUALIFIED**; no stage promotion |
| **Freeze checklist** | [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) — **no global FRZ PASS** |
| **Production delta (this task)** | **0** (documentation only) |
| **Recommended child** | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01-R3-AC3-PROVENANCE-RECOVERY` |
| **Does not close** | **IMPL-01** · **IMPL-01-R2** · **R7-Q** · **PROD-Q** · **Package P** · any global **FRZ-*** PASS |

**Purpose:** Corrective architecture decision for post-R2 audit findings **R3-F01…03** @ `cae02d4…` — AC3 factory provenance after host dependency injection, provider capability evidence, and typed production durability boundary — without new registry overlay, second resolver authority, or durable backend implementation.

---

## 1. Git baseline and evidence continuity

| Check | Result |
|-------|--------|
| Branch | `development` |
| `HEAD` @ analysis | `cae02d4cc6520c81422da511e4efb3b2044cc3ea` (= required **START_HEAD**) |
| `origin/development` | `cae02d4cc6520c81422da511e4efb3b2044cc3ea` (in sync) |
| Ancestry | `cae02d4…` is ancestor of `HEAD` (equality) |
| Working tree @ task start | clean |
| Production / test code changed in this task | **0** |
| Test execution in this task | **0** (`pytest` / `uv run` forbidden) |

**Evidence SHA for this decision document:** resolve after land:

`git log -1 --format=%H -- docs/project/maintainers/qualification/PROD_Q_R1_R7_Q_IMPL_01_R3_AC3_PROVENANCE_AND_PROVIDER_CAPABILITY_DECISION.md`

Historical **CLOSED** statuses (P2, P3, TRACE-X-CERT, etc.) are lineage only — **not** current proof for IMPL-01-R3.

---

## 2. Closed-world owner / contract / caller inventory

### 2.1 Sanctioned strict external-work pipeline (unchanged intent)

```text
Production admission + RuntimeRevision + materialization_artifact_digest
  → RosterAgentContractAuthority (optional on bundle; validated in registry_projection)
  → RuntimeAgentFactoryResolver (single chain; host decoration on AC3 keys only)
  → MaterializedRegistryProjection (derived-only; no overlay)
  → ExternalContractorAdapterAgent (Tier-2 reflex)
  → SanctionedExternalWorkOperationPort → GovernedExternalWorkOrchestrator
  → GovernedExternalWorkHostStoreBundle + durable ProviderInvocationStore
  → GovernedProviderInvocationDispatchGate
```

### 2.2 Semantic owners

| Concern | Semantic owner | Composition / materialization owner | Canonical symbols |
|---------|----------------|-------------------------------------|-------------------|
| Factory from package + revision | Agent Distribution / AP-10 | `VenvBundleRuntimeAgentFactoryResolver` · `build_production_runtime_agent_factory_resolver` | `RuntimeAgentFactoryResolver.resolve_factory` · `InMemoryRuntimeAgentFactoryResolver` (non-prod) |
| Registry materialization | Applications shared | `build_registry_projection` · `build_application_registry` | `RegistryProjectionInputBundle` · `MaterializedRegistryProjection` |
| Contract metadata @ materialization | AC-3 / AC-5 | `RosterAgentContractAuthority.contract_for_binding` | `intergrax/applications/_shared/roster_agent_contract_authority.py` |
| AC3 host port injection (strict) | GCA host composition | `reproject_registry_with_strict_sanctioned_external_work` · `create_governed_contractor_backend_app` | `GovernedContractorHostRuntimeComposition` · `build_external_contractor_adapter_factory` |
| Resolver decoration seam | Serving handoff + projection | `ServingRuntimeHandoff.runtime_agent_factory_resolver` | `establish_serving_runtime_handoff` · `HostOverridingRuntimeAgentFactoryResolver` |
| Provider discover + pin | Integration + operator settings | `resolve_strict_external_work_provider_capabilities` | `ExternalWorkIntegration.discover` · `GovernedContractorBackendSettings.strict_external_work_provider_*` |
| Orchestrator feature flags | GCA orchestrator policy | `build_governed_external_work_production_runtime` | `ExternalWorkProviderCapabilities` |
| GCA host persistence | GCA orchestrator | `GovernedExternalWorkHostStoreBundle` | `GovernedExecutionStore` · F02 follow-on on same store |
| PIS durability gate | `production_external_work_composition` | `ProviderInvocationStore.is_durable` | existing strict gate L301–307 |

### 2.3 Callers @ `cae02d4…` (R3-relevant)

| Caller | Callee | Finding |
|--------|--------|---------|
| `create_governed_contractor_backend_app` (`factory.py` L314–318) | `reproject_registry_with_strict_sanctioned_external_work` | R3-F01 reprojection path |
| `reproject_registry_with_strict_sanctioned_external_work` | `build_strict_sanctioned_external_work_runtime_agent_factory_resolver` | Supplies `HostOverridingRuntimeAgentFactoryResolver` |
| `build_strict_sanctioned_external_work_runtime_agent_factory_resolver` | `build_governed_contractor_agent_builders` → `host_factory` | **Substitutes** delegate factory |
| `HostOverridingRuntimeAgentFactoryResolver.resolve_factory` | returns `host_factory` when key ∈ `factory_keys` | **Skips** `delegate.resolve_factory` |
| `build_application_registry` (`wiring.py` L547–551) | `revision_bound_resolver.resolve_factory` | **Authoritative** materialization invoke site |
| `build_registry_projection` (`registry_projection.py` L472–487) | `_validate_factory_authority` + `build_application_registry` | Provenance validation partial |
| `resolve_strict_external_work_provider_capabilities` | `project_external_work_provider_capabilities` | R3-F02 hardcoded flags |
| `build_strict_external_work_host_session_resources` | `validate_production_governed_external_work_host_store_bundle` | R3-F03 `getattr` durability |
| `establish_serving_runtime_handoff` (`serving_runtime_handoff.py` L147–180) | binds `projection_resolver` to handoff | Identity check vs colocated unit resolver |

---

## 3. Findings

### R3-F01 — AC3 factory provenance (IN-SCOPE BLOCKER)

**Observed @ `cae02d4…`:**

1. `HostOverridingRuntimeAgentFactoryResolver.resolve_factory` (`host_overriding_runtime_agent_factory_resolver.py` L49–51) returns `self._override.host_factory` **without** calling `self._delegate.resolve_factory` when the `(package_digest, builder_key, factory_path)` key matches.
2. `host_factory` is minted in `build_strict_sanctioned_external_work_runtime_agent_factory_resolver` (`strict_sanctioned_external_work_registry_projection.py` L75–80) via `build_governed_contractor_agent_builders(host_runtime)[ExternalContractorAdapterAgent]` — a **host-process** closure, not the artifact-scoped callable from `VenvBundleRuntimeAgentFactoryResolver`.
3. Factory **authority fingerprints** in `registry_projection.py` (`_factory_authority_fingerprint` L174–183, `_registry_factory_wiring_digest` L230–238) record roster/manifest `factory_reference` metadata only — they do **not** prove which callable ran at materialization.
4. `_validate_factory_authority` (`registry_projection.py` L339–359) validates trusted `package_digest` and resolvable `factory_reference` but does **not** invoke `resolve_factory` — override bypass is invisible to validation.
5. `RosterAgentContractAuthority` (`roster_agent_contract_authority.py` L87–99) remains authoritative for **contract** metadata at `_register_binding`; it does not compensate for factory implementation substitution.

**Invariant (required):** For every enabled roster entry, materialization must invoke `RuntimeAgentFactoryResolver.resolve_factory(runtime_revision=…, package_digest=…, factory_reference=…)` such that the returned `CanonicalAgentFactory` is the **artifact-authorized** implementation for that triple, bound to `runtime_revision.installed_agent_package_digests` and `materialization_artifact_digest` alignment (`registry_projection.py` L398–405, L462–468).

**Host DI constraint:** Host may supply **dependencies** (sanctioned port, integration, authorization boundary, `require_sanctioned_external_work_mutations`) **only after** authoritative factory resolution. Host DI must **not** replace the artifact factory with an alternate builder map entry.

**Proof:** Post-authoritative injection **is possible** without new authority:

- `delegate.resolve_factory(...)` establishes artifact provenance (`venv_bundle_runtime_agent_factory_resolver.py` — import isolation + cached factory per digest/reference).
- A **typed wrapper** `CanonicalAgentFactory` delegates `(ctx, binding)` to the authoritative factory, then applies `StrictExternalWorkHostAgentInjection` (frozen host ports) via a Tier-2-local typed mutator (e.g. `inject_sanctioned_external_work_host_dependencies` on `ExternalContractorAdapterAgent`) or fail-closed if agent class mismatch.
- `RosterAgentContractAuthority` and `materialization_artifact_digest` checks remain on the **same** `RegistryProjectionInputBundle` as today (`strict_sanctioned_external_work_registry_projection.py` L102–112).

**Classification:** Current `host_factory` short-circuit is **not** provenance-preserving. This is a **materialization-owner correction**, not a new resolver authority — **does not** trigger STOP (no new semantic owner).

**Rejected alternatives:**

| Alternative | Rejection |
|-------------|-----------|
| Keep returning `host_factory` without delegate | Breaks artifact factory provenance; evidence digest misleads auditors |
| Second registry / overlay | Forbidden by R2 / ADR-01 |
| `compose_builders` on harness for strict | Forbidden (`harness_registry_authority` REVISION_BOUND) |
| Post-activation agent substitution in router | Forbidden |
| Infer factory from `agent_builders` map alone | Host map is not package digest authority |

### R3-F02 — Provider capability provenance (IN-SCOPE BLOCKER)

**Observed @ `cae02d4…`:**

`project_external_work_provider_capabilities` (`resolve_strict_external_work_provider_capabilities.py` L35–49) sets:

- `supports_create=True` — **unconditional**
- `supports_idempotency=True` — **unconditional**

`ExternalWorkCapability` (`intergrax/contracts/external_work.py` L440–452) has **no** `CREATE` or `IDEMPOTENCY` tokens. Descriptor pin validates `provider_id` + `descriptor_digest` only (`resolve_strict_external_work_provider_capabilities` L87–96).

**Analysis:**

| `ExternalWorkProviderCapabilities` field | Mandatory integration contract? | Discoverable via `ExternalWorkCapability`? | Provable from current pin + descriptor @ `cae02d4`? |
|------------------------------------------|--------------------------------|--------------------------------------------|-----------------------------------------------------|
| `supports_create` | `ExternalWorkIntegration.create_work` required on Protocol (`intergrax/integrations/contracts/external_work.py` L76–77) — **behavioral** obligation, not descriptor token | **No** | **No** — hardcoded `True` is not evidenced by descriptor or pin |
| `supports_idempotency` | Protocol docstring L13–15: mutating methods must honour `idempotency_key` | **No** | **No** — hardcoded `True`; quote-first check (L102–110) is vacuous |
| `supports_quote` / `supports_accept` | Quote-first profile | `QUOTE_FIRST` | **Yes** when token present |
| `supports_cancel` | Operation surface | `CANCELLATION` | **Yes** when token present |
| `supports_status_polling` | Reconciliation | `ASYNC_EXECUTION` or `TIMELINE` | **Partial** — mapped from descriptor |
| `supports_native_provider_request_id` | Async correlation | `ASYNC_EXECUTION` | **Partial** |
| `supports_tool_logs` | Evidence | `EVIDENCE_REFS` | **Partial** |
| `supports_receipt_timeline` | Timeline | `TIMELINE` or `EVIDENCE_REFS` | **Partial** |
| `supports_payment_state` | Commercial | `QUOTE_FIRST` (proxy) | **Partial** — semantic coupling |
| `supports_human_wait_state` | HITL | `HUMAN_CONTINUATION` | **Partial** |

**Downstream consumers:** `GovernedExternalWorkOrchestrator` (`orchestrator.py` L303–304) gates on `supports_create`; ERL bridge (`external_effect_contracts.py` L73–76) gates idempotency on `supports_idempotency`. Overstated capabilities break fail-closed semantics.

**Recommended contract-preserving design (single):**

1. **Descriptor-mapped flags** — only tokens in `ExternalWorkCapability` via explicit table in `project_external_work_provider_capabilities` (already partially done).
2. **Integration invariant flags** — new typed helper in `intergrax/contracts/` (e.g. `external_work_integration_mandatory_orchestrator_invariants() -> ExternalWorkIntegrationOrchestratorInvariants`) documenting GEC-2 Protocol L13–15 and L76–77 as **admission invariants**, set `supports_create` / `supports_idempotency` **only** when strict path has admitted a value typed as `ExternalWorkIntegration` **and** operator pin succeeded — **not** from reflection, method presence, or fixture.
3. **Fail-closed:** If `QUOTE_FIRST` ∈ descriptor.capabilities, require mapped `supports_quote`, `supports_accept`, and invariant `supports_idempotency`; if invariant layer not admitted → `StrictExternalWorkProviderResolutionError`.
4. **Pin:** Continues to prove identity + descriptor bytes only; **does not** prove create/idempotency — those come from invariant layer + integration admission.

**Operator decision (if rejected):** Add `ExternalWorkCapability.CREATE` and `ExternalWorkCapability.IDEMPOTENCY` to descriptor schema — **contract expansion**; requires explicit ratification before implementation.

**Rejected:** Hardcoded `True`; inferring from `create_work` method presence; PIS/GER inference; `quote_first_partner_capability_fixture` on strict path.

### R3-F03 — Durability declaration vs qualification (IN-SCOPE BLOCKER)

**Observed @ `cae02d4…`:**

- `GovernedExternalWorkHostStoreBundle.is_durable` (`governed_external_work_host_stores.py` L31–36) uses `getattr(store, "is_durable", False)` on store protocols that **do not** declare `is_durable` on `GovernedExecutionStore` / `ProofReceiptStore` / etc. (`stores.py` L49–76).
- `InMemory*` implementations return `is_durable is False` (e.g. L106–107); `FilesystemHostStore` returns `False` (L272–273) — demo only.
- `validate_production_governed_external_work_host_store_bundle` enforces bundle gate on strict wiring (`strict_external_work_host_wiring.py` L69) but **no qualified production durable implementation** exists in-tree.
- R2 OD-R2-03: **`is_durable` alone insufficient** — restart / corruption / replacement evidence required for qualification.

**Declared vs audited:**

| Layer | What it means @ `cae02d4` |
|-------|---------------------------|
| **Declared** | Runtime boolean on concrete store classes + bundle aggregate |
| **Audited production guarantee** | **NOT PROVEN** — no restart persistence qualification for GCA bundle; F02 continuity depends on same store |

**Minimal typed production durability contract (boundary only):**

1. Extend each GCA store `Protocol` in `stores.py` with `@property def is_durable(self) -> bool` (mirror `ProviderInvocationStore` pattern).
2. `GovernedExternalWorkHostStoreBundle` implements `GovernedExternalWorkHostStoreBundlePort` (`intergrax/contracts/governed_external_work_host_store_bundle.py`) using **typed** member access — remove `getattr`.
3. Add `ProductionDurableGovernedExternalWorkHostStoreBundle` qualification port (in `intergrax/contracts/` or qualification module) requiring evidence hooks: `restart_persistence_class`, `corruption_detection_class`, `backend_replacement_profile` — **declared only** in R3; implementation + proof in qualification child.
4. Strict composition: retain `validate_production_governed_external_work_host_store_bundle` + symmetric check in `production_external_work_composition.py` (as R2 blueprint) — startup fail-closed when not durable.

**Production qualification criteria (evidence, not implementation here):**

- Restart persistence of execution state, F02 follow-on records, GER-relevant fields.
- Incomplete finalization semantics per F02 Option B after provider physical effect.
- Corruption detection on read (typed parse errors → fail-closed).
- Backend replacement per `FRZ-RPL-*` without semantic drift.

**Blocker classification:** Missing qualified durable GCA backend remains **IN-SCOPE BLOCKER** for strict production qualification @ `cae02d4…`.

---

## 4. Current vs proposed AC3 materialization graphs

### 4.1 Current (@ `cae02d4…` — deficient provenance)

```text
ServingRuntimeHandoff.runtime_agent_factory_resolver  (= VenvBundle delegate D)
create_governed_contractor_backend_app
  └─ reproject_registry_with_strict_sanctioned_external_work(bundle, D, host_runtime)
        └─ HostOverridingRuntimeAgentFactoryResolver(delegate=D, override.host_factory=H)
              └─ build_registry_projection(factory_resolver=decorated)
                    └─ build_application_registry
                          └─ resolve_factory on AC3 key → returns H (never calls D.resolve_factory)
```

**Negative:** `registry_factory_wiring_digest` implies package factory authority; materialization uses host closure `H`.

### 4.2 Proposed (R3 recovery target)

```text
D = authoritative VenvBundle RuntimeAgentFactoryResolver (unchanged)
J = StrictExternalWorkHostAgentInjection (frozen ports from GovernedContractorHostRuntimeComposition)

HostOverridingRuntimeAgentFactoryResolver(delegate=D, override.keys=AC3 keys, injection=J)
  resolve_factory:
    authoritative = D.resolve_factory(runtime_revision, package_digest, factory_reference)
    if key ∉ AC3 keys: return authoritative
    return wrap_with_host_injection(authoritative, J)  # still invokes D first

build_registry_projection → build_application_registry → resolve_factory (wrapper calls D inside)
```

**Negative paths:**

| Condition | Outcome |
|-----------|---------|
| AC3 key + delegate resolution fails | `RuntimeAgentFactoryResolutionError` → `RegistryProjectionError` |
| Wrapper + agent class ≠ `ExternalContractorAdapterAgent` | Fail-closed at materialization |
| Missing sanctioned port on injection | Fail-closed before projection or at `domain_job` tier-2 deny (existing) |
| `materialization_artifact_digest` mismatch | Existing `RegistryProjectionError` |

---

## 5. Where factory provenance is established (proof map)

| Stage | Symbol / file | Establishes |
|-------|---------------|-------------|
| Revision trust | `InMemoryRuntimeAgentFactoryResolver.resolve_factory` L93–98 · venv bundle equivalent | `package_digest ∈ runtime_revision.installed_agent_package_digests` |
| Artifact scope | `VenvBundleRuntimeAgentFactoryResolver` | Factory callable loaded under artifact import isolation |
| Roster binding | `factory_reference_for_roster_entry` + `_factory_authority_fingerprint` | Which factory reference is bound to logical agent |
| Materialization invoke | `build_application_registry` `wiring.py` L547–551 | **Must** run resolver for each enabled entry |
| Contract identity | `RosterAgentContractAuthority.contract_for_binding` | AgentContract from package digest + contract_id |
| Materialization identity | `registry_projection.py` L398–405 | `materialization_artifact_digest` alignment |
| **Gap @ `cae02d4`** | `HostOverridingRuntimeAgentFactoryResolver` L49–51 | **Bypasses** delegate — provenance chain broken |

---

## 6. Recommended unified solution (R3-F01…03)

### 6.1 F01 — Provenance-preserving host injection

1. Replace `HostRuntimeAgentFactoryOverride.host_factory` with `StrictExternalWorkHostAgentInjection` (frozen dataclass: port, integration, boundary, require flag).
2. `HostOverridingRuntimeAgentFactoryResolver.resolve_factory`: **always** `authoritative = delegate.resolve_factory(...)`; if key in override set, return `wrap_authoritative_factory_with_strict_external_work_injection(authoritative, injection)`.
3. Implement wrapper in `intergrax/applications/_shared/` (Tier-0) calling Tier-2 typed injection helper in `agents/external_contractor_adapter/` (no `applications/` import in `intergrax/`).
4. `strict_sanctioned_external_work_registry_projection.py`: build injection from `host_runtime`; stop using `build_governed_contractor_agent_builders` as resolver return value (builders remain for non-revision-bound lab only).
5. Optional evidence hardening: extend `_registry_factory_wiring_digest` with `strict_external_work_host_injection_digest` — **operator ratification OD-R3-02** if auditors require digest-level proof.

### 6.2 F02 — Capability truth sourcing

1. Add `intergrax/contracts/external_work_provider_capability_projection.py` with:
   - `project_descriptor_capabilities(descriptor) -> …` (token-mapped fields only)
   - `merge_integration_invariants(integration_admitted: bool) -> supports_create, supports_idempotency`
2. Update `resolve_strict_external_work_provider_capabilities` to use merge; remove unconditional `True`.
3. Keep operator pin unchanged for identity/descriptor bytes.

### 6.3 F03 — Typed durability boundary

1. Add `is_durable` to GCA store Protocols; remove `getattr` from bundle.
2. Document qualification evidence matrix (restart, F02, corruption, RPL) in child QUAL task — no new backend in R3 implementation child until operator approves vendor qualification path.

---

## 7. Minimal file-level implementation plan (child IMPL-01-R3 only)

| File | Action |
|------|--------|
| `intergrax/applications/_shared/host_overriding_runtime_agent_factory_resolver.py` | Injection model; delegate-first `resolve_factory` |
| `intergrax/applications/_shared/strict_external_work_authoritative_factory_injection.py` | **New** — wrapper factory + fail-closed typing |
| `agents/external_contractor_adapter/strict_host_dependency_injection.py` | **New** — typed port injection on materialized agent |
| `applications/.../strict_sanctioned_external_work_registry_projection.py` | Use injection, not `host_factory` |
| `applications/.../resolve_strict_external_work_provider_capabilities.py` | Remove hardcoded create/idempotency |
| `intergrax/contracts/external_work_provider_capability_projection.py` | **New** — descriptor + invariant merge |
| `applications/.../stores.py` | `is_durable` on Protocols |
| `applications/.../governed_external_work_host_stores.py` | Typed bundle durability |
| `applications/.../production_external_work_composition.py` | Symmetric GCA durability gate (if not already present @ child base) |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_strict_composition_recovery_architecture_gate.py` | Extend: delegate-first resolver; no unconditional capability `True` |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_r3_ac3_provenance_and_capabilities.py` | **New** — positive/negative cases below |

**Remove / do not retain:**

| Item | Action |
|------|--------|
| `host_factory` short-circuit without `delegate.resolve_factory` | **Remove** |
| `getattr(..., "is_durable", False)` on bundle | **Remove** |
| Unconditional `supports_create=True` / `supports_idempotency=True` in strict projector | **Remove** |
| `strict_revision_bound_agent_registry_rebind.py` | **Stay deleted** (R2) |
| Second registry or resolver authority | **Forbidden** |

---

## 8. Acceptance tests (DEFERRED — NOT EXECUTED)

### 8.1 Positive (child pytest)

| ID | Intent |
|----|--------|
| G-R3-01 | AC3 materialization: mock delegate records `resolve_factory` invocation before wrapper returns |
| G-R3-02 | Materialized agent has sanctioned port + `require_sanctioned_external_work_mutations=True` |
| G-R3-03 | `registry_projection` succeeds with injection + valid `roster_contract_authority` |
| G-R3-04 | Descriptor with `QUOTE_FIRST` maps quote/accept; invariants supply idempotency for quote-first gate |
| G-R3-05 | Bundle with all `is_durable=True` passes `validate_production_governed_external_work_host_store_bundle` |

### 8.2 Adversarial negative (child pytest)

| ID | Intent |
|----|--------|
| N-R3-01 | Override key + delegate raises → materialization fails closed |
| N-R3-02 | Authoritative factory returns wrong agent class → injection wrapper raises |
| N-R3-03 | Pin/descriptor mismatch → `StrictExternalWorkProviderResolutionError` |
| N-R3-04 | Descriptor missing `QUOTE_FIRST` but orchestrator quote path requested → fail-closed at capability projection or orchestrator |
| N-R3-05 | Any store `is_durable=False` in strict wiring → `ValueError` on validate |
| N-R3-06 | AST gate: `resolve_strict_external_work_provider_capabilities.py` contains no literal `supports_create=True` / `supports_idempotency=True` without invariant helper |

---

## 9. FRZ mapping (applicable @ decision scope — criteria remain OPEN)

| FRZ ID | R3 contribution | Status |
|--------|-----------------|--------|
| **FRZ-PRD-08** | Closes production bypass class: no factory substitution without artifact resolution | **OPEN** — decision only |
| **FRZ-SEC-01 / 04** | Strict pin + fail-closed capabilities; no auth bypass via capability overstating | **OPEN** |
| **FRZ-REL-03 / 05** | Idempotency ownership explicit via invariant layer + orchestrator flags; durability typed boundary | **OPEN** — no restart proof |
| **FRZ-CTR-01 / 02** | Protocol-first capability projection; typed store protocols | **OPEN** |
| **FRZ-BND-02 / 03** | Tier boundaries: injection helper in agents/, wrapper in intergrax/ | **OPEN** |
| **FRZ-OWN-01 / 02** | Single materialization owner; no second registry | **OPEN** |
| **FRZ-TYP-01 / 02** | Remove `getattr` pseudo-contract; typed injection | **OPEN** |
| **FRZ-RPL-01 / 02** | Durability qualification criteria for backend replacement | **OPEN** |
| **FRZ-REG-02 / 03** | Architecture gates + child tests listed §8 | **OPEN** |

**No global PASS** · **no parent IMPL-01 closure** · **no R7-Q qualification**.

---

## 10. Enterprise audit matrix

| Control | @ `cae02d4…` | R3 target |
|---------|--------------|-----------|
| Contracts over implementations | **PARTIAL** — capabilities hardcoded | Descriptor + invariant contracts |
| Layer boundaries | **PASS** — no `intergrax` → `applications` | Preserve |
| Exactly-one composition owner | **PASS** — `factory.py` + projection | Preserve |
| Strong typing | **FAIL** — `getattr` durability | Protocol `is_durable` |
| No second registry/resolver authority | **PASS** — overlay removed | Preserve; fix override semantics |
| Fail-closed | **PARTIAL** — provenance + capabilities | Delegate-first + invariant merge |
| Governance ≠ execution | **PASS** | Preserve |
| Evidence continuity | **FAIL** — wiring digest vs callable | Delegate-first + optional injection digest |
| Regression protection | **PARTIAL** — R2 gates exist | Extend R3 gates §8 |

---

## 11. Operator decisions awaiting ratification

| ID | Decision | If rejected |
|----|----------|-------------|
| **OD-R3-01** | Accept **integration invariant layer** for `supports_create` / `supports_idempotency` without new `ExternalWorkCapability` tokens | Must ratify descriptor enum expansion (`CREATE`, `IDEMPOTENCY`) before implementation |
| **OD-R3-02** | Optional `strict_external_work_host_injection_digest` in `registry_factory_wiring_digest` | Rely on behavioral tests only for injection evidence |
| **OD-R3-03** | Qualification vendor/path for production `GovernedExternalWorkHostStoreBundle` (deferred backend) | Strict production remains blocked |

**Not STOP:** F01 wrapper does not create new semantic owner — it corrects `HostOverridingRuntimeAgentFactoryResolver` behavior within ratified OD-R2-02 seam.

---

## 12. Unresolved blockers and next child

| Blocker | Classification |
|---------|----------------|
| AC3 factory bypass @ `cae02d4` | **IN-SCOPE** → IMPL-01-R3 |
| Hardcoded provider capabilities | **IN-SCOPE** → IMPL-01-R3 |
| `getattr` durability + no qualified durable GCA backend | **IN-SCOPE** (typing + evidence); backend **IN-SCOPE BLOCKER** for qualification |
| HTTP E2E / restart proofs | **DEFERRED** — NOT EXECUTED |
| Parent **IMPL-01** closure | **BLOCKED** until R3 implementation + independent exact-SHA audit |

**Next exact implementation child:** `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01-R3-AC3-PROVENANCE-RECOVERY` (F01→F03 only; bounded diff per §7).

---

## 13. Status

**READY FOR AUDIT** — documentation @ commit containing this file; code evidence @ `cae02d4cc6520c81422da511e4efb3b2044cc3ea`; **production delta = 0**.

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
