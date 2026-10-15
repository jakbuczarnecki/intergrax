# PROD-Q-R1-R7-Q-P2-C-ADR-01 — Canonical Package Closure Contract Authority

| Field | Value |
|-------|-------|
| **Status** | **ACCEPTED — operator-ratified** (2026-10-15) |
| **Revision** | **R0** (ratified — no semantic redesign) |
| **Decision date** | 2026-10-15 |
| **Branch** | `development` |
| **Implementation evidence SHA** | `8e43b160447ca43de6631fe2a1b7f99948b64f2f` (IMPL-01 + P2-C HTTP qualification chain) |
| **Ratification** | Operator closure **PROD-Q-R1-R7-Q-P2-OPERATOR-CLOSURE** on `development` (documentation record; independent GitHub SHA audit pending on closure commit) |
| **Parent** | **PROD-Q-R1-R7-Q-P2** — **CLOSED — operator exact-SHA acceptance** (does **not** close **PROD-Q-R1-R7-Q** or **PROD-Q**) |
| **Ratification scope** | Architecture **3.A + 3.C** only — **not** certification of every production entrypoint or global architectural freeze |
| **Baseline ADR** | [`PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md`](PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md) (**R6** — ratified deployment authority; does not close this slice) |
| **Gate plan** | [`PROD_Q_R1_R7_Q_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md`](PROD_Q_R1_R7_Q_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md) · P2 qualification record [`PROD_Q_R1_R7_Q_P2_STRICT_PRODUCTION_INTEGRATED_E2E.md`](PROD_Q_R1_R7_Q_P2_STRICT_PRODUCTION_INTEGRATED_E2E.md) |
| **Does not close** | **PROD-Q-R1-R7-Q** · **PROD-Q-R1-R7** · **PROD-Q** · any **FRZ-PRD** global PASS |

---

## 1. Confirmed root cause (static @ HEAD)

Strict GCA HTTP assembly (`create_governed_contractor_strict_production_asgi_app` → `create_governed_contractor_process_app` → `create_governed_contractor_backend_app` → `build_harness_host_runtime` → `wire_application_environment`) runs canonical package closure when `conformance_check=True` (default):

`wire_application_environment` → `assert_manifest_package_closure` → `build_application_package` → `collect_application_dependencies`.

`collect_application_dependencies` **unconditionally** calls `materialize_manifest_contract_authority_lab_compat(manifest)` before iterating bindings (`intergrax/applications/_shared/package_wiring.py` L56–61).

`materialize_manifest_contract_authority_lab_compat` iterates every enabled binding and invokes `resolve_agent_contract_from_binding`, which requires a resolvable local agent class / `*.contract.build_agent_contract` (`intergrax/applications/_shared/roster_agent_contract_authority.py` L126–137, `agent_resolution.py` L83+).

The P2-C-ID-R1 strict host manifest uses **revision-bound declarative** `AgentBinding(contract_id="search")` with **no** `agent_type` / `import_path` (`tests/unit/applications/test_production_registry_startup_ac3_final.py` `_ac3_revision_bound_host_manifest`). Materialization therefore fails **before** HTTP, TestClient, or admission execution — independent of D4 session / `ServingRuntimeHandoff` correctness.

**Not sufficient:** removing the `lab_compat` call without supplying revision-bound `RosterAgentContractAuthority` would break versioned AGENT dependency materialization for class-mounted bindings and would not prove revision/digest alignment for strict serving.

**Prior qualification note (ID-01):** declarative branch in `collect_application_dependencies` (L64–66) can emit `ref=search` without calling `resolve_roster_agent_contract`, but the unconditional `lab_compat` materialization above still aborts the function for declarative-only rosters.

---

## 2. Closed-world ownership graph

### 2.1 Contract metadata authority (strict production)

| Concern | Exactly-one authority @ strict production |
|---------|-------------------------------------------|
| AgentContract snapshots for installed packages | `AgentPackageContractAuthorityService` + durable `AgentArtifactMetadataStore` rows (install authority — EBH-2E-AR1-A-R1-R1) |
| Revision-bound view for one serving revision | `RosterAgentContractAuthority` keyed by `(package_digest, contract_id)` derived from `RuntimeRevision` + `EffectiveRoster` + manifest binding map (`RosterAgentContractAuthority.from_revision_roster`) |
| Canonical resolution API | `resolve_roster_agent_contract(binding, contract_authority=…, allow_compatibility_resolver=False)` → fail-closed `StrictAgentContractAuthorityError` |
| Lab / monorepo manifest-only snapshots | `ManifestAgentContractAuthority` via `materialize_manifest_contract_authority_lab_compat` — **documented non-production** (`AGENT_DISTRIBUTION.md`, `strict_product_manifest_ci_gates.py`) |

**Strict production truth for contract metadata:** **`RosterAgentContractAuthority`** resolved through **`RegistryProjectionAuthorityResolver.roster_contract_authority_for`** and carried on **`RegistryProjectionInputBundle.roster_contract_authority`**.

### 2.2 Who selects contract authority

| Layer | Owner | Role |
|-------|-------|------|
| Lifecycle / revision stores | Agent Distribution | Persist revision, roster, digests, artifact metadata |
| Projection authority resolution | `RegistryProjectionAuthorityResolver` | Exactly-one canonical resolver; forbids caller-supplied roster overrides |
| Production bundle assembly | `build_production_registry_projection_input_bundle_for_revision` | Binds `roster_contract_authority` onto frozen `RegistryProjectionInputBundle` |
| Registry materialization | `build_registry_projection` | Validates authority via `_validate_roster_contract_authority`; registers agents with same authority |
| Package closure (today) | `collect_application_dependencies` | **Incorrectly** selects `lab_compat` manifest authority internally — **no production selector** |
| Environment wiring | `wire_application_environment` | Invokes package closure; does not accept contract authority today |
| Strict GCA composition root | `create_governed_contractor_backend_app` + `build_harness_host_runtime` | Sanctioned path that must **supply** authority to wiring/closure when `registry_projection` / `ProductionProcessComposition` present |

### 2.3 Availability at GCA strict composition root

| Artifact | Available at strict GCA factory? |
|----------|----------------------------------|
| `ServingRuntimeHandoff` | Yes — via `process_composition.serving_runtime_handoff` (identity + factory resolver) |
| `RegistryProjectionInputBundle.roster_contract_authority` | Yes — via `ProductionProcessLifecycle` / `projection_input_store.get(runtime_revision_id)` (same bundle used to establish handoff in `strict_production_serving_launcher.py` L180–228) |
| `MaterializedRegistryProjection` | Yes — passed into `build_harness_host_runtime`; carries **agent registry read model**, not authority handle |
| `RosterAgentContractAuthority` on handoff object | **No** — not a field on `ServingRuntimeHandoff`; must be threaded from projection input store or explicit injection parameter |

### 2.4 Package closure contract chain (callers)

```text
wire_application_environment (environment_wiring.py)
  └─ assert_manifest_package_closure (package_wiring.py)
       ├─ build_application_package
       │    └─ collect_application_dependencies  ← lab_compat (defect)
       └─ validate_application_package_closure
            └─ collect_application_dependencies (recompute expected)

build_application_package / validate_* also used by:
  - scripts/maintenance/check_application_package.py
  - scripts/gates/check_application_production_gates.py
  - strict_product_manifest_ci_gates.py (explicit lab_compat — sanctioned)
  - qualification: test_prod_q_r1_r7_q_p2_c_package_closure_regression.py
```

**Strict production integrated path (additional):**

```text
open_strict_production_serving_session → ServingRuntimeHandoff
create_governed_contractor_strict_production_asgi_app
  → create_governed_contractor_process_app
  → create_governed_contractor_backend_app
  → build_harness_host_runtime(..., registry_projection=...)
  → wire_application_environment → assert_manifest_package_closure
```

**Parallel production path (not package-closure blocker for P2-C but same authority model):**

`bootstrap_production_registry_projection` → `build_production_registry_projection_for_revision` → `build_registry_projection(bundle)` — already uses `bundle.roster_contract_authority`.

### 2.5 Layer boundary

| Check | Verdict |
|-------|---------|
| `ContractAuthority` union lives in `intergrax/applications/_shared/roster_agent_contract_authority.py` (Tier-3 applications) | OK |
| Authority data sourced from Agent Distribution stores via existing resolver | OK — no new registry |
| Package closure accepting injected `ContractAuthority` | OK — typed boundary; applications-owned wiring |
| Tier-0/1 importing applications | Forbidden — unchanged |
| Governance widening | Not required — closure remains applications environment conformance |

### 2.6 Must validation verify revision contract / version / digest?

| Requirement | Current | Target |
|-------------|---------|--------|
| Agent `ref` on roster | `validate_application_package_closure` checks `dep.ref` ∈ manifest roster contract ids | Keep |
| Version constraint for class-mounted agents | `={contract.version}` via `resolve_roster_agent_contract` | Keep for lab with `ManifestAgentContractAuthority` |
| Version constraint for declarative revision-bound agents | `*` (declarative branch) — **weak** | **Use `RosterAgentContractAuthority.contract_for_binding`** → `={snapshot.version}` |
| Digest / package identity | Not in package closure today; enforced in projection (`_validate_roster_contract_authority`, install records) | Package closure **must not** re-resolve agent classes; optional **consistency check** that AGENT deps match `registry_projection.agent_registry` contract ids + versions when projection supplied (additive, same validator) |

---

## 3. Architecture alternatives

### 3.A — Explicit contract-authority injection (preferred)

**Design:** Add optional typed `contract_authority: ContractAuthority | None` to `collect_application_dependencies`, `validate_application_package_closure`, `assert_manifest_package_closure`, and `wire_application_environment` / `build_harness_host_runtime`. Strict production composition root resolves **`RosterAgentContractAuthority`** once from **`RegistryProjectionInputBundle`** (projection input store @ `handoff.identity.runtime_revision_id`) and passes it explicitly. When `contract_authority` is `None`, only **explicit lab entrypoints** may materialize `ManifestAgentContractAuthority` (see 3.C).

| Dimension | Assessment |
|-----------|------------|
| **Semantic owner** | Agent Distribution artifact metadata + revision roster; applications consume via `RosterAgentContractAuthority` |
| **Composition owner** | Strict host: `create_governed_contractor_backend_app` / `build_harness_host_runtime`; lifecycle: projection bundle assembly (`build_production_registry_projection_input_bundle_for_revision`) |
| **Changed contracts / callers** | `package_wiring.py`, `environment_wiring.py`, `harness_host_runtime.py`, `applications/governed_contractor_application/host/factory.py`; tests + gate scripts that intentionally use lab authority |
| **Fail-closed** | `contract_authority=None` + binding needs class resolution → `StrictAgentContractAuthorityError`; production factory must pass non-None `RosterAgentContractAuthority` for strict integrated path |
| **Package closure impact** | Removes unconditional `lab_compat`; AGENT deps use authority for all non-pure-declarative cases; declarative bindings use authority for version pinning |
| **Frozen-revision identity** | Same authority instance as registry projection — no second truth |
| **Tenant / revision / digest continuity** | Preserved via existing bundle + handoff identity checks |
| **Backwards compatibility** | Lab/harness: explicit `None` → gated lab materialization only where already sanctioned (CI gates, dev registry) |
| **Pluginability** | Unchanged — authority service behind `RosterAgentContractAuthority` |
| **Risks** | Missing injection on a strict host → fail-closed (desired); incomplete caller audit |
| **Regression requirements** | P2-C regression module; architecture gate forbidding `lab_compat` in production host modules; unit tests for declarative `search` + authority |

### 3.B — Reuse registry projection evidence only

**Design:** Derive expected AGENT dependencies from `MaterializedRegistryProjection.agent_registry` / harness snapshot instead of `collect_application_dependencies` lab path.

| Dimension | Assessment |
|-----------|------------|
| **Semantic owner** | Projection read model (downstream of roster authority) |
| **Composition owner** | `build_harness_host_runtime` when `registry_projection` provided |
| **Fail-closed** | Partial — registry lists ids/contracts but **does not replace** version constraint materialization in `ApplicationPackage.dependencies` without authority |
| **Frozen-revision identity** | Risk of **derived-second** dependency list diverging from `ApplicationPackage` checksum contract unless `collect_application_dependencies` still uses roster authority |
| **Constraint fit** | **Insufficient alone** — does not remove `lab_compat` from `collect_application_dependencies` without coupling package builder to projection artifact; must combine with **3.A** for single truth |
| **Verdict** | **Supporting mechanism only** (optional cross-check in `validate_application_package_closure` when `registry_projection` present), **not** standalone fix |

### 3.C — Isolate lab compatibility

**Design:** Restrict `materialize_manifest_contract_authority_lab_compat` to `strict_product_manifest_ci_gates.py`, test fixtures, and explicit lab helpers; production paths forbidden by AST/architecture gate (existing EBH-2E patterns).

| Dimension | Assessment |
|-----------|------------|
| **Semantic owner** | Lab CI gates (`strict_product_manifest_ci_gates.py`) |
| **Composition owner** | CI gate entrypoints only |
| **Fail-closed** | Production without injected authority → hard error (requires 3.A plumbing) |
| **Package closure impact** | Mandatory companion to 3.A — **removing lab_compat from `collect_application_dependencies` without injection fails closed everywhere** |
| **Constraint: no implicit mode detection** | **Satisfied** if lab materialization happens only when caller passes `ManifestAgentContractAuthority` explicitly or uses sanctioned CI helper — **not** via `execution_mode` branching inside `collect_application_dependencies` |
| **Verdict** | **Required complement** to 3.A; **not sufficient** alone |

### 3.D — Constraint satisfaction matrix

| Constraint | A (+C) | B alone | C alone |
|------------|--------|---------|---------|
| Exactly one contract truth | Yes | No | No |
| One sanctioned composition owner | Yes | Ambiguous | No |
| No implicit runtime mode detection | Yes (explicit param) | Risky | Partial |
| No string-based authority switching | Yes | Yes | Yes |
| No alternate package validator | Yes | No (fork risk) | Yes |
| No package-closure bypass | Yes | Bypass temptation | Yes |
| No new registry | Yes | Yes | Yes |
| No local agent-class re-instantiation | Yes | Yes | Yes |
| No Governance widening | Yes | Yes | Yes |
| Fail-closed missing/stale contract | Yes | Partial | Yes (but breaks lab) |

**Decision (operator-ratified 2026-10-15):** **ACCEPTED — 3.A + 3.C**, with optional **3.B** cross-check in `validate_application_package_closure` when `registry_projection` is supplied to wiring (optional — not required for P2 closure). **3.B** alone and **3.C** alone remain **rejected** as standalone fixes (see matrix §3.D).

---

## 4. Preferred solution (proposal)

1. **Remove** unconditional `materialize_manifest_contract_authority_lab_compat` from `collect_application_dependencies`.
2. **Require** typed `contract_authority: ContractAuthority | None` parameter on collection/validation/assertion APIs.
3. **Resolve** production authority only at strict composition boundary from existing `RegistryProjectionInputBundle.roster_contract_authority` (projection input store keyed by attested `runtime_revision_id`) — **no second** `RegistryProjectionAuthorityResolver` pass in package wiring.
4. **Thread** authority: `create_governed_contractor_backend_app` → `build_harness_host_runtime` → `wire_application_environment` → `assert_manifest_package_closure`.
5. **Retain** `materialize_manifest_contract_authority_lab_compat` for **`strict_product_manifest_ci_gates.py`**, explicit test helpers, and callers that construct `ManifestAgentContractAuthority` off-line and pass it in.
6. **Optional:** when `registry_projection` is available during validation, assert AGENT dependency refs/versions align with `registry_projection.agent_registry` contracts (fail-closed).

**Justification:** Matches EBH-2E install/projection authority model; reuses frozen bundle already required for handoff; preserves single validator; satisfies “no local agent-class re-instantiation” for declarative revision-bound hosts.

---

## 5. Expected change surface (implementation wave — not in this task)

| File / symbol | Change |
|---------------|--------|
| `package_wiring.collect_application_dependencies` | `contract_authority` param; lazy use of authority; no default lab_compat |
| `package_wiring.validate_application_package_closure` | pass-through authority; optional projection cross-check |
| `package_wiring.assert_manifest_package_closure` | pass-through authority |
| `environment_wiring.wire_application_environment` | `agent_contract_authority` optional param |
| `harness_host_runtime.build_harness_host_runtime` | pass authority to wiring |
| `governed_contractor_application/host/factory.py` | resolve from `process_composition` + projection input store; pass to harness builder |
| `strict_product_manifest_ci_gates.py` | pass lab `ManifestAgentContractAuthority` explicitly into closure calls |
| `capability_graph_wiring.build_environment_seed_capability_graph` | **Deferred follow-up** — still uses `lab_compat` @ HEAD; strict host uses `build_environment_capability_graph_from_wiring` path — verify no P2-C blocker before expanding scope |
| Architecture tests | Extend EBH-2E production path gates to `package_wiring` / GCA factory |

**Out of scope:** new validator, Governance changes, HTTP/E2E replay, physical activation.

---

## 6. Migration / compatibility

- **Lab / dev hosts** using `wire_application_environment` without authority: must pass explicitly materialized `ManifestAgentContractAuthority` from CI gate helper or migrate to roster authority in integration tests that simulate production.
- **Scripts** (`check_application_package.py`, `check_application_production_gates.py`): pass lab authority via existing gate helper or document non-strict usage.
- **Checksum / package JSON artifacts:** AGENT `version_constraint` for declarative revision-bound bindings may change from `*` to `={version}` when authority supplied — intentional tightening for strict parity.
- **Backwards compatibility:** default `None` behavior = **fail-closed** for bindings requiring class resolution; lab callers must opt in explicitly (3.C).

---

## 7. Targeted lightweight test matrix (post-implementation)

| ID | Scope | Command / module |
|----|-------|------------------|
| T1 | Declarative `contract_id=search` + `RosterAgentContractAuthority` | extend `test_prod_q_r1_r7_q_p2_c_package_closure_regression.py` |
| T2 | `collect_application_dependencies` rejects missing authority for class mount | unit `test_package_wiring.py` |
| T3 | GCA factory strict path does not import `lab_compat` | architecture gate (mirror `test_ebh_2e_ar1_a_r1_r1_install_authority_closure.py`) |
| T4 | `wire_application_environment` strict with injected roster authority → zero closure violations | unit/integration with AC3 fixture |
| T5 | CI gates still pass lab manifest closure | `strict_product_manifest_ci_gates` tests |

**Deferred:** `test_prod_q_r1_r7_q_p2_c_http_execution_admission.py` full module; P2-B integrated module; D4 replay.

---

## 8. FRZ mapping (local contribution only)

| Criterion | Relevance |
|-----------|-----------|
| **FRZ-PRD-03** | Unsupported / inconsistent production configuration must fail closed — spurious `lab_compat` on strict host violates strict-vs-lab separation |
| **FRZ-PRD-07** | Strict vs lab semantics — package closure must use revision-bound authority on strict integrated path |
| **FRZ-PRD-08** | Production bypass prevention — manifest class resolution is a lab bypass for declarative revision-bound serving |
| **FRZ-REG-02 / 03** | Registry projection authority alignment with environment conformance |
| **FRZ-CTR-01** | Structural replaceability — injected authority preserves port boundaries |

**No global FRZ PASS** implied by this ADR proposal.

---

## 9. Outstanding blockers

| Blocker | Owner |
|---------|-------|
| Independent exact-SHA audit of operator ratification + IMPL-01 @ `8e43b160447ca43de6631fe2a1b7f99948b64f2f` | Auditor |
| P2-B full integrated module replay @ current HEAD (optional regression; static continuity recorded) | Operator-scheduled qualification |
| `capability_graph_wiring` lab_compat on seed graph (if strict wiring hits seed path) | Follow-up architecture check |
| Parent **PROD-Q-R1-R7-Q** closure (P3 + E.1 replay) | R7-Q program |

**Closed since R0:** IMPL-01 implementation; P2-C HTTP module execution @ `8e43b1604` (partial log — see P2 qualification record).

---

## 10. Disposition

| Item | Value |
|------|-------|
| **Maximum status** | **ACCEPTED — operator-ratified** (independent exact-SHA audit of closure commit still required) |
| **Parent PROD-Q-R1-R7-Q-P2** | **CLOSED — operator exact-SHA acceptance** — implementation @ `8e43b1604`; ADR ratification records architecture authority; **not** a substitute for independent code audit |
| **FRZ promotion** | **None** (local P2 contributions only) |

---

## 11. Operator ratification record (implementation @ `8e43b160447ca43de6631fe2a1b7f99948b64f2f`)

**Accepted architecture (3.A + 3.C), subject to documented conditions:**

- Strict production contract metadata authority: **`RosterAgentContractAuthority`** via **`RegistryProjectionAuthorityResolver`** and trusted revision/roster/artifact metadata.
- Explicit typed **`ContractAuthority`** injection through sanctioned GCA composition (`resolve_strict_package_contract_authority_for_serving_handoff` → `build_harness_host_runtime` → `wire_application_environment` → `assert_manifest_package_closure` → `collect_application_dependencies`).
- Exactly **one** canonical package-closure validator (`assert_manifest_package_closure` / `validate_application_package_closure` — no second mechanism).
- **`ManifestAgentContractAuthority`** / `materialize_manifest_contract_authority_lab_compat` **explicit** for lab/CI only; **zero** implicit lab compatibility fallback on strict production integrated path.
- **No** additional registry, semantic owner, authorization authority, or execution path.

**Evidence basis (not re-executed in ratification task):** P2-C HTTP @ `8e43b1604` (6 passed / partial log); IMPL-01 static chain; P2 qualification record [`PROD_Q_R1_R7_Q_P2_STRICT_PRODUCTION_INTEGRATED_E2E.md`](PROD_Q_R1_R7_Q_P2_STRICT_PRODUCTION_INTEGRATED_E2E.md).

**Does not claim:** global **FRZ-*** PASS; **PROD-Q-R1-R7-Q** PASS; full external business execution (terminal delegate / LLM stubbed).
