# AGENT_SYSTEM - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** AGENT_SYSTEM
- **Constituent domains:** AGENT_CONTRACTS_AND_ASSEMBLY · AGENT_DISTRIBUTION
- **Tier(s):** Tier-0 agent contracts · Tier-1 `AgentRegistry` / routing policy · Tier-3 registry bootstrap surfaces
- **audited_sha:** `654a7c0e3fe823a43a2620645848248023e1c64e`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`
  - `docs/project/architecture/AGENT_DISTRIBUTION.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md`
  - `docs/project/maintainers/plans/AGENT_DISTRIBUTION.md`
- **Scope in:**
  - `AgentContract` schema, assembly validation, and registry storage semantics
  - `AgentRegistry.register` / `get_contract` / `list_contracts` / `from_agents`
  - `evaluate_agent_routing` production-mode fail-closed posture
  - `allowed_tools` author-time vs registry resolution path
  - production certification operational metadata (`owner_*`, `on_call_contact`, `runbook_ref`)
  - registry bootstrap identity projection vs package/contract canonical identity
  - Agent Distribution registry projection boundary (§21) vs Tier-1 execution registry
  - Nexus `AgentRouter` consumption of `AgentRegistry` routing decisions
- **Scope out:**
  - remediation implementation
  - full TOOLS domain audit
  - Nexus routing redesign
  - Agent Distribution activation/materialization re-audit beyond registry projection identity
  - product host operational qualification
- **Prior audit reference(s):** ACP + ACP-CLOSE + ACP-FINISH **Done**; AUDIT-IDEAL §12–§20 **Done** (incl. 31.1 on-call); Protocol v2 [`STRATEGIC_HARNESS_MODEL`](STRATEGIC_HARNESS_MODEL.md), [`TIER_LAYER_BOUNDARIES`](TIER_LAYER_BOUNDARIES.md) (TL-FIX-B)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `995c34ee5b9ca355e9bf3ec02c425f51d6cedaf4`

## Executive summary

**Verdict: FAIL.** Six accepted findings (5 HIGH, 1 MEDIUM) show production routing can admit non-`production_eligible` agents while gating eligible ones on owner metadata; registered `AgentContract` instances are mutable and returned by reference; authors can inject `allowed_tools` without skill/tool resolution; `AgentRegistry.from_agents` silently rewrites canonical contract identity; `AgentContract` does not reject unknown fields; and documented on-call certification requirements diverge from assembly/routing enforcement. Positive controls: Agent / Nexus / UER / Tier-3 responsibility split holds; DEPRECATED and RETIRED routing is blocked; centralized `AgentRoutingDecision` exists; Agent Distribution activation uses atomic/CAS store boundary; no finding invalidates the full distribution design.

## Verdict

**FAIL** - 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-AGENT_SYSTEM-01

**Production routing bypasses `production_eligible` gate**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT / GOVERNANCE BYPASS
- **Status at publication:** ACCEPTED
- **Remediation block:** AGSYS-CONTRACT-INTEGRITY
- **Claim falsified:** Production routing fails closed; `production_eligible` must be positively established before an agent may participate in production routing; lifecycle state and operational metadata are additional gates, not substitutes.
- **Observation:** `evaluate_agent_routing(contract, production_mode=True)` returns `routable=True` when `contract.production_eligible` is `False` (line 49–50). Conversely, when `production_eligible=True`, owner/runbook checks apply. A non-production-eligible agent can therefore route in production mode while a production-eligible agent is subjected to stricter metadata gates.
- **Location:**
  - `intergrax/runtime/registry/agent_routing_policy.py` - `evaluate_agent_routing()` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
  - `intergrax/runtime/registry/agent_registry.py` - `is_routable()`, `find_by_capability()` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
- **Reproduction:**
  1. `git show 654a7c0e3fe823a43a2620645848248023e1c64e:intergrax/runtime/registry/agent_routing_policy.py` - `if not contract.production_eligible: return AgentRoutingDecision(routable=True)`.
  2. Contrast with owner/runbook checks that apply only when `production_eligible=True`.
  3. `tests/unit/runtime/architecture/test_agent_routing_policy.py` - supporting evidence for routing branches.
- **Impact:** Non-production-eligible agents can participate in production Nexus selection; governance posture is inverted relative to certification intent.
- **Confidence:** CONFIRMED

### AUDIT-20260818-AGENT_SYSTEM-02

**Registered `AgentContract` is mutable and exposed by reference**

- **Severity:** HIGH
- **Category:** CONTRACT / STATE INTEGRITY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** AGSYS-CONTRACT-INTEGRITY
- **Claim falsified:** Registry execution truth uses immutable validated contract snapshots or an explicit versioned/validated transition mechanism; no ambient post-registration mutation of canonical routing/security state.
- **Observation:** `AgentContract` uses `ConfigDict(arbitrary_types_allowed=True)` without `frozen=True` or `validate_assignment=True`. `AgentRegistry.register` stores the validated model instance directly in `_contracts`; `get_contract()` and `list_contracts()` return those stored models. Routing-critical metadata can change after registration without re-running assembly/certification validation.
- **Location:**
  - `intergrax/contracts/agent_contract_meta.py` - `AgentContract.model_config` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
  - `intergrax/runtime/registry/agent_registry.py` - `_contracts`, `get_contract()`, `list_contracts()` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
- **Reproduction:**
  1. `git show 654a7c0e3fe823a43a2620645848248023e1c64e:intergrax/contracts/agent_contract_meta.py` - no `frozen` / `validate_assignment`.
  2. `git show 654a7c0e3fe823a43a2620645848248023e1c64e:intergrax/runtime/registry/agent_registry.py` - `self._contracts[meta.id] = meta`; `get_contract` returns stored instance.
  3. `tests/unit/runtime/registry/test_agent_registry.py` - registry behavior evidence.
- **Impact:** Post-registration mutation of lifecycle, permissions, or production metadata can bypass assembly gates without audit trail.
- **Confidence:** CONFIRMED

### AUDIT-20260818-AGENT_SYSTEM-03

**`allowed_tools` author injection bypasses canonical resolution**

- **Severity:** HIGH
- **Category:** CONTRACT / PERMISSION BOUNDARY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** AGSYS-CONTRACT-INTEGRITY
- **Claim falsified:** One canonical agent tool-permission declaration/resolution path; authors cannot bypass `SkillManifest` / `ToolContract` resolution by directly injecting resolved `allowed_tools`.
- **Observation:** Architecture states `allowed_tools` is resolved by `AgentRegistry` from `skills` + `extra_tools`. Assembly validation (`validate_contract_metadata`) rejects `allowed_tools` only when `skills` or `extra_tools` are also present. A contract with `skills=[]`, `extra_tools=[]`, `allowed_tools=[...]` passes assembly and `register()` skips `resolve_contract_tools()` because the `if meta.skills or meta.extra_tools` guard is false.
- **Location:**
  - `intergrax/runtime/registry/agent_assembly_resolver.py` - `validate_contract_metadata()` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
  - `intergrax/runtime/registry/agent_registry.py` - `register()`, `resolve_contract_tools` guard @ `654a7c0e3fe823a43a2620645848248023e1c64e`
  - `intergrax/contracts/agent_contract_meta.py` - `allowed_tools` field semantics @ `654a7c0e3fe823a43a2620645848248023e1c64e`
- **Reproduction:**
  1. Author `AgentContract(skills=[], extra_tools=[], allowed_tools=["tool.a"])` - passes `assert_agent_assembly_valid`.
  2. `register()` stores author-supplied `allowed_tools` without `resolve_contract_tools`.
  3. Contrast with path when `skills` or `extra_tools` present - resolution enforced.
- **Impact:** Tool permission boundary can be declared without catalog/skill manifest validation.
- **Confidence:** CONFIRMED

### AUDIT-20260818-AGENT_SYSTEM-04

**`AgentRegistry.from_agents` silently rewrites canonical contract identity**

- **Severity:** HIGH
- **Category:** IDENTITY / ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** AGSYS-IDENTITY-PROJECTION
- **Claim falsified:** Canonical agent identity is not silently rewritten by registry bootstrap; identity mismatch fails closed or uses an explicit typed alias/binding contract.
- **Observation:** `AgentRegistry.from_agents(dict)` compares `contract.id` to dictionary key; on mismatch executes `contract.model_copy(update={"id": agent_id})` without error. This permits registry-local mutation of package/agent-declared canonical identity. Path remains in use by `legal_application` serving config (`LegalAgentServingConfig.from_agents` → `AgentRegistry.from_agents`).
- **Location:**
  - `intergrax/runtime/registry/agent_registry.py` - `from_agents()` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
  - `applications/legal_application/serving/fastapi_router.py` - `from_agents()` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
  - `docs/project/architecture/AGENT_DISTRIBUTION.md` - §6 identity model; §21 registry projection @ persistence time
- **Reproduction:**
  1. `git show 654a7c0e3fe823a43a2620645848248023e1c64e:intergrax/runtime/registry/agent_registry.py` - silent `model_copy(update={"id": agent_id})`.
  2. `git show 654a7c0e3fe823a43a2620645848248023e1c64e:applications/legal_application/serving/fastapi_router.py` - `AgentRegistry.from_agents(dict(agents))`.
- **Impact:** Registry key and package-declared identity can diverge without operator visibility; conflicts with Distribution identity/runtime projection model.
- **Confidence:** CONFIRMED

### AUDIT-20260818-AGENT_SYSTEM-05

**`AgentContract` accepts unknown fields (no `extra="forbid"`)**

- **Severity:** HIGH
- **Category:** CONTRACT DEFECT / FAIL-OPEN CONFIGURATION
- **Status at publication:** ACCEPTED
- **Remediation block:** AGSYS-CONTRACT-INTEGRITY
- **Claim falsified:** Canonical `AgentContract` rejects unknown fields; critical production/capability/permission/lifecycle metadata cannot silently disappear because of spelling or schema drift.
- **Observation:** `AgentContract.model_config = ConfigDict(arbitrary_types_allowed=True)` does not specify `extra="forbid"`. Pydantic v2 default allows extra fields to be ignored on validation, so misspelled or drifted contract keys are not a fail-fast validation boundary.
- **Location:**
  - `intergrax/contracts/agent_contract_meta.py` - `AgentContract`, `model_config` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
- **Reproduction:**
  1. `git show 654a7c0e3fe823a43a2620645848248023e1c64e:intergrax/contracts/agent_contract_meta.py` - no `extra="forbid"`.
  2. Instantiate with unknown key (e.g. `production_eligble=True`) - validation succeeds; intended field ignored.
- **Impact:** Schema drift and typos in production-critical metadata fail open rather than fail fast at assembly.
- **Confidence:** CONFIRMED

### AUDIT-20260818-AGENT_SYSTEM-06

**On-call certification claim diverges from assembly/routing enforcement**

- **Severity:** MEDIUM
- **Category:** IMPLEMENTATION / DOCUMENTATION DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** AGSYS-CONTRACT-INTEGRITY
- **Claim falsified:** Documented certification/lifecycle contract and actual enforcement agree on required operational metadata fields.
- **Observation:** ACP plan marks AUDIT-IDEAL-31.1 (Owner/on-call mandatory on all certified agents) as **Done**. `AgentContract` exposes `on_call_contact`. `validate_lifecycle_metadata` and `evaluate_agent_routing` require `owner_team`, `owner_contact`, and `runbook_ref` for `production_eligible` agents but do not require `on_call_contact`. Architecture cites on-call gate tooling (`check_on_call_ownership_model.py`) but runtime assembly/routing paths omit the field.
- **Location:**
  - `intergrax/contracts/agent_contract_meta.py` - `on_call_contact` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
  - `intergrax/runtime/registry/agent_assembly_resolver.py` - `validate_lifecycle_metadata()` @ `654a7c0e3fe823a43a2620645848248023e1c64e`
  - `intergrax/runtime/registry/agent_routing_policy.py` - production metadata checks @ `654a7c0e3fe823a43a2620645848248023e1c64e`
  - `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` - AUDIT-IDEAL-31.1 **Done** @ persistence time
- **Reproduction:**
  1. Compare plan **Done** claim for 31.1 with `validate_lifecycle_metadata` - no `on_call_contact` check.
  2. `evaluate_agent_routing` - owner/runbook only; no on-call.
- **Impact:** Operators may believe on-call is enforced for production certification when assembly/routing do not require it; requires explicit architectural decision to enforce or narrow claim.
- **Confidence:** CONFIRMED

## Falsification log (negative results)

1. **Agent / Nexus / UER / Tier-3 responsibility split invalid** - not falsified; contracts and registry remain Tier-1 execution surfaces; Distribution owns catalog/install/activation; Nexus routes via capability + registry.
2. **DEPRECATED and RETIRED agents routable in production** - not falsified; `evaluate_agent_routing` blocks both lifecycle states before production-mode checks.
3. **No centralized routing decision type** - not falsified; `AgentRoutingDecision` dataclass centralizes routability outcome.
4. **Agent Distribution activation lacks atomic boundary** - not falsified; architecture §20.5 requires traffic commit + registry projection in same activation boundary (not re-audited in depth here).
5. **Full Agent Distribution design invalid** - not falsified; findings are contract/registry integrity and identity projection gaps, not wholesale design rejection.
6. **Prior ACP/AUDIT-IDEAL delivery never occurred** - not falsified; historical **Done** rows remain valid delivery facts; this audit records residual enforcement gaps.

## Prior-audit comparison

First canonical Protocol v2 `AGENT_SYSTEM` layer snapshot at `654a7c0e3fe823a43a2620645848248023e1c64e`. Supplements - does not rewrite - ACP/ACP-CLOSE/ACP-FINISH harness closeout, AUDIT-IDEAL §12–§20, and Protocol v2 [`STRATEGIC_HARNESS_MODEL`](STRATEGIC_HARNESS_MODEL.md) / [`TIER_LAYER_BOUNDARIES`](TIER_LAYER_BOUNDARIES.md) (TL-FIX-B agent ownership). Discoveries are contract integrity, routing governance, and registry identity projection gaps beyond prior **Done** registers.

## Provider / backend abstraction

`NOT APPLICABLE - AGENT_SYSTEM scope is agent contracts, registry execution truth, and routing policy; no material external provider/backend substitution boundary in this layer.`

## Positive controls

1. **Agent / Nexus / UER / Tier-3 split** - documented ownership: Tier-2 agents, Tier-1 registry/routing, Nexus execution, Tier-3 bootstrap wiring @ audited SHA.
2. **DEPRECATED / RETIRED routing blocked** - `evaluate_agent_routing` returns `routable=False` for both states @ audited SHA.
3. **Centralized `AgentRoutingDecision`** - single policy function consumed by `AgentRegistry.is_routable` @ audited SHA.
4. **Agent Distribution atomic activation boundary** - architecture §20.5 traffic commit + registry projection coordination (conceptually sound; identity projection gap is separate).
5. **Skill/tool resolution path exists** - when `skills` or `extra_tools` declared, `resolve_contract_tools` runs at register @ audited SHA.

**FAIL qualification:** verdict means contract/registry integrity and identity projection gaps remain - **not** that the agent platform model or prior harness delivery is invalid.

## Root-cause remediation grouping

Planning only - **audit persistence does NOT implement remediation.**

### AGSYS-CONTRACT-INTEGRITY - production eligibility, immutable contract, tool resolution, fail-fast schema, on-call parity

**Findings:** 01, 02, 03, 05, 06

**Primary plan owner:** AGENT_CONTRACTS_AND_ASSEMBLY

Production routing requires positive `production_eligible`; registered routing contract state is immutable or version-controlled; one canonical `allowed_tools` resolution path; `AgentContract` rejects unknown fields; certification operational metadata contract matches enforcement (including on-call decision).

### AGSYS-IDENTITY-PROJECTION - canonical identity preservation in registry bootstrap

**Findings:** 04

**Primary plan owner:** AGENT_DISTRIBUTION (cross-reference ACP registry bootstrap surface)

Registry projection must preserve canonical package/contract identity; dictionary-key alias rewrite must fail closed or use explicit typed binding; distinguish manifest-only/bootstrap compatibility from activated runtime projection truth; reuse Distribution §6 identity model - no second identity subsystem. Cross-ref **TL-FIX-B** (single implementation authority) where registry bootstrap could admit competing identities.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `654a7c0e3fe823a43a2620645848248023e1c64e`; current `development` HEAD was not re-audited.
- AGSYS-03 does not expand into full TOOLS domain audit.
- AGSYS-01 does not prescribe Nexus routing redesign - repair belongs to agent routing policy/contract.
- AGSYS-06 does not silently weaken on-call target - enforcement vs claim narrowing requires explicit architectural decision during remediation.
- Tests are supporting evidence, not standalone proof.
- Remediation not performed in this task.

## Open questions / blocked items

- On-call (AGSYS-06): enforce `on_call_contact` in assembly/routing vs narrow AUDIT-IDEAL-31.1 claim - operator decision deferred to remediation.
- `from_agents` dict bootstrap: clean-cut removal vs explicit alias contract - prefer removal if no required consumer beyond documented compatibility path.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 6 (`AUDIT-20260818-AGENT_SYSTEM-01` … `AUDIT-20260818-AGENT_SYSTEM-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none
