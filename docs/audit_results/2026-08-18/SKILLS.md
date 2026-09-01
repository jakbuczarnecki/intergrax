# SKILLS - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** SKILLS
- **Constituent domains:** SKILLS (SkillManifest · catalog · registry · profile · resolver · contract merge)
- **Tier(s):** Tier-0 `SkillManifest` / `SkillRegistry` · Tier-1 `SkillResolver` · Tier-2 `AgentRegistry` contract merge · Tier-3 `SkillProfile` / `ToolProfile` wiring
- **audited_sha:** `2df2f07d10aa19c4d62694f21858be501a3d6d18`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 3 HIGH / 3 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/SKILLS.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/SKILLS.md`
- **Scope in:**
  - `SkillManifest` frozen contract (`skill_id`, `version`, `qualified_id`, `tool_ids`, dependencies, risk metadata)
  - `SkillCatalog` vs `SkillRegistry` vs `SkillProfile` host availability model
  - `build_registry_from_profile()` / `register_all_catalog_bundles` bootstrap semantics
  - `SkillResolver.validate_skills()` / `resolve_skills()` / `resolve()` version and identity behavior
  - `ResolvedSkillPack` provenance fields and `AgentRegistry.register()` consumption
  - `resolve_contract_tools()` contract merge into `AgentContract.allowed_tools`
  - `extend_tool_profile_for_skills()` and `ToolProfile.enabled` monotonic authority
  - `SkillProfile` structural validation vs unknown skill/bundle references
  - `AgentRegistry._bootstrap_default_skill_registry()` ambient fallback when `skill_registry` omitted
  - Skill / Tool / Agent / Integration responsibility separation
  - SK-BRIDGE prompt/policy helper partial wiring (positive control - honestly documented)
- **Scope out:**
  - remediation implementation
  - second Skill runtime design
  - SK-EXP catalog expansion delivery (historical **Done** facts)
  - universal SK-BRIDGE end-to-end consumption re-audit beyond documented partial state
  - full Governed Execution re-audit beyond skill-tool intersection touchpoints
  - TOOLS monotonic-authority remediation implementation (cross-layer coordinate only)
- **Prior audit reference(s):** SK-EXP through SK-EXP5 **Done**; AUDIT-IDEAL-12.1/12.2 **Done**; Protocol v2 [`TOOLS`](TOOLS.md) (TOOLS-01 monotonic tool authority - coordinate with SKILLS-03); [`AGENT_SYSTEM`](AGENT_SYSTEM.md) (contract integrity - separate layer)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `1d17272ceb2f486320e7265bfd62ca872961d74b`

## Executive summary

**Verdict: FAIL.** Six accepted findings (3 HIGH, 3 MEDIUM) show missing host `SkillRegistry` authority bootstraps the full first-party catalog via `register_all_catalog_bundles=True` instead of failing closed; agent-declared versioned `SkillManifest` references resolve only by `skill_id` so effective capability provenance can disagree with the registered declaration; `extend_tool_profile_for_skills()` silently enlarges host `ToolProfile.enabled` rather than validating skill requirements against existing host tool availability; `ResolvedSkillPack` is discarded after `allowed_tools` materialization limiting auditability; explicit unknown `SkillProfile` skill/bundle ids are not fail-fast; and architecture vs plan publish conflicting current catalog counts (153/43 vs 150/42). Positive controls: `SkillManifest` is frozen; resolver deterministically expands dependencies and rejects cycles/unknown deps; tool existence is validated when `ToolRegistry` supplied; effective risk tier is maximum across graph; Skill / Tool / Agent / Integration split is sound; Skills remain declarative composition not execution runtime; SK-BRIDGE prompt/policy incompleteness is honestly documented.

## Verdict

**FAIL** - 0 CRITICAL / 3 HIGH / 3 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-SKILLS-01

- **Severity:** HIGH
- **Category:** AUTHORIZATION / HOST-BOUNDARY DEFECT
- **Status at publication:** ACCEPTED
- **Claim falsified:** Production-shaped registration never interprets missing host Skill availability authority as "enable all"; canonical paths require explicit host registry/profile projection.
- **Substance:** `AgentRegistry.register()` automatically calls `_bootstrap_default_skill_registry()` whenever an agent declares skills / `extra_tools` but no `skill_registry` was injected. That fallback calls `build_registry_from_profile(SkillProfile(register_all_catalog_bundles=True))`, making the complete first-party Skill catalog available for resolution. Absence of Tier-3 `SkillRegistry` authority does not fail closed.
- **Evidence:**
  - `intergrax/runtime/registry/agent_registry.py` - `_bootstrap_default_skill_registry()` ambient fallback on missing injection
  - `intergrax/skills/registry/factory.py` - `build_registry_from_profile(SkillProfile(register_all_catalog_bundles=True))`
- **Confidence:** HIGH - direct code path; bootstrap is unconditional when skills declared and registry omitted.
- **Target invariant:** Production-shaped registration must never interpret missing host Skill availability authority as "enable all". If an all-catalog laboratory/bootstrap mode remains useful, it must be explicit, named, and not an ambient fallback.

### AUDIT-20260818-SKILLS-02

- **Severity:** HIGH
- **Category:** CONTRACT / VERSION IDENTITY DEFECT
- **Status at publication:** ACCEPTED
- **Claim falsified:** One explicit version-identity model - either version-pinned references resolve the exact declared version, or `AgentContract` declares logical skill identity and runtime/profile projection explicitly owns the resolved version.
- **Substance:** `AgentContract.skills` contains full versioned `SkillManifest` objects with `version` and `qualified_id = skill_id@version`. `SkillResolver.validate_skills()` checks only `manifest.skill_id` existence; `resolve_skills()` extracts only `skill_id`; `resolve()` uses registry manifest and current version/content. `SkillRegistry` is keyed only by `skill_id` with one active version per id. An agent can declare skill X@v1 while effective tools/dependencies/risk resolve from registry X@v2.
- **Evidence:**
  - `intergrax/skills/core/contracts.py` - `SkillManifest.version`, `qualified_id`
  - `intergrax/skills/resolver.py` - `validate_skills()` / `resolve_skills()` id-only checks and extraction
  - `intergrax/skills/registry/runtime.py` - single active version per `skill_id`
- **Confidence:** HIGH - version fields present on contract but ignored during resolution.
- **Target invariant:** Choose one explicit version-identity model (A: version-pinned resolution; B: logical identity with explicit runtime/profile version ownership). Do not preserve ambiguous mixture; do not invent compatibility aliases.

### AUDIT-20260818-SKILLS-03

- **Severity:** HIGH
- **Category:** AUTHORIZATION / CAPABILITY EXPANSION DEFECT
- **Status at publication:** ACCEPTED
- **Claim falsified:** Skill requirements may request/require tools but may not silently enlarge host `ToolProfile` availability authority.
- **Substance:** `extend_tool_profile_for_skills()` obtains every `tool_id` referenced by enabled skills and appends missing ids directly to `ToolProfile.enabled`. `ToolProfile` is the Tier-3 host tool availability contract; a Skill requirement can therefore expand host explicit tool availability rather than being validated against it.
- **Evidence:**
  - `intergrax/applications/_shared/skill_tool_profile.py` - `extend_tool_profile_for_skills()` appends to `enabled`
  - `intergrax/tools/registry/profile.py` - `ToolProfile.enabled` host authority surface
  - `intergrax/applications/_shared/skill_bridge_wiring.py` - Tier-3 wiring invocation path
- **Confidence:** HIGH - append semantics without intersection/fail-closed validation.
- **Target invariant:** Skill requirements ⊆ `ToolProfile` availability, otherwise fail static environment validation with actionable diagnostics. Coordinate with accepted TOOLS monotonic-authority invariant; do not create a second permission subsystem.

### AUDIT-20260818-SKILLS-04

- **Severity:** MEDIUM
- **Category:** PROVENANCE / CONTRACT COMPLETENESS GAP
- **Status at publication:** ACCEPTED
- **Claim falsified:** Resolved capability provenance required for execution/audit has one canonical durable or immutable runtime owner.
- **Substance:** `SkillResolver` produces `ResolvedSkillPack` (expanded `skill_ids`, effective `tool_ids`, `prompt_instruction_ids`, `policy_fragment_ids`, maximum `risk_tier`). `AgentRegistry.register()` receives this pack from `resolve_contract_tools()` but immediately discards it (`_ = resolved_pack`) after `allowed_tools` materialization. Canonical resolved capability/provenance is not retained or bound to registered agent/runtime revision.
- **Evidence:**
  - `intergrax/skills/resolver.py` - `ResolvedSkillPack` structure
  - `intergrax/skills/integration/contract_resolution.py` - `resolve_contract_tools()` returns pack
  - `intergrax/runtime/registry/agent_registry.py` - `_ = resolved_pack` discard after merge
- **Confidence:** HIGH - discard is explicit; architecture documents partial SK-BRIDGE consumption separately.
- **Target invariant:** Preserve or reference the canonical resolved snapshot; do not duplicate the Skill graph in several structures.

### AUDIT-20260818-SKILLS-05

- **Severity:** MEDIUM
- **Category:** CONFIGURATION VALIDATION GAP
- **Status at publication:** ACCEPTED
- **Claim falsified:** Explicit host configuration references fail fast during environment validation; unknown requested skill/bundle ids are not silently ignored.
- **Substance:** `SkillProfile` structurally validates fields but does not fail when explicit enabled skill ids do not exist in catalog/registry or enabled bundle ids do not exist. `build_registry_from_profile()` iterates existing catalog entries; unknown profile ids can produce unexpectedly empty/partial registry. `is_skill_enabled()` catches unknown bundle `KeyError` and continues.
- **Evidence:**
  - `intergrax/skills/registry/profile.py` - structural validation without reference existence checks
  - `intergrax/skills/registry/factory.py` - profile-driven registry build skips unknown ids silently
- **Confidence:** HIGH - validation gap observable from profile/factory interaction.
- **Target invariant:** Explicit host configuration references must fail fast during environment validation.

### AUDIT-20260818-SKILLS-06

- **Severity:** MEDIUM
- **Category:** DOCUMENTATION / EVIDENCE DRIFT
- **Status at publication:** ACCEPTED
- **Claim falsified:** Catalog counts derive from one authoritative gate/register; architecture and plan do not publish conflicting current counts.
- **Substance:** Current SKILLS architecture states gate-tested shipped catalog: **153** skills / **43** bundles. SKILLS implementation plan header still states **150** skills / **42** bundles. Both are canonical owning docs with conflicting current counts.
- **Evidence:**
  - `docs/project/architecture/SKILLS.md` - maturity boundary and catalog-proven table (**153** / **43**)
  - `docs/project/maintainers/plans/SKILLS.md` - plan header (**150** / **42**)
  - Gate: `test_sk_exp_skill_bundles.py` (referenced by architecture)
- **Confidence:** HIGH - direct doc comparison; authoritative register is gate-tested count in architecture.
- **Target invariant:** Single source of truth for current catalog counts; preserve historical counts only when explicitly labeled historical.

## Falsification log (negative results)

| Control | Result |
| ------- | ------ |
| `SkillManifest` frozen; rejects unknown fields | NOT falsified |
| `SkillResolver` deterministic transitive `requires_skills` expansion | NOT falsified |
| `requires_skills` cycles rejected | NOT falsified |
| Unknown skill dependencies rejected during resolution | NOT falsified |
| Tool existence validated when `ToolRegistry` supplied | NOT falsified |
| Effective risk tier = maximum across resolved graph | NOT falsified |
| Skill / Tool / Agent / Integration responsibility split sound | NOT falsified |
| Skills remain declarative composition, not execution runtime | NOT falsified |
| SK-BRIDGE prompt/policy bridge incompleteness honestly documented | NOT falsified |

## Prior-audit comparison

SK-EXP through SK-EXP5 and AUDIT-IDEAL-12.1/12.2 **Done** rows remain historical delivery facts. Protocol v2 SKILLS layer identifies authority-integrity, version/provenance, and evidence-sync gaps beyond prior harness closeout - not a retraction of shipped catalog scale proof.

## Provider / backend abstraction

`NOT APPLICABLE - SKILLS scope is declarative capability composition, host availability authority, and contract merge; external provider behavior is out of scope except where skill-required tools intersect ToolRuntime governance.`

## Positive controls

- `SkillManifest` is frozen and rejects unknown fields.
- `SkillResolver` deterministically expands transitive dependencies.
- `requires_skills` cycles are rejected.
- Unknown dependencies are rejected during resolution.
- Tool existence is validated when a `ToolRegistry` is supplied.
- Effective risk tier is maximum across resolved graph.
- Skill / Tool / Agent / Integration responsibility split is sound.
- Skills remain declarative composition, not an execution runtime.
- Prompt/policy bridge incompleteness is already honestly documented (SK-BRIDGE.1/2 partial).

## Root-cause remediation grouping

Remediation is **documentation-owned planning only** in this persistence task - no implementation.

### SKILLS-AUTHORITY-INTEGRITY - explicit host Skill/Tool availability, fail-closed bootstrap, profile consistency

**Findings:** 01, 03, 05

**Intent:** Production paths require explicit host `SkillRegistry` / `SkillProfile` projection; missing authority must not enable-all; skill tool requirements validate against `ToolProfile` availability without silent expansion; explicit unknown skill/bundle configuration fails fast.

### SKILLS-IDENTITY-PROVENANCE - version identity and resolved capability snapshot

**Findings:** 02, 04

**Intent:** One unambiguous Skill version-identity model; canonical `ResolvedSkillPack` provenance/risk snapshot retained or referenced for audit and execution binding.

### SKILLS-EVIDENCE-SYNC - catalog count single source of truth

**Findings:** 06

**Intent:** Architecture and plan publish aligned current catalog counts from one authoritative gate/register; historical counts labeled explicitly.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `2df2f07d10aa19c4d62694f21858be501a3d6d18`; current `development` HEAD was not re-audited.
- Audit persistence synchronizes architecture and plan target invariants only - **does not implement remediation**.
- SK-BRIDGE universal consumption gaps are documented partial state, not re-opened as new findings in this layer.

## Open questions / blocked items

- SKILLS-02: operator has not yet chosen model A (version-pinned) vs model B (logical identity + explicit runtime version ownership) - remediation must pick one explicitly.
- SKILLS-01: whether a named laboratory all-catalog bootstrap mode remains desirable after fail-closed production path - if yes, must be explicit and non-ambient.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 6 (`AUDIT-20260818-SKILLS-01` … `AUDIT-20260818-SKILLS-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none
