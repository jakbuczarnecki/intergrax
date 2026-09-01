# PLATFORM-PLUGIN-9 - Program Closeout Evidence

**Program:** Intergrax Platform Plugin Architecture - Global Audit, Unification & Third-Party Extensibility
**Evaluated commit:** `57c0985b1b3cf4f1bf9f8b8dc47cc0b965dddcc3` (pre-closeout baseline; PLUGIN-9 artifacts added on `development`)
**Scope:** PLATFORM-PLUGIN-1 through PLATFORM-PLUGIN-9
**Date:** 2026-08-12

---

## Executive result

| Field | Value |
|-------|-------|
| **Recommendation** | `PROGRAM_CLOSED` |
| **Evaluated branch** | `development` |
| **Required ancestors** | All PLUGIN-2..8 stage commits verified as ancestors |

The Platform Plugin program meets all twenty closeout criteria. No material architectural gaps block closure. Residual items are documented intentional limitations or future domain-owned work - not program defects.

---

## Program evidence (PLUGIN-1..9)

| Stage | Principal artifact | Status | Key proof |
|-------|-------------------|--------|-----------|
| PLUGIN-1 | `PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md` | Done | Extension inventory, taxonomy, DO-NOT-UNIFY baseline |
| PLUGIN-2 | `architecture/PLATFORM_PLUGINS.md` | Done | Frozen architecture, invariants §25, public matrix §20 |
| PLUGIN-3 | `intergrax/core/plugins/package_contract.py` | Done | `[tool.intergrax.plugin]` manifest; multi-capability metadata |
| PLUGIN-4 | `intergrax/core/plugins/discovery.py` | Done | Shared EP scan/load; security/policy/tool-invocation adoption |
| PLUGIN-5 | Architecture §12.3–§12.4; author guide §14 | Done | Domain-owned config/DI; no global container |
| PLUGIN-6 | `intergrax/core/plugins/platform_semantics.py` | Done | Compatibility, lifecycle vocabulary, conflict kinds |
| PLUGIN-7 | `intergrax/core/plugins/platform_qualification.py` | Done | Trust model, production gates, delivery sources |
| PLUGIN-8 | Reference wheel + host-embedded examples; E2E | Done | `test_plugin8_dual_mode_tool_e2e.py`; scaffold `extensions/` |
| PLUGIN-9 | This document + conformance suite + CI gate | Done | `tests/contract/core/plugins/test_platform_plugin_contract.py` |

---

## Architecture invariants (§25)

| # | Invariant | Result |
|---|-----------|--------|
| 1 | Domain capability contracts authoritative | PASS |
| 2 | Platform coordination does not bypass domain gates | PASS |
| 3 | Installed code trusted in-process | PASS |
| 4 | Installation ≠ activation | PASS |
| 5 | Discovery ≠ qualification | PASS |
| 6 | Qualification capability/domain-specific | PASS |
| 7 | Third-party extensions need no core source changes | PASS |
| 8 | Domain semantics not erased | PASS |
| 9 | Public contracts have compatibility ownership | PASS |
| 10 | Additive compatibility preserved | PASS |
| 11 | No single global EP group | PASS |
| 12 | Secrets not plugin metadata | PASS |
| 13 | Multi-capability packages allowed | PASS |
| 14 | No global DI container or secret API | PASS |

---

## Dual-mode proof

| Path | Evidence | Result |
|------|----------|--------|
| **External** | wheel → `iter_entry_point_specs` / `load_entry_point_plugins` → compatibility → `evaluate_package_production_admission` → `register_tool_plugin` → `RuntimeToolInvoker` | PASS - `test_plugin8_dual_mode_tool_e2e.py` |
| **Local** | `.py` module → `build_host_embedded_capability_subject` → `require_production_qualification` → `register_tool_plugin` → same invoker | PASS - E2E + scaffold gate |
| Same domain runtime | Both use `ToolPlugin`, `ToolWiringContext`, `build_registry_from_profile` | PASS |
| Qualification before registration | Generated `host/tool_wiring.py` calls `require_production_qualification` first | PASS - scaffold test + contract assertion |

---

## Public extension surface matrix (12 surfaces)

All twelve canonical surfaces in architecture §20.1 reviewed.

| Surface | Public contract works | EP group accurate | Local registration claim |
|---------|----------------------|-------------------|--------------------------|
| Integrations | Yes | `intergrax.integrations` | `register_integration_plugin()` - documented |
| Tools | Yes | `intergrax.tools` | Scaffold + `register_tool_plugin()` - executable |
| Skills | Yes | `intergrax.skills` | `register_skill_plugin()` - documented |
| Context | Yes (EP + registry) | `intergrax.context` | Host composition only - doc gap fixed (author guide) |
| Memory stores | Yes | `intergrax.memory_stores` | Factory callables - external-EP-first (acceptable) |
| RAG chunkers | Yes | `intergrax.rag.chunkers` | Bootstrap registry - external-EP-first (acceptable) |
| RAG retrievers | Yes | `intergrax.rag.retrievers` | Same (acceptable) |
| RAG rerankers | Yes | `intergrax.rag.rerankers` | Same (acceptable) |
| Vendor Knowledge | Yes | `intergrax.vendor_knowledge.providers` | Host builder - DO-NOT-UNIFY (acceptable) |
| Security defenses | Yes | `intergrax.security_defenses` | Profile/bootstrap - external-EP-first (acceptable) |
| Policy rules | Yes | `intergrax.policy_rules` | Bundle bootstrap - external-EP-first (acceptable) |
| Tool invocation patterns | Yes | `intergrax.tool_invocation_patterns` | Runtime config - external-EP-first (acceptable) |

**False public claims:** none found.
**Documentation gaps fixed:** Context status in `EXTENSION_AUTHOR_GUIDE.md` (was "Planned", now "Public EP - qualification rollout domain-owned").
**Material usability gaps:** local-registration helpers not scaffolded for all domains - classified **acceptable external-EP-first** per frozen architecture; not a closeout blocker.

---

## Deprecation result

### A. Intentionally preserved (DO-NOT-UNIFY §23)

Vendor Knowledge contribution catalog · `RuntimePlugin` host composition · `AgentRegistry` · RAG per-type registries · integration registry v2 · policy YAML + EP handlers · observability SDK · task execution registry · shipped integration manifest bootstrap · Tier-0 domain catalogs (shared loader utility only).

### B. Public legacy / duplicate paths

No competing public duplicate registration API was found that requires runtime `DeprecationWarning` in PLUGIN-9. Domain-specific loaders that differ from Tier-0 discovery are intentional, not duplicates.

| Path | Classification | Action |
|------|----------------|--------|
| Context author-guide "Planned" vs existing EP | Documentation drift | Fixed in PLUGIN-9 (guide table) |
| Token optimization descriptor (no loader) | IEP / future review | Documented in architecture §23; no deprecation |

### C. Internal implementation detail

Integration registry v2, embedding/document handler registries, hook internals - not public API; no user-facing deprecation.

**Breaking removals:** none in PLUGIN-9.
**Future obligations:** if a future domain program promotes additional local-registration scaffolds, that is domain-owned - not Platform Plugin program scope.

---

## CI gates

| Gate | Location | Protects |
|------|----------|----------|
| PR smoke - contract suite | `tests/contract/core/plugins/test_platform_plugin_contract.py` (via `ci_smoke` + explicit step) | Cross-stage invariants PLUGIN-3..8 |
| PR smoke - PLUGIN-8 E2E | `tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py` | External wheel + local embedded dual-mode |
| PR smoke - scaffold qualification | `tests/unit/scaffold/test_scaffold_local_extension_qualification.py` | Qualification before registration |
| Nightly gate | `pytest tests/unit -m "gate and not no_ci"` | Includes contract + scaffold gate tests |
| Existing unit suites | `test_platform_plugin_package_contract.py`, `test_plugin_discovery.py`, `test_platform_plugin_semantics.py`, `test_platform_plugin_qualification.py` | Stage-specific contracts (nightly full unit gate) |

**Workflow modified:** yes - `.github/workflows/unit-tests.yml` (`ci-smoke` job, step `Platform Plugin program gate (PLUGIN-9)`).
**Wheel E2E protected in PR smoke:** yes (`uv` available in CI via `setup-uv`).

---

## Security / trust statement

| Claim | Status |
|-------|--------|
| Trusted in-process Python execution | **Yes** - `PlatformPluginTrustModel.TRUSTED_IN_PROCESS` only |
| Sandbox / process isolation | **Not claimed** - architecture §18, §22 |
| Package signing / verification guarantee | **Not claimed** |
| Host/operator trust decision | **Yes** - qualification gates are explicit host decisions |

---

## Remaining risks (non-blocking)

1. **Local-registration scaffolds** exist only for Tools (and documented helpers for Integrations/Skills) - other domains remain external-EP-first by design.
2. **Context qualification rollout** remains domain-owned (CE-2); EP is public but production qualification posture varies by host.
3. **Token optimization plugin** descriptor has no production loader - tracked as future review, not Platform Plugin defect.
4. **Per-loader import isolation** varies by domain - bounded failure model documented; not a program closeout gap.

---

## Closeout criteria scorecard

| # | Criterion | Met |
|---|-----------|-----|
| 1 | Canonical architecture matches runtime | Yes |
| 2 | Package contract executable | Yes |
| 3 | Shared discovery semantics stable | Yes |
| 4 | Public EP groups accurately documented | Yes |
| 5 | Config/secrets/DI ownership clear | Yes |
| 6 | Lifecycle/compatibility vocabulary stable | Yes |
| 7 | Qualification distinct from discovery/enabled | Yes |
| 8 | Production gate fail-closed where required | Yes |
| 9 | External wheel path E2E | Yes |
| 10 | Local embedded path E2E | Yes |
| 11 | Scaffold gates qualification before registration | Yes |
| 12 | No second local-plugin framework | Yes |
| 13 | No global runtime wrapper | Yes |
| 14 | No sandbox claims | Yes |
| 15 | DO-NOT-UNIFY preserved | Yes |
| 16 | Cross-stage conformance tests pass | Yes |
| 17 | CI protects critical invariants | Yes |
| 18 | Documentation matches APIs | Yes (Context row fixed) |
| 19 | Duplicate legacy paths handled | Yes (none requiring runtime deprecation) |
| 20 | No material public-extension contradiction | Yes |

**Score:** 20 / 20

---

## Final recommendation

`PROGRAM_CLOSED`

The Platform Plugin program is complete. No PLUGIN-10 or further platform-level implementation stage is authorized. Future work on individual extension surfaces remains domain-owned.
