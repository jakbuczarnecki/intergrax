# INTEGRAx-PLUGIN-CONFIGURATION-CONTRACT-SCOPED-RECERTIFICATION

## Metadata

| Field | Value |
|-------|-------|
| **Task** | `INTEGRAx-PLUGIN-CONFIGURATION-CONTRACT-SCOPED-RECERTIFICATION` |
| **Commit under certification** | `009ad0c62ba6afbd07a7e4f2dbe8b4bbbab1d2c6` (`fix(plugins): isolate decision plugin selection`) |
| **Accepted prior chain** | `ca9af5d9` → `de575ed90` |
| **Branch at evidence capture** | `development` |
| **HEAD at evidence capture** | `888584b987e7e4fc29a8a00d9f573395a0c51f6d` (later commits did not modify cert scope files) |
| **Date** | 2026-09-13 |

## Change classification

| Dimension | Verdict |
|-----------|---------|
| **Change type** | **CLASS C** — scoped public application configuration contract evolution |
| **Architecture reopen** | **SCOPED** (not FULL) — Decision plugin selection/admission path only |
| **Public configuration contract changed** | **YES** |
| **Backward compatible** | **NO** |
| **New production capability** | **NO** |

## Old contract (`*_kinds`)

`DecisionPluginProfile` previously accepted loose **kind id lists** (strings). Hosts named logical kinds (e.g. `council`) without binding to a specific distribution + entry-point locator. Runtime could **discover and evaluate** entry points broadly; unrelated broken installs could affect startup.

```text
DecisionPluginProfile
  verification_stage_kinds: list[str]
  strategy_kinds: list[str]
  artifact_kinds: list[str]
```

## New contract (`*_plugins`)

Hosts declare **typed, immutable** `PlatformPluginSelectionRef` per requested plugin. Semantics: *this application depends on this exact plugin identity at this entry-point locator in this distribution package*. Selection precedes import; only matched specs undergo manifest/capability admission.

```text
DecisionPluginProfile
  verification_stage_plugins: list[PlatformPluginSelectionRef]
  strategy_plugins: list[PlatformPluginSelectionRef]
  artifact_plugins: list[PlatformPluginSelectionRef]
```

`PlatformPluginSelectionRef` (`intergrax/core/plugins/selection_ref.py`): `extra="forbid"`, `frozen=True`; fields `plugin_id`, `entry_point_group`, `entry_point_name`, `distribution` (normalized via `normalize_distribution_package_name`); non-empty validation; `location_key` = `(distribution, entry_point_group, entry_point_name)`; matcher `entry_point_spec_matches_selection_ref`.

`DecisionPluginProfile` rejects duplicate `plugin_id` and duplicate `location_key` per list (`sub_profiles.py`).

## Breaking-change declaration

**Backward compatible: NO.** Legacy `*_kinds` keys are unknown fields → Pydantic validation error (`extra="forbid"` on `DecisionPluginProfile` and nested `PlatformPluginSelectionRef`). No silent ignore, no automatic kind→ref inference.

## Migration requirement

**Explicit operator migration required.** Do not add backward-compatibility shims for `*_kinds`. Map each former kind selection to a manifest-backed `plugin_id` plus distribution and entry-point locator discovered from the plugin package metadata.

### Before (invalid after `009ad0c`)

```yaml
decision_profile:
  plugins:
    discover_entry_points: true
    strategy_kinds:
      - council
```

### After (repo contract)

```yaml
decision_profile:
  plugins:
    discover_entry_points: true
    strategy_plugins:
      - plugin_id: external_council_variant
        entry_point_group: intergrax.decision_strategies
        entry_point_name: external_council
        distribution: my-decision-strategy-pkg
```

(Use the real `plugin_id` from `[tool.intergrax.plugin.capabilities]` in the plugin’s `pyproject.toml`.)

## Selection pipeline (certified semantics)

```text
Application profile (PlatformPluginSelectionRef lists)
  → DecisionPluginLoadPolicy (requested_*_plugins tuples when non-empty)
  → iter_entry_point_specs (metadata only)
  → locator resolve (requested path) / ignore unrequested EPs
  → manifest + capability binding
  → production admission (when policy requires)
  → load_entry_point_targets_for_specs (import admitted only)
  → domain registry merge
  → application_decision_composition (no vendor types in composition API)
```

**Not accepted:** discover all → import all → validate after import (unrestricted path exists only when `requested_plugins is None`; Tier-3 composition passes `None` only when the corresponding `*_plugins` list is **empty**, so explicit empty selection does not load plugins).

## Plugin isolation guarantee

| Class | Behavior |
|-------|----------|
| **Requested** | Must resolve to exactly one EP spec; manifest `plugin_id` must match; capability/admission must pass; else **fail closed** (STRICT composition raises). |
| **Unrequested** | Installed but not listed → **ignored**; not imported; cannot block startup (`test_decision_plugin_pre_load_admission.py`, `test_decision_plugin_selection_isolation.py`). |

## Fail-closed matrix

| Case | Expected | Evidence |
|------|----------|----------|
| Requested locator missing | Reject (`REQUESTED_PLUGIN_LOCATOR_NOT_FOUND`) | `test_requested_missing_locator_fails_closed` |
| Requested locator ambiguous | Reject (`REQUESTED_PLUGIN_LOCATOR_AMBIGUOUS`) | `plan_decision_plugin_admission` + reason codes in `admission.py` |
| Distribution mismatch | No match → locator not found | `entry_point_spec_matches_selection_ref` |
| Manifest `plugin_id` mismatch | Reject (`MANIFEST_PLUGIN_ID_MISMATCH`) | `test_requested_manifest_plugin_id_mismatch_fails_closed` |
| Capability mismatch | Reject (binding disposition) | pre-load admission tests |
| Admission denied | Reject (`PRODUCTION_ADMISSION_DENIED` etc.) | policy + composition tests |
| Unrequested broken plugin installed | Ignored | `test_unrelated_invalid_manifest_does_not_block_host`, isolation tests |
| Requested valid plugin | Admitted + loaded | composition + isolation tests |
| Legacy `*_kinds` in profile | Validation error | `test_legacy_kind_fields_rejected_by_extra_forbid` |
| Duplicate `plugin_id` / locator in profile | Validation error | `test_decision_plugin_profile_contract.py` |

No silent fallback to “load everything” when `*_plugins` is non-empty.

## Architecture compatibility

| Area | Changed |
|------|---------|
| Decision lifecycle, verification/deliberation semantics, governance, human review, decision authority | **NO** |
| `DecisionExecutionAuthorization`, `ExecutionRequest`, `ExecutionRuntime`, Execution Engine | **NO** |
| Canonical Decision → Governance → Execution path | **NO** |

`PlatformPluginSelectionRef` is a **declarative locator DTO**, not a service locator (no global registry lookup, no mutable singleton, no import-by-string without admission).

## Frozen invariants (INV-1…INV-10)

| Invariant | Result |
|-----------|--------|
| INV-1 Decision ≠ Execution | **PASS** |
| INV-2 Governance fail-closed | **PASS** |
| INV-3 Canonical Execution owner | **PASS** |
| INV-4 No parallel runtime | **PASS** |
| INV-5 Contracts first | **PASS** (profile + selection ref are contract models) |
| INV-6 Plugin extensibility | **PASS** (add plugin via EP + manifest + profile ref) |
| INV-7 Persistence abstraction | **PASS** (unchanged) |
| INV-8 Identity authority | **PASS** (`plugin_id` + manifest binding) |
| INV-9 Evidence ≠ control | **PASS** |
| INV-10 Qualification ≠ production | **PASS** |

## Test evidence

```bash
uv run pytest tests/unit/runtime/architecture/test_ds_plugin_architecture_gates.py -q
uv run pytest tests/unit/runtime/test_decision_plugin_composition.py -q
uv run pytest tests/unit/runtime/test_decision_plugin_pre_load_admission.py -q
uv run pytest tests/unit/runtime/test_decision_plugin_selection_isolation.py -q
uv run pytest tests/unit/applications/test_application_decision_composition.py -q
uv run pytest tests/unit/applications/test_decision_plugin_profile_contract.py -q
```

**Result (2026-09-13):** 78 + 5 = **83 passed**, 0 failed, 0 skipped (combined run in session log `.tmp/session/INTEGRAX-PLUGIN-CONFIGURATION-CONTRACT-SCOPED-RECERTIFICATION/pytest.log`).

## Static quality (scope)

```bash
uv run ruff check <scope files>
uv run ruff format --check <scope files>
uv run pyright <scope files>
```

**Observations:** Ruff reports pre-existing unused imports in `sub_profiles.py` and `application_decision_composition.py` (F401); format `--check` would reformat several scope files (cosmetic); Pyright reports one pre-existing trajectory provider typing issue in `application_decision_composition.py` — **not introduced by `009ad0c`**, no cert blocker.

## Baseline eligibility

`009ad0c62ba6afbd07a7e4f2dbe8b4bbbab1d2c6` = **ELIGIBLE CERTIFIED BASELINE CANDIDATE** (scoped plugin configuration contract), pending formal platform freeze. **Do not freeze in this task.**

## Final verdict

**SCOPED RECERTIFIED WITH OBSERVATIONS** (static-quality findings above; contract, isolation, and regression tests satisfied).
