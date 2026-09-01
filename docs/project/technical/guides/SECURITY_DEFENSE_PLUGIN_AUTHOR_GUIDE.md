# Security Defense Plugin Author Guide

**Status:** canonical developer guide · **PLATFORM-PLUGIN-DOCS-5**
**Architecture owner:** [`docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md) (security hook middleware)
**Platform catalog:** [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md) · [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

This guide is the **implementation workflow** for third-party Security Defense plugins. A defense plugin inspects runtime operations at declared security `HookPoint`s. It is **not** a `PolicyRuleHandler`, **not** a sandbox, and **not** cryptographic attestation — third-party code still runs as **trusted in-process Python**.

---

## Developer journey (D1–D16)

| D | Topic | Status | Section |
|---|-------|--------|---------|
| D1 | Purpose | COMPLETE | §1 |
| D2 | Public contract | COMPLETE | §2 |
| D3 | Minimal implementation | COMPLETE | §3 |
| D4 | External package | COMPLETE | §4 |
| D5 | Local / host path | COMPLETE | §5 |
| D6 | Configuration | COMPLETE | §6 |
| D7 | Secrets / credentials | N/A | §7 |
| D8 | DI / composition | COMPLETE | §8 |
| D9 | Registration / discovery | COMPLETE | §9 |
| D10 | Qualification | COMPLETE | §10 |
| D11 | Runtime use | COMPLETE | §11 |
| D12 | Lifecycle / cleanup | N/A | §12 |
| D13 | Failure behavior | COMPLETE | §13 |
| D14 | Testing | COMPLETE | §14 |
| D15 | Production checklist | COMPLETE | §15 |
| D16 | Troubleshooting | COMPLETE | §16 |

**Overall:** **COMPLETE** for the external-EP author path. Local registration uses advanced host composition (`register_security_defense_plugin`) — no Tools-style scaffold.

**Shared truths (all Platform Plugin surfaces):**

- `installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified`
- Third-party plugins run as **trusted in-process Python**
- Qualification is **host-owned semantic approval**, not cryptographic attestation
- Secrets do not belong in plugin metadata or entry-point values
- There is **no universal Platform Plugin lifecycle/unload manager** and **no sandbox**

---

## 1. Purpose — Security Defense vs Policy vs Sandbox

| Surface | Role |
|---------|------|
| **Security defense** (`SecurityDefensePlugin`) | Synchronous inspection at UAEP `HookPoint`s; may allow or block (or modify when `fail_mode=FAIL_OPEN`) via `SecurityInspectionResult` |
| **Policy rule handler** (`PolicyRuleHandler`) | Evaluates declarative policy rules in the PolicyEngine / bundle flow — see [`POLICY_RULE_PLUGIN_AUTHOR_GUIDE.md`](POLICY_RULE_PLUGIN_AUTHOR_GUIDE.md) |
| **Built-in V-SEC middleware** | Shipped prompt/tool/tenant defenses wired from `ApplicationSecurityProfile` toggles — not replaced by defense plugins |

Defense plugins are **hook-level security middleware**. They do not evaluate YAML policy bundles and do not isolate untrusted code.

---

## 2. Public contract

Import from `intergrax.runtime.security.defense_plugin`:

| Symbol | Role |
|--------|------|
| `SecurityDefensePlugin` | `@runtime_checkable` Protocol — author-facing contract |
| `SecurityInspectionResult` | Pydantic result: `allowed`, `reasons`, `plugin_id`, `hook_point` |
| `SecurityFailMode` | `FAIL_CLOSED` (default block) or `FAIL_OPEN` (modify with reason, do not block) |
| `PluginSecurityDefenseMiddleware` | Tier-1 wrapper that hosts invoke — runs `inspect` on a thread pool with timeout |
| `DEFAULT_DEFENSE_INSPECTION_TIMEOUT_MS` | Default wall-clock budget (`100` ms) |

`HookContext` and `HookResult` live in `intergrax.runtime.hooks.hook_context`. `HookPoint` enum in `intergrax.runtime.hooks.hook_point`.

### `SecurityDefensePlugin` required attributes

| Attribute | Type | Semantics |
|-----------|------|-----------|
| `plugin_id` | `str` | Stable id — used in profile enablement and registry |
| `version` | `str` | Author version string |
| `hook_points` | `frozenset[HookPoint]` | Points this plugin inspects; undeclared points are skipped |
| `priority` | `int` | Middleware ordering (lower runs earlier in `MiddlewarePipeline`) |
| `fail_mode` | `SecurityFailMode` | Behavior when `allowed=False` |

### Required method

```python
def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult: ...
```

`inspect` is **synchronous**. The middleware runs it via `asyncio.to_thread` with a timeout.

### Registry and bootstrap APIs

| API | Module | Role |
|-----|--------|------|
| `register_security_defense_plugin(plugin, *, override=False)` | `intergrax.runtime.security.defense_registry` | Register shipped or host-loaded instance |
| `get_security_defense_plugin(plugin_id)` | same | Lookup by id (shipped bundles or dynamic registry) |
| `resolve_security_defense_plugins(plugin_ids, bundle_ids)` | same | Resolve profile ids to plugin instances |
| `load_security_defense_plugin_report(*, discover_entry_points=True)` | `intergrax.runtime.security.defense_plugin_loader` | Load all `intergrax.security_defenses` EPs; returns typed admission report |
| `bootstrap_security_providers(*, discover_entry_points=False)` | `intergrax.core.security_bootstrap` | Catalog bootstrap entry — loads EPs when requested |

Entry point group:

```text
intergrax.security_defenses
```

EP target semantics (`instantiate_entry_point_target`): **class → instantiated once**; **pre-built instance → returned as-is** (see `test_plugin_discovery.py::test_security_loader_instantiates_class_targets_once`).

### `ApplicationSecurityProfile` (Tier-3 enablement)

On `ApplicationEnvironmentProfile.security_profile`:

| Field | Role |
|-------|------|
| `defense_bundle_ids` | Shipped bundle ids (e.g. `harness.strict_injection`) |
| `defense_plugin_ids` | Explicit dynamic / EP plugin ids |
| Other V-SEC toggles | `prompt_defense_enabled`, `tool_injection_defense_enabled`, … — independent of defense plugins |

Enablement is **profile-driven**: discovered EP plugins are registered in the global defense registry but attached to the Nexus middleware pipeline only when their id appears in `defense_plugin_ids` or `defense_bundle_ids` (via `register_application_security_hooks`).

---

## 3. Minimal implementation

**Test fixture — packaging reference, not production sample.**

```python
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.security.defense_plugin import (
    SecurityFailMode,
    SecurityInspectionResult,
)


class BlockJailbreakDefense:
    plugin_id = "acme.block_jailbreak"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 57
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        if point != HookPoint.BEFORE_TOOL_CALL:
            return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)
        arguments = ctx.runtime_state.get("arguments")
        if isinstance(arguments, dict):
            blob = " ".join(str(v).lower() for v in arguments.values())
            if "jailbreak" in blob:
                return SecurityInspectionResult(
                    allowed=False,
                    reasons=["blocked token: jailbreak"],
                    plugin_id=self.plugin_id,
                    hook_point=point.value,
                )
        return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)
```

Shipped reference implementation: `harness.strict_injection` in `intergrax/runtime/security/defense_registry.py`.

---

## 4. External package

`pip install` does **not** activate a defense plugin. Discovery and profile enablement are separate steps.

### Package structure

```text
acme_security_defense/
  pyproject.toml
  src/acme_security_defense/
    __init__.py
    plugin.py          # SecurityDefensePlugin implementation
```

### `pyproject.toml`

```toml
[project]
name = "acme-security-defense"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["intergrax"]

[project.entry-points."intergrax.security_defenses"]
block_jailbreak = "acme_security_defense.plugin:BlockJailbreakDefense"
```

Entry-point **name** is metadata; **enablement** uses `plugin_id` on the class (`acme.block_jailbreak` in the example above).

### Activation sequence

```text
1. pip install acme-security-defense          # installed
2. INTERGRAX_DISCOVER_PLUGINS=1               # or bootstrap_catalogs(discover_entry_points=True)
3. bootstrap_catalogs() / bootstrap_security_providers(discover_entry_points=True)  # discovered → registry
4. ApplicationSecurityProfile.defense_plugin_ids = ["acme.block_jailbreak"]  # enabled on host
5. register_application_security_hooks(nexus, profile, …)   # host runtime wiring
```

---

## 5. Local / host path

**Classification:** advanced host composition (no Tools-style scaffold).

Hosts may register a defense plugin without an entry point:

```python
from intergrax.runtime.security.defense_registry import register_security_defense_plugin
from intergrax.applications._shared.application_security_wiring import register_application_security_hooks

register_security_defense_plugin(BlockJailbreakDefense())  # override=False by default
register_application_security_hooks(
    nexus,
    profile,  # defense_plugin_ids must include BlockJailbreakDefense.plugin_id
    options=wiring.options,
)
```

`register_security_defense_plugin(..., override=True)` is required to replace a **shipped** bundle id on explicit host registration. EP loader uses `SecurityDefenseAdmissionPolicy` — shipped-id override denied by default (`shipped_id_override="error"`); authorized override requires explicit policy (`allow` / `warn_override`) or `LEGACY_UNCONDITIONAL_OVERRIDE_POLICY` (§9).

There is **no** `register_security_defense_plugin` helper exposed as a first-class local-plugin scaffold — parity with Tools is intentionally absent.

---

## 6. Configuration and runtime flow

```text
ApplicationSecurityProfile.defense_plugin_ids / defense_bundle_ids
  → resolve_security_wiring_options() / security_runtime_bridge
  → resolve_security_defense_plugins(ids, bundle_ids)
  → PluginSecurityDefenseMiddleware per plugin
  → MiddlewarePipeline on NexusLoop (sorted by priority)
  → before(hook_point, HookContext) on each matching HookPoint
  → SecurityInspectionResult.allowed → HookResult ALLOW | BLOCK | MODIFY
```

Tenant scope: when `enforce_tenant_scope=True` (default), middleware blocks if `tenant_id` ≠ `resource_tenant_id` in `HookContext.runtime_state` before calling `inspect`.

Blocks emit `platform.security.defense_blocked` on the runtime event bus when an `event_bus` is wired.

---

## 7. Secrets / credentials

Defense plugins receive operation context via `HookContext.runtime_state` only. Do not read secrets from EP metadata. Host injects integration-backed values into `runtime_state` when needed.

---

## 8. DI / composition

Defense plugins are **stateless or self-contained** instances. The host does not inject a wiring context. Prefer reading `ctx.runtime_state` keys documented for each `HookPoint` (e.g. `tool_id`, `arguments` at `BEFORE_TOOL_CALL`).

---

## 9. Registration / discovery

| Step | API |
|------|-----|
| Scan EP group | `iter_entry_point_specs("intergrax.security_defenses")` |
| Load + register | `load_security_defense_plugin_report(discover_entry_points=True)` |
| Catalog bootstrap | `bootstrap_catalogs(discover_entry_points=True)` also calls `bootstrap_security_providers` |

Default production posture: `discover_entry_points=False` until `INTERGRAX_DISCOVER_PLUGINS` is set (same as other Tier-0 catalogs).

### Conflict semantics — `SecurityDefenseAdmissionPolicy`

`load_security_defense_plugin_report` uses configurable `SecurityDefenseAdmissionPolicy` (production default: fail-closed):

```python
@dataclass(frozen=True)
class SecurityDefenseAdmissionPolicy:
    ep_name_conflict: ConflictPolicy = "error"
    plugin_id_conflict: ConflictPolicy = "error"
    shipped_id_override: Literal["error", "warn_override", "allow"] = "error"
    on_load_failure: LoadIsolation = "isolate"
```

| Scenario | Current behavior (production default) |
|----------|--------------------------------------|
| Duplicate EP name | `PluginConflictError` (`ep_name_conflict="error"`) |
| Duplicate EP `plugin_id` | `error` (`plugin_id_conflict="error"`) |
| EP `plugin_id` collides with **shipped** bundle | **Denied** unless `shipped_id_override="allow"` or `"warn_override"` |
| Host `register_security_defense_plugin` without `override` on shipped id | `ValueError: cannot override shipped defense plugin` |
| Host duplicate dynamic id without `override` | `ValueError: defense plugin already registered` |

**Legacy migration:** `LEGACY_UNCONDITIONAL_OVERRIDE_POLICY` restores pre-ENTERPRISE-2 unconditional EP override + fail-fast load semantics for lab hosts only.

Discovery remains **opt-in** (`discover_entry_points=False` default; `INTERGRAX_DISCOVER_PLUGINS` opt-in).

---

## 10. Qualification

`installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified`.

Host applications should gate third-party defense plugins through semantic qualification where the host applies it (`evaluate_package_production_admission` / `require_production_qualification`). Qualification is **not** cryptographic attestation.

**Production package/version qualification:** `QUALIFICATION_STILL_DEFERRED` on the standard host path — automatic production admission is not wired. Semantic host approval remains operator-owned.

---

## 11. Runtime behavior summary

| `SecurityInspectionResult` | `fail_mode` | Middleware `HookResult` |
|----------------------------|-------------|-------------------------|
| `allowed=True` | any | `ALLOW` |
| `allowed=False` | `FAIL_CLOSED` | `BLOCK` + reason |
| `allowed=False` | `FAIL_OPEN` | `MODIFY` + reason (does not block) |
| Inspection timeout | any | `BLOCK` |
| Tenant scope mismatch | any | `BLOCK` (before `inspect`) |
| `point` not in `hook_points` | any | `ALLOW` (plugin not invoked) |

Plugin exceptions inside `inspect` propagate from the worker thread and are **not** converted to fail-open — treat as runtime failure.

---

## 12. Lifecycle / cleanup

No unload API. Dynamic registry can be cleared in tests via `reset_security_defense_registry_for_tests()` only.

---

## 13. Failure behavior and loader isolation

`load_security_defense_plugin_report` uses `SecurityDefenseAdmissionPolicy.on_load_failure` (production default: **`isolate`**):

| Failure | Behavior (default policy) |
|---------|---------------------------|
| `discover_entry_points=False` | Returns empty report; no EP scan |
| `load_entry_point_value` error | Recorded in `DomainPluginLoadReport.failed`; siblings continue |
| Target not `SecurityDefensePlugin` | `rejected` in load report |
| Shipped-id collision without authorized override | `rejected` |
| Profile references unknown `defense_plugin_id` | Plugin skipped at resolve time; **STRICT** execution mode may fail assembly validation |
| Profile enables bundle id with no registry entry | Silently skipped in `resolve_security_defense_plugins` |

**Legacy:** `LEGACY_UNCONDITIONAL_OVERRIDE_POLICY` uses fail-fast load (`on_load_failure="fail_fast"`).

Production hosts load EPs through `bootstrap_security_providers` (report-based path). Use `load_security_defense_plugin_report` directly in tests and advanced host composition.

---

## 14. Testing

| Test | Path |
|------|------|
| EP class vs instance | `tests/unit/core/plugins/test_plugin_discovery.py` |
| Shipped bundle + middleware block | `tests/unit/runtime/security/test_sec_planes.py` |
| Catalog bootstrap discovers fixture EP | `tests/unit/runtime/security/test_sec_planes_evol.py` |
| Fixture package | `tests/fixtures/plugin_packages/intergrax_security_defense_fixture/` |

Example unit test pattern:

```python
from intergrax.runtime.security.defense_plugin_loader import load_security_defense_plugin_report
from intergrax.runtime.security.defense_registry import get_security_defense_plugin, reset_security_defense_registry_for_tests

reset_security_defense_registry_for_tests()
report = load_security_defense_plugin_report(discover_entry_points=True)
assert any(item.name == "fixture_ep.defense" for item in report.accepted)
assert get_security_defense_plugin("fixture_ep.defense") is not None
```

---

## 15. Production checklist

- [ ] `plugin_id` stable and documented for operators
- [ ] `hook_points` minimal — only points you inspect
- [ ] `fail_mode` explicitly chosen; `FAIL_OPEN` requires product sign-off
- [ ] `inspect` completes within inspection timeout (default 100 ms)
- [ ] `INTERGRAX_DISCOVER_PLUGINS` documented for deployment
- [ ] `ApplicationSecurityProfile.defense_plugin_ids` / `defense_bundle_ids` list only intended plugins
- [ ] Qualification recorded for third-party wheels (package qualification deferred)
- [ ] Shipped-id override policy understood — EP cannot silently replace shipped defenses by default
- [ ] No secrets in EP metadata or `pyproject.toml` plugin tables

---

## 16. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Plugin never runs | Not in `defense_plugin_ids` / `defense_bundle_ids`, or `register_application_security_hooks` not called |
| Plugin not in registry | Discovery disabled; run `bootstrap_security_providers(discover_entry_points=True)` |
| Wrong plugin behavior | Check `SecurityDefenseAdmissionPolicy` — shipped override denied by default |
| `PluginLoadError` at bootstrap | Broken EP import — with default policy, isolated in `failed` report; legacy fail-fast if `LEGACY_UNCONDITIONAL_OVERRIDE_POLICY` |
| Always blocked before `inspect` | Tenant scope mismatch in `runtime_state` |
| Timeout blocks | `inspect` too slow — optimize or reduce work |
| `ValueError: cannot override shipped` | Host registration without `override=True` on shipped id |

---

**Reference example gap (DOCS-6):** no installable package under `examples/platform_plugins/` — use `tests/fixtures/plugin_packages/intergrax_security_defense_fixture/` (test fixture only).
