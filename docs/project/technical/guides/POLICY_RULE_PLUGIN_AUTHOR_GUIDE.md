# Policy Rule Handler Plugin Author Guide

**Status:** canonical developer guide · **PLATFORM-PLUGIN-DOCS-5**
**Architecture owner:** [`docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md) (PolicyEngine §42.11)
**Platform catalog:** [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md) · [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

This guide documents the **author contract** for `PolicyRuleHandler` plugins and the **current** host wiring path. Policy handlers evaluate **declarative rules** (`DeclarativePolicyRule`); they are distinct from **Security Defense** hook middleware — see [`SECURITY_DEFENSE_PLUGIN_AUTHOR_GUIDE.md`](SECURITY_DEFENSE_PLUGIN_AUTHOR_GUIDE.md).

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

**Overall:** **COMPLETE** — contract, packaging, YAML schema, EP discovery, typed `DeclarativePolicyRuntime` composition, and declarative enforcement at the standard tool boundary are shipped. Production package/version qualification remains **deferred** (§10).

**Shared truths:** `installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified` · trusted in-process Python · host-owned qualification · no secrets in EP metadata · no sandbox.

---

## 1. Purpose — Policy vs Security Defense

| Surface | Evaluates | Runtime path |
|---------|-----------|----------------|
| **Policy rule handler** | `DeclarativePolicyRule` records (YAML / inline) keyed by `rule_id` | `PolicyRuleRegistry.evaluate_rule` → `PolicyRuleAction` |
| **Security defense** | Live `HookContext` at `HookPoint`s | `SecurityDefensePlugin.inspect` → middleware block/allow |
| **PolicyEngine** (facade) | Agent decisions, budgets, tool access, replay policies | `intergrax.runtime.policy.policy_engine` — separate from handler EP loading |

A package providing a handler **does not** activate a policy. YAML / `PolicyRulesProfile` and handler registration are **host-owned** composition steps.

---

## 2. Public contract

### Handler protocol

Import from `intergrax.runtime.policy.rules.registry`:

```python
@runtime_checkable
class PolicyRuleHandler(Protocol):
    rule_id: str

    def evaluate(
        self,
        rule: DeclarativePolicyRule,
        *,
        context: dict[str, str],
    ) -> PolicyRuleAction: ...
```

### Rule schema

`DeclarativePolicyRule` (`intergrax.runtime.policy.rules.schema`):

| Field | Type | Notes |
|-------|------|-------|
| `rule_id` | `str` | Configured rule instance identity (audit / evidence) |
| `handler_id` | `str` | Runtime handler lookup — must match `PolicyRuleHandler.rule_id` |
| `resource_kind` | `str` | e.g. `tool`, `agent`, `capability` |
| `resource_id` | `str` | Target id or `*` |
| `action` | `PolicyRuleAction` | `allow`, `deny`, `require_hitl` |
| `conditions` | `dict[str, Any]` | Handler-specific; shipped `deny_tool` ignores extra keys today |

### Shipped handler

`DenyToolRuleHandler` (`rule_id = "deny_tool"`) is registered by default on every new `PolicyRuleRegistry()`.

### Registry

```python
class PolicyRuleRegistry:
    def register(self, handler: PolicyRuleHandler) -> None: ...
    def evaluate_rule(
        self, rule: DeclarativePolicyRule, *, context: dict[str, str]
    ) -> PolicyRuleAction: ...
```

- Unknown `handler_id` → `PolicyRuleAction.DENY` (fail-closed; `unknown_handler=True` on outcome).
- Duplicate `rule_id` on `register()` → governed by admission policy (`error` default; shipped handler collision denied).
- Handler allowlist enforced at registration when configured on `PolicyRulesProfile`.

### EP loader

```python
def load_policy_rule_plugins(registry: PolicyRuleRegistry) -> int:
    """Register handlers from intergrax.policy_rules entry points."""
```

Entry point group:

```text
intergrax.policy_rules
```

EP semantics: class instantiated once; instance returned as-is (`test_plugin_discovery.py::test_policy_loader_preserves_class_and_instance_semantics`).

### YAML loading

```python
from intergrax.runtime.policy.rules.loader import load_policy_rules_from_path
```

Supports `.yaml`, `.yml`, `.json` — top-level **list** of rule objects.

### `PolicyRulesProfile` (Tier-3)

On `ApplicationEnvironmentProfile.policy_rules`:

| Field | Role |
|-------|------|
| `rules_path` | Path to YAML/JSON rules file |
| `inline_rules` | List of rule dicts merged after file rules |

---

## 3. Minimal implementation

Custom handler for the shipped `deny_tool` rule type:

```python
from intergrax.runtime.policy.rules.registry import PolicyRuleHandler
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule, PolicyRuleAction


class DenySandboxExecHandler:
    rule_id = "deny_tool"

    def evaluate(
        self,
        rule: DeclarativePolicyRule,
        *,
        context: dict[str, str],
    ) -> PolicyRuleAction:
        if rule.resource_kind != "tool":
            return PolicyRuleAction.ALLOW
        tool_id = context.get("tool_id", "")
        if rule.resource_id in ("*", tool_id) and rule.action == PolicyRuleAction.DENY:
            if tool_id == "sandbox.exec":
                return PolicyRuleAction.DENY
        return PolicyRuleAction.ALLOW
```

Example YAML (`applications/lab_application/policy/rules/harness_lab.yaml`):

```yaml
- rule_id: lab.sandbox.exec
  handler_id: deny_tool
  resource_kind: tool
  resource_id: sandbox.exec
  action: deny
  conditions: {}
```

---

## 4. External package

`pip install` does **not** activate policy rules or handlers.

### `pyproject.toml`

```toml
[project]
name = "acme-policy-rules"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["intergrax"]

[project.entry-points."intergrax.policy_rules"]
deny_sandbox = "acme_policy_rules.handlers:DenySandboxExecHandler"
```

### Three-part activation (package ≠ policy ≠ handler)

```text
1. pip install acme-policy-rules                    # handler code installed
2. Host builds PolicyRuleRegistry:
     registry = PolicyRuleRegistry()
     load_policy_rule_plugins(registry)              # EP handlers registered (host must call)
3. Host sets PolicyRulesProfile on ApplicationEnvironmentProfile:
     policy_rules = PolicyRulesProfile(rules_path=Path("policy/rules.yaml"))
4. wire_policy_bundle(env) → RuntimePolicyBundle with domain_fragments
```

**Current gap:** `wire_policy_bundle` / `build_runtime_policy_bundle` create `PolicyRuleRegistry()` but **do not** call `load_policy_rule_plugins`. Hosts that need EP handlers must invoke the loader explicitly and pass the populated registry into bundle composition (advanced composition — §5).

---

## 5. Local / host path

**Classification:** local `PolicyRuleRegistry.register()` supported; EP path requires explicit host bootstrap.

```python
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.plugin_loader import load_policy_rule_plugins

registry = PolicyRuleRegistry()  # includes shipped DenyToolRuleHandler
registry.register(DenySandboxExecHandler())
# Optional EP merge:
load_policy_rule_plugins(registry)
```

Pass the registry into `RuntimePolicyBundle.domain_fragments["policy_rule_registry"]` when building a custom bundle. Shipped `build_runtime_policy_bundle(policy_rules=profile)` always creates a **fresh** registry without EP handlers.

There is no `register_policy_rule_plugin()` catalog helper analogous to Tools.

---

## 6. Configuration / DI

```text
PolicyRulesProfile (rules_path + inline_rules + enforcement_mode + handler_allowlist)
  → policy_wiring.build_runtime_policy_bundle / wire_policy_bundle
  → DeclarativePolicyRuntime on RuntimePolicyBundle.declarative_policy_runtime:
       registry = PolicyRuleRegistry (shipped deny_tool + EP handlers when discovery enabled)
       rules = tuple[DeclarativePolicyRule, ...]
       provenance = PolicyBundleProvenance | None
       enforcement_mode = PolicyEnforcementMode
  → DeclarativePolicyEnforcer at standard tool invocation boundary
```

Tenant / application policy configuration belongs on **`ApplicationEnvironmentProfile.policy_rules`** (and domain-specific fragments), not in EP metadata.

---

## 7. Secrets / credentials

Policy handlers receive `context: dict[str, str]` — typically `tool_id`, agent ids, etc. No credentials in YAML or EP tables.

---

## 8. DI / composition

Handlers are stateless instances. Host may register the same handler class once. For testing, build an isolated `PolicyRuleRegistry` per bundle.

---

## 9. Registration / discovery

| Mechanism | Called from shipped Tier-3 wiring? |
|-----------|-----------------------------------|
| `load_policy_rule_plugin_report(registry, ...)` | **Yes** — when `policy_rules` set and `INTERGRAX_DISCOVER_PLUGINS` enabled |
| `load_policy_rule_plugins(registry)` | Legacy int wrapper; report API preferred |
| `PolicyRuleRegistry.register(handler)` | Host / tests; EP path via loader |
| `load_policy_rules_from_path` | Yes — via `PolicyRulesProfile` when `wire_policy_bundle` runs |

EP discovery uses the same `INTERGRAX_DISCOVER_PLUGINS` / explicit `discover_entry_points` posture as other catalogs.

When `policy_rules` is set but discovery is disabled, typed runtime is built with shipped `deny_tool` handler only and an empty load report.

### Loader failure isolation

Standard policy wiring uses `PolicyRuleLoadPolicy` with `on_load_failure="isolate"`:

- Broken EP → recorded in `DomainPluginLoadReport.failed`; siblings continue loading
- Invalid handler type → `rejected` in load report
- Legacy `load_policy_rule_plugins` without policy → fail-fast preserved for compatibility

---

## 10. Qualification

Semantic host approval for third-party handler packages. Installing a handler wheel does not imply rules in YAML reference it or that the host enabled evaluation.

**Production package/version qualification:** `QUALIFICATION_STILL_DEFERRED` — `evaluate_package_production_admission` requires caller-supplied package version and platform compatibility evidence not wired on the standard host path. Handler allowlist and provenance are shipped; automatic production qualification is **not** claimed.

---

## 11. Runtime evaluation path (current)

**Shipped flow:**

```text
Tool invocation request
  → DeclarativePolicyEnforcer (tool gateway boundary)
  → bundle.declarative_policy_runtime.rules
  → for each matching rule: registry.evaluate_rule(rule, context=PolicyEvaluationContext)
  → aggregate PolicyRuleAction (ALLOW | DENY | REQUIRE_HITL)
  → PolicyEnforcementMode.ENFORCE: DENY blocks tool; REQUIRE_HITL → canonical Nexus HITL lifecycle
  → PolicyEnforcementMode.AUDIT_ONLY: non-blocking audit trace only
```

Typed runtime is on `RuntimePolicyBundle.declarative_policy_runtime` — not `domain_fragments` string-key lookup.

Unknown handler → **DENY** fail-closed. `REQUIRE_HITL` reaches canonical Nexus pause/approve/resume (`WAITING_FOR_HUMAN`).

**Historical baseline (ENTERPRISE-1):** rules lived in `domain_fragments`; EP handlers were not loaded; unknown handler was fail-open. Closed by ENTERPRISE-3/4.

---

## 12. Lifecycle / cleanup

No unload API. Registry is per-bundle instance.

---

## 13. Failure behavior

| Condition | Behavior |
|-----------|----------|
| Unknown `rule_id` | `evaluate_rule` → `DENY` (`unknown_handler=True`) |
| Handler missing for YAML `rule_id` | Same — fail-closed |
| Malformed YAML / JSON | `ValueError` or `ValidationError` at load time |
| Duplicate handler `rule_id` | Admission policy governs (`error` default; shipped handler collision denied) |
| Non-allowlisted handler EP | Rejected at registration when allowlist configured |
| Handler raises in `evaluate` | Exception propagates to caller |
| `policy_rules` profile unset | No rules fragment; default registry still created when other fragments set |
| Disabled / empty policy bundle | No rules in fragments |

Policy **deny** is only effective when a host path calls `evaluate_rule` and enforces `PolicyRuleAction.DENY`.

---

## 14. Testing

| Test | Path |
|------|------|
| EP class vs instance | `tests/unit/core/plugins/test_plugin_discovery.py` |
| YAML lab rules | `applications/lab_application/policy/rules/harness_lab.yaml` |

```python
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule, PolicyRuleAction

registry = PolicyRuleRegistry()
registry.register(DenySandboxExecHandler())
action = registry.evaluate_rule(
    DeclarativePolicyRule(
        rule_id="lab.sandbox.exec",
        handler_id="deny_tool",
        resource_kind="tool",
        resource_id="sandbox.exec",
        action=PolicyRuleAction.DENY,
    ),
    context={"tool_id": "sandbox.exec"},
)
assert action == PolicyRuleAction.DENY
```

---

## 15. Production checklist

- [ ] Handler `rule_id` matches YAML `handler_id` values
- [ ] YAML validated in CI (`load_policy_rules_from_path`)
- [ ] `INTERGRAX_DISCOVER_PLUGINS` enabled when using EP handlers
- [ ] `PolicyEnforcementMode` set (`audit_only` vs `enforce`) per deployment profile
- [ ] Handler allowlist configured for production when required
- [ ] Understand fail-closed on unknown handlers
- [ ] Duplicate handler / shipped-handler collision policy understood
- [ ] Semantic qualification recorded for third-party wheels (package qualification deferred)
- [ ] Policy config separated from handler package install

---

## 16. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Handler never invoked | Discovery disabled; EP handler not in registry |
| YAML rules ignored at runtime | `PolicyEnforcementMode.AUDIT_ONLY` (audit-only, non-blocking) |
| Tool not blocked despite deny YAML | `AUDIT_ONLY` mode, or no matching rule/handler |
| `DENY` despite expected allow | Unknown handler fail-closed, or handler exception |
| `TypeError` at load | EP target not `PolicyRuleHandler` |
| `PluginLoadError` on broken EP import | Enterprise/policy-governed load (`PolicyRuleLoadPolicy`, default `on_load_failure="isolate"`) records failure; siblings may continue. Legacy `load_policy_rule_plugins(registry)` without policy → fail-fast blocks remaining EPs |
| Custom handler not used | `rule_id` mismatch with YAML |

---

## Historical baseline (ENTERPRISE-1 audit — closed)

| Gap (baseline) | Status |
|----------------|--------|
| EP handler bootstrap not in `wire_policy_bundle` | **CLOSED** (ENTERPRISE-3) |
| Declarative rules not evaluated at runtime | **CLOSED** (ENTERPRISE-4) |
| No centrally governed handler allowlist | **CLOSED** (ENTERPRISE-4) |
| No provenance-tracked policy bundles | **CLOSED** — `PolicyBundleProvenance` shipped; signing not required |
| Production package/version qualification | **DEFERRED** — `QUALIFICATION_STILL_DEFERRED` |

**Reference example gap (DOCS-6):** no installable example under `examples/platform_plugins/`.
