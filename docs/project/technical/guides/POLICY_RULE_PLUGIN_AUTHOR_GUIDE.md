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
| D9 | Registration / discovery | PARTIAL | §9 |
| D10 | Qualification | COMPLETE | §10 |
| D11 | Runtime use | PARTIAL | §11 |
| D12 | Lifecycle / cleanup | N/A | §12 |
| D13 | Failure behavior | COMPLETE | §13 |
| D14 | Testing | COMPLETE | §14 |
| D15 | Production checklist | COMPLETE | §15 |
| D16 | Troubleshooting | COMPLETE | §16 |

**Overall:** **PARTIAL** — contract, packaging, YAML schema, and local `PolicyRuleRegistry.register` are documented; **shipped Tier-3 wiring does not call `load_policy_rule_plugins`**, and declarative rules in `RuntimePolicyBundle.domain_fragments` are **not yet consumed by runtime enforcement** (see §11, RUNTIME_CAPABILITY_GAPS in audit ledger).

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
| `rule_id` | `str` | Selects handler — must match `PolicyRuleHandler.rule_id` |
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

- Unknown `rule_id` → `PolicyRuleAction.ALLOW` (fail-open for missing handler).
- Duplicate `rule_id` on `register()` → **silent overwrite** (last wins).

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
- rule_id: deny_tool
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
PolicyRulesProfile (rules_path + inline_rules)
  → _resolve_policy_rules() in policy_wiring.py
  → domain_fragments["policy_rules"]: list[DeclarativePolicyRule]
  → domain_fragments["policy_rule_registry"]: PolicyRuleRegistry (shipped wiring: empty of EP handlers)
  → RuntimePolicyBundle on ApplicationEnvironmentProfile
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
| `load_policy_rule_plugins(registry)` | **No** — host must call |
| `PolicyRuleRegistry.register(handler)` | Host / tests only |
| `load_policy_rules_from_path` | Yes — via `PolicyRulesProfile` when `wire_policy_bundle` runs |

EP discovery uses the same `INTERGRAX_DISCOVER_PLUGINS` / explicit `discover_entry_points` posture as other catalogs when the host invokes the loader.

### Loader failure isolation (AUDIT F004)

`load_policy_rule_plugins` is **fail-fast**:

- `PluginLoadError` on import failure → aborts entire handler load loop
- `TypeError` if target is not `PolicyRuleHandler` → aborts entire load
- No `on_load_failure="isolate"`

---

## 10. Qualification

Semantic host approval for third-party handler packages. Installing a handler wheel does not imply rules in YAML reference it or that the host enabled evaluation.

---

## 11. Runtime evaluation path (current)

**Intended flow:**

```text
Runtime event / tool request
  → host selects applicable DeclarativePolicyRule list
  → for each rule: registry.evaluate_rule(rule, context={"tool_id": ...})
  → PolicyRuleAction (ALLOW | DENY | REQUIRE_HITL)
  → enforcement owner (ToolRuntime / PolicyEngine / HITL) applies decision
```

**Current shipped wiring:**

- `domain_fragments["policy_rules"]` is populated when `PolicyRulesProfile` is set.
- `domain_fragments["policy_rule_registry"]` holds a registry with **shipped `deny_tool` handler only**.
- **No production code** calls `evaluate_rule` on that registry or iterates `policy_rules` fragments for enforcement.

Authors can unit-test evaluation today; end-to-end runtime enforcement through `PolicyEngine` for declarative EP/YAML rules is **not wired**. Document as `RUNTIME_CAPABILITY_GAP` — see audit §DOCS-5.

Evaluation is **separate from enforcement**: returning `DENY` from a handler has no effect until a host subscriber applies it.

---

## 12. Lifecycle / cleanup

No unload API. Registry is per-bundle instance.

---

## 13. Failure behavior

| Condition | Behavior |
|-----------|----------|
| Unknown `rule_id` | `evaluate_rule` → `ALLOW` |
| Handler missing for YAML `rule_id` | Same — fail-open |
| Malformed YAML / JSON | `ValueError` or `ValidationError` at load time |
| Duplicate handler `rule_id` | Silent overwrite on `register()` |
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
        rule_id="deny_tool",
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

- [ ] Handler `rule_id` matches YAML `rule_id` values
- [ ] YAML validated in CI (`load_policy_rules_from_path`)
- [ ] Host calls `load_policy_rule_plugins` if using EP handlers
- [ ] Host wires `evaluate_rule` + enforcement (not provided by shipped wiring alone)
- [ ] Understand fail-open on unknown handlers
- [ ] Duplicate handler ids audited — last `register` wins
- [ ] Qualification for third-party handler wheels
- [ ] Policy config separated from handler package install

---

## 16. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Handler never invoked | Shipped wiring does not call `load_policy_rule_plugins` or `evaluate_rule` |
| YAML rules ignored at runtime | Rules in `domain_fragments` only — no consumer wired |
| `ALLOW` despite deny YAML | No handler for `rule_id`, or `evaluate_rule` not called |
| `TypeError` at load | EP target not `PolicyRuleHandler` |
| `PluginLoadError` | Broken EP import — blocks entire policy EP load |
| Custom handler not used | `rule_id` mismatch with YAML |

---

## Enterprise gaps (documented, not implemented)

| Gap | Category |
|-----|----------|
| EP handler bootstrap not in `wire_policy_bundle` | EXTENSIBILITY / DX |
| Declarative rules not evaluated at runtime | GOVERNANCE |
| No centrally governed handler allowlist | GOVERNANCE / OPERATOR_CONTROL |
| No signed / provenance-tracked policy bundles | GOVERNANCE |

See `PLATFORM_PLUGIN_DOCUMENTATION_AUDIT.md` §DOCS-5 `ENTERPRISE_ROADMAP_CANDIDATES`.

**Reference example gap (DOCS-6):** no installable example under `examples/platform_plugins/`.
