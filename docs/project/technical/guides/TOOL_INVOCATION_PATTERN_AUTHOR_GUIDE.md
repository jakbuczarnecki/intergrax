# Tool Invocation Pattern Author Guide

**Status:** canonical developer guide · **PLATFORM-PLUGIN-DOCS-5** · HARNESS-01-R5-W1
**Architecture owner:** [`docs/project/architecture/TOOLS.md`](../../architecture/TOOLS.md) · ADR-TOOL-003 · ADR-HARNESS-001
**Platform catalog:** [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md) · [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

This guide documents custom **`ToolInvocationPattern`** plugins - orchestration of **how** tool call batches run. A **Tool** defines **what** operation exists (`ToolPlugin` / `ToolContract`); a pattern defines planner→invoke→observe sequencing before atomic invoke-port calls.

**Public extension invariant:** Public extensions must not import `intergrax.runtime.nexus.*`. Nexus is an internal Execution Engine implementation detail. Canonical ABI: `intergrax.tools.invocation_pattern` (see [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md)).

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

**Overall:** **COMPLETE** for external-EP authoring. Host instance-override composition for custom patterns is **internal / migration pending** (Wave 3/6) - see §5.

**Shared truths:** `installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified` · trusted in-process Python · host-owned qualification · no secrets in EP metadata · no sandbox.

---

## 1. Purpose - Tool vs Invocation Pattern

| Layer | Contract | Question answered |
|-------|----------|-------------------|
| **Tool** (`ToolPlugin`) | `ToolContract` + handler | What operation can the LLM invoke? |
| **Invocation pattern** (`ToolInvocationPattern`) | `execute(...)` orchestration | How are one or more planned calls batched, looped, or parallelized? |
| **Atomic invoke** (`ToolInvocationInvokerPort`) | Public invoke port | How is a single prepared tool call executed? |

Shipped pattern ids (no EP required): `single_pass`, `bounded_react`, `parallel_batch`, `parallel_semantic_batch`, `deterministic_chain`. The Execution Engine selects a shipped pattern from host configuration; authors do not import Execution Engine internals to use these modes.

---

## 2. Public contract

Import from `intergrax.tools.invocation_pattern` (Tools-domain public ABI):

```python
from intergrax.tools.invocation_pattern import (
    ToolInvocationInvokerPort,
    ToolInvocationPattern,
    ToolInvocationPatternContext,
    ToolInvocationPatternResult,
    ToolInvocationPlannerPort,
    ToolInvocationStopReason,
    list_tool_invocation_pattern_ids,
    load_tool_invocation_pattern,
)
```

### `ToolInvocationPattern` protocol

```python
@runtime_checkable
class ToolInvocationPattern(Protocol):
    @property
    def pattern_id(self) -> str: ...

    def execute(
        self,
        *,
        context: ToolInvocationPatternContext,
        invoker: ToolInvocationInvokerPort,
        planner: ToolInvocationPlannerPort,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[ChatMessage],
    ) -> ToolInvocationPatternResult: ...
```

### Supporting public types

| Type | Role |
|------|------|
| `ToolInvocationPatternContext` | Neutral run view (`run_id`, `agent_id`, `user_message`, `tools_mode`, `max_parallel_tool_calls`) |
| `ToolInvocationInvokerPort` | `invoke_tool(agent_id=, request=) → ToolExecutionResult` |
| `ToolInvocationPlannerPort` | `plan_tools(...) → ToolPlanDecision` |
| `ToolInvocationPatternResult` | Orchestration outcome (`loop_iterations`, `stop_reason`, `pattern_id`, `appended_messages`, …) |

`ToolInvocationStopReason` literals: `empty_tool_calls`, `max_iterations`, `planner_final_answer`, `legacy_single_pass`.

The Execution Engine adapts this contract into its private orchestration layer after `execute` returns. Authors never receive or construct Nexus state types.

### Resolution APIs (public)

| API | Role |
|-----|------|
| `load_tool_invocation_pattern(pattern_id)` | Load custom pattern from EP by entry-point **name** |
| `list_tool_invocation_pattern_ids()` | Sorted EP names in `intergrax.tool_invocation_patterns` |

Entry point group:

```text
intergrax.tool_invocation_patterns
```

EP loader:

1. Scans entry-point specs until `spec.name == pattern_id`
2. Loads and instantiates the target (class → `()`, instance → as-is)
3. Validates `isinstance(..., ToolInvocationPattern)`
4. Returns `None` if name not found

**Note:** EP entry-point **name** must match the `pattern_id` the host enables for the run.

### Host configuration (author view)

Plugin authors enable a custom pattern by having the **host** select the entry-point id (application environment / tools profile). Do **not** import platform-internal runtime configuration types from Execution Engine packages.

Shipped mode ids are configuration tokens (`single_pass`, `bounded_react`, …). Mapping those tokens onto shipped pattern instances is an Execution Engine concern - not part of the public extension ABI.

---

## 3. Minimal implementation

```python
from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.invocation_pattern import (
    ToolInvocationInvokerPort,
    ToolInvocationPatternContext,
    ToolInvocationPatternResult,
    ToolInvocationPlannerPort,
)


class EchoOncePattern:
    @property
    def pattern_id(self) -> str:
        return "echo_once"

    def execute(
        self,
        *,
        context: ToolInvocationPatternContext,
        invoker: ToolInvocationInvokerPort,
        planner: ToolInvocationPlannerPort,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[ChatMessage],
    ) -> ToolInvocationPatternResult:
        _ = context, invoker, planner, plan, allowed_tool_ids, max_iterations, planner_input
        return ToolInvocationPatternResult(
            pattern_id=self.pattern_id,
            stop_reason="empty_tool_calls",
            loop_iterations=0,
        )
```

Do not invent retry/concurrency semantics beyond what your `execute` implementation provides. Parallelism for shipped modes is owned by the Execution Engine.

Golden proof package: [`intergrax_reference_enterprise_plugin`](../../../../examples/platform_plugins/intergrax_reference_enterprise_plugin/) (`reference_enterprise_single_pass`).

---

## 4. External package

`pip install` does **not** select a pattern - the host must reference the EP name.

### `pyproject.toml`

```toml
[project]
name = "acme-tool-patterns"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["intergrax"]

[project.entry-points."intergrax.tool_invocation_patterns"]
echo_once = "acme_tool_patterns.echo_once:EchoOncePattern"
```

Entry-point name `echo_once` must match the host-selected pattern id.

### Activation

```text
1. pip install acme-tool-patterns
2. INTERGRAX_DISCOVER_PLUGINS=1 (if other catalogs need discovery - pattern loader scans metadata at lookup time)
3. Host enables tool_invocation_pattern_id / equivalent profile field = "echo_once"
4. Execution Engine resolves the public pattern per request
```

Patterns are **not** registered at `bootstrap_catalogs` - lookup is **lazy** on each resolution when a custom pattern id is set.

---

## 5. Local / host path

**Classification:** host composition - **internal / migration pending** (HARNESS-01-R5 Waves 3/6).

Plugin authors are **not** host composition owners. There is no supported public authoring API that asks extension authors to call private tool-loop helpers or mutate Execution Engine configuration objects.

Until a neutral host/configuration surface is documented for Wave 6:

- Prefer the **external entry-point** path (§4) for discoverable custom patterns.
- Host-local instance override remains an **internal** composition concern - not a public golden path.

Do **not** call private Execution Engine tool-loop APIs from authoring code or tutorials.

---

## 6. Configuration

| Audience | Surface |
|----------|---------|
| **Plugin author** | Implement `ToolInvocationPattern`; publish EP name; document required host pattern id |
| **Host / application owner** | Select shipped mode id or custom `tool_invocation_pattern_id` via application environment / tools profile |

Resolution precedence (Execution Engine, conceptual):

```text
host instance override (composition-time, internal)
  → entry_point_pattern_id (EP)
  → shipped mode id
  → max_iterations > 1 → bounded_react
  → single_pass
```

Authors must not import or mutate Execution Engine-owned configuration types.

---

## 7. Secrets / credentials

Patterns receive `ToolInvocationInvokerPort` - credentials flow through tool wiring and integrations, not EP metadata.

---

## 8. DI / composition

`execute` receives `context`, `invoker`, `planner` - use these ports; do not construct global registries inside the pattern.

---

## 9. Registration / discovery

Custom patterns are discovered **at lookup time** via `importlib.metadata` entry points - not during `bootstrap_catalogs`.

### Performance (AUDIT F009)

`load_tool_invocation_pattern(pattern_id)` iterates **all** EP specs in the group on each lookup until the name matches - **O(N)** per resolution. Shipped mode selection does not scan EPs.

### Failure isolation

Pattern EPs are loaded **on demand**, not in a group bootstrap loop:

- Wrong type → `TypeError` at lookup time (fails that resolution only)
- Missing id → `None` → falls through to mode-based resolution
- Broken import on matching EP → `PluginLoadError` for that lookup

---

## 10. Qualification

Third-party pattern packages should pass host semantic qualification before production pattern-id assignment.

---

## 11. Runtime selection flow

```text
Execution Engine tools step
  → resolve public ToolInvocationPattern
        (EP id / shipped mode / iterations fallback)
  → ToolInvocationPattern.execute(
        context=ToolInvocationPatternContext,
        invoker=ToolInvocationInvokerPort,
        planner=ToolInvocationPlannerPort,
        ...
     )
  → ToolInvocationInvokerPort per planned call
  → ToolInvocationPatternResult → traces / messages / aggregate
```

---

## 12. Lifecycle / cleanup

Pattern instances are created per EP load or supplied by host composition. No unload API.

---

## 13. Failure behavior

| Condition | Behavior |
|-----------|----------|
| Unknown custom pattern id | `load_tool_invocation_pattern` returns `None` → fall back to mode / iterations |
| EP target not `ToolInvocationPattern` | `TypeError` |
| `execute` raises | Propagates - tool step fails (use public Tools-domain errors where applicable) |
| `empty_tool_calls` stop reason | Valid outcome - no tool traces |
| Unsupported shipped mode token | Execution Engine defaults to `single_pass` |

---

## 14. Testing

| Test | Path |
|------|------|
| Public ABI / bridge evidence | `tests/unit/runtime/nexus/tools/test_plug_02_r1_public_invocation_pattern_evidence.py` |
| EP load + resolution precedence | `tests/unit/runtime/nexus/tools/test_tool_invocation_registry.py` |
| Reference enterprise pattern | `tests/unit/platform_plugins/test_reference_enterprise_plugin.py` |
| Public authoring Nexus-free gate | `tests/qualification/harness_01/test_harness_01_public_authoring_surfaces.py` |

```python
from intergrax.tools.invocation_pattern import load_tool_invocation_pattern

loaded = load_tool_invocation_pattern("custom_pattern")
assert loaded is not None
assert loaded.pattern_id == "custom_pattern"
```

Unit-test a pattern by constructing `ToolInvocationPatternContext` and fakes for `ToolInvocationInvokerPort` / `ToolInvocationPlannerPort` - no Execution Engine imports required.

---

## 15. Production checklist

- [ ] `pattern_id` stable and matches EP name if using host pattern id
- [ ] Prefer shipped mode ids when possible - less EP surface
- [ ] Document `max_tool_iterations` interaction with mode for host owners
- [ ] Qualification for custom pattern wheels
- [ ] Understand O(N) EP scan if many patterns installed
- [ ] Pattern does not bypass `ToolInvocationInvokerPort` / tool scope policy
- [ ] No `intergrax.runtime.nexus.*` imports in the plugin package
- [ ] Traces include `pattern_id` on `ToolInvocationPatternResult`

---

## 16. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Shipped mode used instead of custom | Host pattern id typo; EP name ≠ `pattern_id` |
| `TypeError` on run | EP target does not implement `ToolInvocationPattern` |
| Pattern never loads | Package not installed; wrong EP group |
| Instance override ignored | Host composition not wired (internal / Wave 6) - use EP path |
| Slow resolution | Many EPs - O(N) scan per lookup (F009) |
| No tool traces | `stop_reason=empty_tool_calls` - planner returned no plan |

---

**Reference examples:** installable external EP - [`intergrax_reference_enterprise_plugin`](../../../../examples/platform_plugins/intergrax_reference_enterprise_plugin/) (`reference_enterprise_single_pass`).
