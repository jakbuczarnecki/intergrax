# Tool Invocation Pattern Author Guide

**Status:** canonical developer guide · **PLATFORM-PLUGIN-DOCS-5**
**Architecture owner:** [`docs/project/architecture/TOOLS.md`](../../architecture/TOOLS.md) · ADR-TOOL-003
**Platform catalog:** [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md) · [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

This guide documents custom **`ToolInvocationPattern`** plugins — orchestration of **how** tool call batches run. A **Tool** defines **what** operation exists (`ToolPlugin` / `ToolContract`); a pattern defines planner→invoke→observe sequencing before atomic `RuntimeToolInvoker` calls.

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

**Overall:** **COMPLETE** for external-EP and local instance-override paths. Shipped modes (`ToolInvocationMode`) require no EP.

**Shared truths:** `installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified` · trusted in-process Python · host-owned qualification · no secrets in EP metadata · no sandbox.

---

## 1. Purpose — Tool vs Invocation Pattern

| Layer | Contract | Question answered |
|-------|----------|-------------------|
| **Tool** (`ToolPlugin`) | `ToolContract` + handler | What operation can the LLM invoke? |
| **Invocation pattern** (`ToolInvocationPattern`) | `execute(...)` orchestration | How are one or more planned calls batched, looped, or parallelized? |
| **Atomic invoke** (`RuntimeToolInvoker`) | Unchanged (Plane 2b) | How is a single tool call executed? |

Shipped patterns (no EP required): `single_pass`, `bounded_react`, `parallel_batch`, `parallel_semantic_batch`, `deterministic_chain` — see `ToolInvocationMode`.

---

## 2. Public contract

Import from `intergrax.runtime.nexus.tools.tool_invocation_pattern`:

### `ToolInvocationPattern` protocol

```python
@runtime_checkable
class ToolInvocationPattern(Protocol):
    @property
    def pattern_id(self) -> str: ...

    def execute(
        self,
        *,
        state: RuntimeState,
        invoker: RuntimeToolInvoker,
        planner: ToolPlannerProtocol,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[ChatMessage],
    ) -> ToolInvocationResult: ...
```

### `ToolInvocationResult`

Dataclass with `tool_traces`, `loop_iterations`, `stop_reason`, `pattern_id`, `appended_messages`, `used_native_tool_messages`, optional `aggregate`.

`ToolInvocationStopReason` literals include `empty_tool_calls`, `max_iterations`, `budget_exceeded`, `planner_final_answer`, `legacy_single_pass`.

### `ToolInvocationMode` (shipped patterns)

`intergrax.runtime.nexus.config_types.ToolInvocationMode`:

| Mode | Shipped class | `pattern_id` |
|------|---------------|--------------|
| `SINGLE_PASS` | `SinglePassPattern` | `single_pass` |
| `BOUNDED_REACT` | `BoundedReactPattern` | `bounded_react` |
| `PARALLEL_BATCH` | `ParallelBatchPattern` | `parallel_batch` |
| `PARALLEL_SEMANTIC_BATCH` | `ParallelSemanticBatchPattern` | `parallel_semantic_batch` |
| `DETERMINISTIC_CHAIN` | `DeterministicChainPattern` | `deterministic_chain` |

### Resolution APIs

| API | Role |
|-----|------|
| `pattern_for_mode(mode)` | Map `ToolInvocationMode` → shipped pattern instance |
| `resolve_invocation_pattern(mode=, max_iterations=, pattern_override=, entry_point_pattern_id=)` | Full resolution precedence |
| `load_tool_invocation_pattern(pattern_id)` | Load custom pattern from EP by entry-point **name** |
| `list_tool_invocation_pattern_ids()` | Sorted EP names in `intergrax.tool_invocation_patterns` |
| `shipped_pattern_ids()` | Frozenset of shipped mode values |

Entry point group:

```text
intergrax.tool_invocation_patterns
```

EP loader (`tool_invocation_registry.py`):

1. Scans `iter_entry_point_specs(EP_TOOL_INVOCATION_PATTERNS)` until `spec.name == pattern_id`
2. `load_entry_point_value` → `instantiate_entry_point_target` (class → `()`, instance → as-is)
3. Validates `isinstance(..., ToolInvocationPattern)`
4. Returns `None` if name not found

**Note:** EP entry-point **name** must match the `pattern_id` used in `RuntimeConfig.tool_invocation_pattern_id`.

### Runtime config (`RuntimeConfig`)

| Field | Precedence |
|-------|------------|
| `tool_invocation_pattern` | Instance override — **highest** |
| `tool_invocation_pattern_id` | EP lookup by id |
| `tool_invocation_mode` | Shipped `pattern_for_mode` |
| `max_tool_iterations` | When mode unset and `> 1` → `BoundedReactPattern` |

---

## 3. Minimal implementation

```python
from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan


class EchoOncePattern:
    @property
    def pattern_id(self) -> str:
        return "echo_once"

    def execute(
        self,
        *,
        state: RuntimeState,
        invoker: RuntimeToolInvoker,
        planner: ToolPlannerProtocol,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[ChatMessage],
    ) -> ToolInvocationResult:
        _ = state, invoker, planner, plan, allowed_tool_ids, max_iterations, planner_input
        return ToolInvocationResult(
            pattern_id=self.pattern_id,
            stop_reason="empty_tool_calls",
            loop_iterations=0,
        )
```

Do not invent retry/concurrency semantics beyond what your `execute` implementation provides. Parallelism for shipped modes is defined in `intergrax/runtime/nexus/tools/patterns/`.

---

## 4. External package

`pip install` does **not** select a pattern — runtime config must reference the EP name.

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

Entry-point name `echo_once` must match `tool_invocation_pattern_id="echo_once"`.

### Activation

```text
1. pip install acme-tool-patterns
2. INTERGRAX_DISCOVER_PLUGINS=1 (if other catalogs need discovery — pattern loader scans metadata at lookup time)
3. RuntimeConfig.tool_invocation_pattern_id = "echo_once"
4. run_bounded_tool_loop / ToolsStep resolves pattern per request
```

Patterns are **not** registered at `bootstrap_catalogs` — lookup is **lazy** on each resolution when `tool_invocation_pattern_id` is set.

---

## 5. Local / host path

**Classification:** advanced host composition — pass a pattern instance directly.

```python
from intergrax.runtime.nexus.tools.tool_loop import run_bounded_tool_loop

result = run_bounded_tool_loop(
    state=state,
    invoker=invoker,
    tool_planner=planner,
    planner_input="run",
    allowed_tool_ids=("demo.tool",),
    max_iterations=1,
    pattern=EchoOncePattern(),  # bypasses EP and mode
)
```

Or set on runtime config:

```python
state.context.config.tool_invocation_pattern = EchoOncePattern()
```

No `register_tool_invocation_pattern()` catalog helper — external-EP-first for discoverable ids, instance override for host-local patterns.

---

## 6. Configuration

| Config surface | Field |
|----------------|-------|
| `RuntimeConfig` | `tool_invocation_mode`, `tool_invocation_pattern_id`, `tool_invocation_pattern`, `max_tool_iterations`, `tool_chain_spec` (for `deterministic_chain`) |
| Agent / plan context | `plan_context_invocation.py` may override mode from adaptive recommendations |

Resolution order (`resolve_invocation_pattern`):

```text
pattern_override (instance)
  → entry_point_pattern_id (EP)
  → tool_invocation_mode (shipped)
  → max_iterations > 1 → bounded_react
  → single_pass
```

---

## 7. Secrets / credentials

Patterns receive `RuntimeToolInvoker` — credentials flow through tool wiring and integrations, not EP metadata.

---

## 8. DI / composition

`execute` receives `state`, `invoker`, `planner` — use these; do not construct global registries inside the pattern. Shipped patterns call `execute_planned_tool_calls` and planner APIs.

---

## 9. Registration / discovery

Custom patterns are discovered **at lookup time** via `importlib.metadata` entry points — not during `bootstrap_catalogs`.

### Performance (AUDIT F009)

`load_tool_invocation_pattern(pattern_id)` iterates **all** EP specs in the group on each lookup until the name matches — **O(N)** per resolution. Shipped `pattern_for_mode` does not scan EPs.

`ENTERPRISE_ROADMAP_CANDIDATE`: indexed/cached pattern registry — classified as **ordinary hardening** unless operator inventory requirements apply; see audit §DOCS-5 (priority suggestion: medium / hardening).

### Failure isolation

Pattern EPs are loaded **on demand**, not in a group bootstrap loop:

- Wrong type → `TypeError` at lookup time (fails that resolution only)
- Missing id → `None` → falls through to mode-based resolution
- Broken import on matching EP → `PluginLoadError` for that lookup

Unlike security/policy loaders, one broken **non-matching** EP does not block other patterns unless its import runs during scan (metadata load is lazy per matching spec only when name matches — unrelated broken EPs are skipped until their name is requested).

---

## 10. Qualification

Third-party pattern packages should pass host semantic qualification before production `tool_invocation_pattern_id` assignment.

---

## 11. Runtime selection flow

```text
ToolsStep / run_bounded_tool_loop
  → resolve_tool_invocation_pattern(
        mode=config.tool_invocation_mode,
        max_iterations=config.max_tool_iterations,
        pattern_override=config.tool_invocation_pattern,
        entry_point_pattern_id=config.tool_invocation_pattern_id,
     )
  → ToolInvocationPattern.execute(...)
  → RuntimeToolInvoker per planned call (unchanged)
  → ToolInvocationResult → traces / messages / aggregate
```

Adaptive hook (`tool_engine_hook`) may adjust `tool_invocation_mode` before resolution in `plan_context_invocation.py`.

---

## 12. Lifecycle / cleanup

Pattern instances are created per EP load or supplied by host. No unload API.

---

## 13. Failure behavior

| Condition | Behavior |
|-----------|----------|
| Unknown `tool_invocation_pattern_id` | `load_tool_invocation_pattern` returns `None` → fall back to mode / iterations |
| EP target not `ToolInvocationPattern` | `TypeError` |
| `execute` raises | Propagates — tool step fails |
| `empty_tool_calls` stop reason | Valid outcome — no traces |
| Unsupported mode | `pattern_for_mode` defaults to `SinglePassPattern` for unknown enum handling via last branch |
| Budget exceeded | Shipped patterns set `stop_reason="budget_exceeded"` where applicable |

---

## 14. Testing

| Test | Path |
|------|------|
| Shipped mode mapping | `tests/unit/runtime/nexus/tools/test_tool_invocation_pattern.py` |
| EP load + resolution precedence | `tests/unit/runtime/nexus/tools/test_tool_invocation_registry.py` |
| Parallel / chain patterns | `tests/unit/runtime/nexus/tools/test_parallel_batch_pattern.py`, etc. |

```python
from intergrax.runtime.nexus.tools.tool_invocation_registry import load_tool_invocation_pattern

loaded = load_tool_invocation_pattern("custom_pattern")
assert loaded is not None
assert loaded.pattern_id == "custom_pattern"
```

---

## 15. Production checklist

- [ ] `pattern_id` stable and matches EP name if using `tool_invocation_pattern_id`
- [ ] Prefer shipped `ToolInvocationMode` when possible — less EP surface
- [ ] Document `max_tool_iterations` interaction with mode
- [ ] Qualification for custom pattern wheels
- [ ] Understand O(N) EP scan if many patterns installed
- [ ] Pattern does not bypass `RuntimeToolInvoker` / tool scope policy
- [ ] Traces include `pattern_id` on `ToolInvocationResult`

---

## 16. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Shipped mode used instead of custom | `tool_invocation_pattern_id` typo; EP name ≠ `pattern_id` |
| `TypeError` on run | EP target does not implement `ToolInvocationPattern` |
| Pattern never loads | Package not installed; wrong EP group |
| Instance override ignored | Check `tool_invocation_pattern` set on correct `RuntimeConfig` |
| Slow resolution | Many EPs — O(N) scan per lookup (F009) |
| No tool traces | `stop_reason=empty_tool_calls` — planner returned no plan |

---

**Reference example gap (DOCS-6):** no installable package under `examples/platform_plugins/` — use unit-test `_CustomPattern` in `test_tool_invocation_registry.py`.
