# plan_to_langgraph.py
# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import TypedDict, Dict, Any, List, Callable, Optional
from dataclasses import asdict
import re  # NEW

# LangGraph
from langgraph.graph import StateGraph, END

from .supervisor import Plan, PlanStep
from .supervisor_components import BaseSupervisorComponent

# -----------------------------
# 1) State schema + utils
# -----------------------------
class PipelineState(TypedDict, total=False):
    """Global state traveling through the LangGraph pipeline."""
    query: str
    artifacts: Dict[str, Any]      # map: artifact name -> value
    step_status: Dict[str, Any]    # map: step_id -> {"status": "ok"|"error", "error": Optional[str]}
    debug_logs: List[str]          # chronological logs
    scratchpad: Dict[str, Any]     # free-form workspace

def _ensure_state_defaults(state: PipelineState) -> PipelineState:
    state.setdefault("artifacts", {})
    state.setdefault("step_status", {})
    state.setdefault("debug_logs", [])
    state.setdefault("scratchpad", {})
    return state

def _append_log(state: PipelineState, msg: str) -> None:
    state["debug_logs"].append(msg)

def _resolve_inputs(state: PipelineState, step: PlanStep) -> Dict[str, Any]:
    """
    Reads inputs declared in the plan. Convention:
      - bare name 'foo' → take from artifacts['foo'] if present, else from state['foo'] if present
      - names ending with '?' are optional (no error if missing)
      - the reserved name 'query' maps to state['query']
    """
    inputs: Dict[str, Any] = {}
    for name in (step.inputs or []):
        opt = name.endswith("?")
        key = name[:-1] if opt else name
        if key == "query":
            inputs[key] = state.get("query")
            continue
        if key in state.get("artifacts", {}):
            inputs[key] = state["artifacts"][key]
            continue
        if key in state:
            inputs[key] = state[key]
            continue
        if not opt:
            raise KeyError(f"Required input '{key}' for step {step.id} not found")
    return inputs

def _persist_outputs(state: PipelineState, step: PlanStep, result: Any) -> None:
    """
    Persists node results into artifacts. If result is a dict and step.outputs are named,
    we map names to result keys when possible; else we store the whole result under each named output.
    """
    artifacts = state["artifacts"]
    out_names = step.outputs or []
    if isinstance(result, dict) and out_names:
        for name in out_names:
            artifacts[name] = result.get(name, result)
    elif out_names:
        for name in out_names:
            artifacts[name] = result
    else:
        artifacts[step.id] = result

# -----------------------------
# 1b) Readable node names
# -----------------------------
def _slugify(text: str, max_len: int = 40) -> str:
    """
    Make a LangGraph-safe, readable identifier from a title.
    Keeps ASCII letters/numbers/underscores, collapses spaces, trims length.
    """
    text = (text or "").strip()
    if not text:
        return "step"
    s = re.sub(r"[^A-Za-z0-9]+", "_", text)   # non-alnum -> _
    s = re.sub(r"_+", "_", s).strip("_")       # collapse
    if not s:
        s = "step"
    return s[:max_len]

def _make_node_name(step: PlanStep, used: set) -> str:
    """
    Compose a unique node name from step.id and slugified title.
    Example: S1__UX_Audit_Report_Generation
    """
    base = f"{step.id}__{_slugify(step.title)}"
    name = base
    i = 2
    while name in used:
        name = f"{base}__{i}"
        i += 1
    used.add(name)
    return name

# -----------------------------
# 2) Node factory
# -----------------------------
def make_node_fn(step: PlanStep, component: Optional[BaseSupervisorComponent]) -> Callable[[PipelineState], PipelineState]:
    """
    Returns a LangGraph node function that executes one plan step.
    Resolution:
      - GENERAL → pure reasoning node (if a component was provided, we call it; else no-op synthesis scaffold)
      - TOOL/RAG → call provided component.instance.execute(**inputs)
    Contract for components:
      component.instance.execute(**inputs) -> Any (dict recommended)
    """
    method = (step.method or "GENERAL").upper()

    def node_fn(state: PipelineState) -> PipelineState:
        state = _ensure_state_defaults(state)
        _append_log(state, f"[{step.id}] START {method} | component={step.component or '∅'} | title={step.title}")

        try:
            resolved_inputs = _resolve_inputs(state, step)
            _append_log(state, f"[{step.id}] inputs = {resolved_inputs}")

            if method in ("TOOL", "RAG"):
                if not component or not getattr(component, "instance", None):
                    raise RuntimeError(f"Step {step.id} requires component '{step.component}' but it's unavailable.")
                exec_fn = getattr(component.instance, "execute", None)
                if not callable(exec_fn):
                    raise RuntimeError(f"Component '{component.name}' has no callable execute(...)")
                result = exec_fn(**resolved_inputs)
            else:
                if component and getattr(component, "instance", None):
                    exec_fn = getattr(component.instance, "execute", None)
                    if not callable(exec_fn):
                        raise RuntimeError(f"Component '{component.name}' has no callable execute(...)")
                    result = exec_fn(**resolved_inputs)
                else:
                    result = {"final_answer": resolved_inputs.get("draft_answer") or resolved_inputs.get("answer") or ""}

            _persist_outputs(state, step, result)
            state["step_status"][step.id] = {"status": "ok"}
            _append_log(state, f"[{step.id}] ok → outputs={step.outputs or [step.id]}")

        except Exception as e:
            state["step_status"][step.id] = {"status": "error", "error": repr(e)}
            _append_log(state, f"[{step.id}] ERROR: {e}")
        return state

    return node_fn

# -----------------------------
# 3) Build graph
# -----------------------------
def topo_order(steps: List[PlanStep]) -> List[PlanStep]:
    """
    Stable topological ordering:
      - primary: honor explicit depends_on
      - fallback: preserve original order when no dependencies given
    """
    if any(getattr(s, "depends_on", None) for s in steps):
        id_to_step = {s.id: s for s in steps}
        indeg = {s.id: 0 for s in steps}
        graph = {s.id: [] for s in steps}

        for s in steps:
            deps = getattr(s, "depends_on", []) or []
            for dep in deps:
                if dep in id_to_step:
                    graph[dep].append(s.id)
                    indeg[s.id] += 1

        queue = [sid for sid, d in indeg.items() if d == 0]
        ordered_ids: List[str] = []
        while queue:
            sid = queue.pop(0)
            ordered_ids.append(sid)
            for nxt in graph[sid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)

        if len(ordered_ids) != len(steps):
            remain = [s.id for s in steps if s.id not in ordered_ids]
            ordered_ids.extend(remain)

        return [id_to_step[sid] for sid in ordered_ids]

    return list(steps)

def build_langgraph_from_plan(
    plan: Plan,
    components_by_name: Dict[str, BaseSupervisorComponent],
    *,
    entry_inject: Optional[Dict[str, Any]] = None
):
    """
    Transforms a Plan into a runnable LangGraph pipeline.
    - Creates one node per PlanStep (function built by make_node_fn)
    - Connects nodes in a stable topological order (parallel branches are linearized)
    - Returns: (graph, compiled)
    """
    g = StateGraph(PipelineState)

    def _entry(state: PipelineState) -> PipelineState:
        state = _ensure_state_defaults(state)
        if entry_inject:
            state.update(entry_inject)
        return state

    g.add_node("ENTRY", _entry)

    ordered_steps = topo_order(plan.steps)
    node_names: List[str] = []
    used: set = set()

    for step in ordered_steps:
        comp = components_by_name.get(step.component) if step.component else None
        node_fn = make_node_fn(step, comp)
        # READABLE NAME instead of "NODE_S1"
        node_name = _make_node_name(step, used)   # e.g., "S1__UX_Audit"
        g.add_node(node_name, node_fn)
        node_names.append(node_name)

    g.set_entry_point("ENTRY")
    if node_names:
        prev = "ENTRY"
        for nn in node_names:
            g.add_edge(prev, nn)
            prev = nn
        g.add_edge(prev, END)
    else:
        g.add_edge("ENTRY", END)

    app = g.compile()
    return g, app
