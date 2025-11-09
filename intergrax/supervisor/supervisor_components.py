# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional, TypedDict

# Reuse your PipelineState definition if it already exists. Otherwise:
class PipelineState(TypedDict, total=False):
    query: str
    artifacts: Dict[str, Any]
    step_status: Dict[str, Any]
    debug_logs: List[str]
    last_output: Any      # last result (shortcut for quick hand-off to the next step)
    cursor: Dict[str, Any]  # arbitrary pointers/offsets passed between steps

@dataclass
class ComponentResult:
    ok: bool = True
    # What to merge into state.artifacts (e.g., {"ux_report": {...}, "answer": "..."}).
    produces: Dict[str, Any] = field(default_factory=dict)
    # Optional "main result" — automatically stored in state["last_output"].
    output: Any = None
    # Diagnostic logs.
    logs: List[str] = field(default_factory=list)
    # Additional entries for step_status (e.g., {"details": "..."}); status is inferred from `ok`.
    meta: Dict[str, Any] = field(default_factory=dict)

# Context passed to components — they can pull LLMs, retrievers, etc. from here.
@dataclass
class ComponentContext:
    llm_adapter: Any = None
    resources: Dict[str, Any] = field(default_factory=dict)  # e.g., retriever, ranker, etc.

@dataclass
class Component:
    name: str
    description: str
    use_when: str = ""
    examples: List[str] = field(default_factory=list)
    available: bool = True
    # Actual step implementation.
    fn: Callable[[PipelineState, ComponentContext], ComponentResult] = None

    def run(self, state: PipelineState, ctx: ComponentContext) -> ComponentResult:
        if not self.available:
            return ComponentResult(ok=False, logs=[f"{self.name} unavailable"], meta={"error": "unavailable"})
        try:
            return self.fn(state, ctx)
        except Exception as e:
            return ComponentResult(ok=False, logs=[f"{self.name} error: {e}"], meta={"error": str(e)})

# Syntactic sugar: decorator for quick registration.
def component(name: str, description: str, use_when: str = "", examples: Optional[List[str]] = None, available: bool = True):
    def wrap(fn):
        return Component(
            name=name,
            description=description,
            use_when=use_when,
            examples=examples or [],
            available=available,
            fn=fn
        )
    return wrap
