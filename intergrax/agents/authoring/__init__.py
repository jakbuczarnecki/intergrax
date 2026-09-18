# © Artur Czarnecki. All rights reserved.

"""Agent authoring facade (Phase DX-2.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.agents.authoring.base import IntergraxAgent
    from intergrax.agents.authoring.decorators import step as step_decorator

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "IntergraxAgent": ("intergrax.agents.authoring.base", "IntergraxAgent"),
    "complete": ("intergrax.agents.authoring.decisions", "complete"),
    "continue_to": ("intergrax.agents.authoring.decisions", "continue_to"),
    "continue_with": ("intergrax.agents.authoring.decisions", "continue_with"),
    "delegate_handoff": ("intergrax.agents.authoring.decisions", "delegate_handoff"),
    "delegate_to": ("intergrax.agents.authoring.decisions", "delegate_to"),
    "fail_step": ("intergrax.agents.authoring.decisions", "fail_step"),
    "finish": ("intergrax.agents.authoring.decisions", "finish"),
    "pause_for_human": ("intergrax.agents.authoring.decisions", "pause_for_human"),
    "request_replan": ("intergrax.agents.authoring.decisions", "request_replan"),
    "to_step_outcome": ("intergrax.agents.authoring.decisions", "to_step_outcome"),
    "step": ("intergrax.agents.authoring.decorators", "step"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> object:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = spec
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value
    return value
