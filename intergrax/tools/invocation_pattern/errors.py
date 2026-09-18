# © Artur Czarnecki. All rights reserved.

"""Tool invocation pattern selection errors (public Tools domain)."""


class ToolInvocationPatternError(Exception):
    """Base error for tool invocation pattern resolution."""


class ToolInvocationPatternResolutionError(ToolInvocationPatternError):
    """Explicit ``tool_invocation_pattern_id`` could not be resolved (fail-closed)."""

    def __init__(self, pattern_id: str, *, reason: str = "not_found") -> None:
        self.pattern_id = pattern_id
        self.reason = reason
        super().__init__(f"Tool invocation pattern {pattern_id!r} could not be resolved ({reason})")
