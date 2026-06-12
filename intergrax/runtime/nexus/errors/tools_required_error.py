# © Artur Czarnecki. All rights reserved.

"""Raised when tools_mode=required but no tool executed (TOOL-ENG-8)."""


class ToolsRequiredError(RuntimeError):
    """Host policy requires at least one tool invocation per run."""

    def __init__(self, *, run_id: str, message: str | None = None) -> None:
        self.run_id = run_id
        super().__init__(message or f"tools_mode='required' but no tools were executed (run_id={run_id}).")
