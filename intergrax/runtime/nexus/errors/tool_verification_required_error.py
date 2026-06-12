# © Artur Czarnecki. All rights reserved.

"""Raised when a high-risk tool executes without required verification (TOOL-ENG-7)."""


class ToolVerificationRequiredError(RuntimeError):
    """Post-tool verify gate — host policy requires approval before HIGH+ tools proceed."""

    def __init__(
        self,
        *,
        run_id: str,
        tool_id: str,
        risk_level: str,
        message: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.tool_id = tool_id
        self.risk_level = risk_level
        super().__init__(
            message
            or (
                f"Tool '{tool_id}' (risk={risk_level}) requires verification before "
                f"continuing (run_id={run_id})."
            )
        )
