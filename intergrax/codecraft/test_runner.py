# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CraftTestRunner — run test command template in sandbox (ECC-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.tools.providers.sandbox._session import resolve_sandbox_session
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True)
class CraftTestResult:
    passed: bool
    skipped: bool
    command: str
    stdout: str
    stderr: str
    exit_code: int | None


class CraftTestRunner:
    """Execute profile test template inside an active sandbox session."""

    def __init__(self, profile: CodeCraftProfile) -> None:
        self._profile = profile

    def run(
        self,
        ctx: ToolWiringContext,
        *,
        rel_path: str = "craft_main.py",
    ) -> CraftTestResult:
        if not self._profile.require_tests:
            return CraftTestResult(
                passed=True,
                skipped=True,
                command="",
                stdout="",
                stderr="",
                exit_code=None,
            )

        session = resolve_sandbox_session(ctx)
        if session is None:
            return CraftTestResult(
                passed=False,
                skipped=False,
                command="",
                stdout="",
                stderr="sandbox_session_not_configured",
                exit_code=None,
            )

        command = self._profile.test_command_template.format(path=rel_path)
        if command.strip().startswith("pytest"):
            runner_code = (
                "import subprocess, sys\n"
                f"completed = subprocess.run({command.split()!r}, capture_output=True, text=True)\n"
                "print(completed.stdout, end='')\n"
                "print(completed.stderr, end='', file=sys.stderr)\n"
                "sys.exit(completed.returncode)\n"
            )
            result = session.execute(
                "run_python",
                {"code": runner_code, "language": "python", "timeout_s": 120},
            )
        else:
            result = session.execute(
                "run_script",
                {"path": rel_path, "args": [], "timeout_s": 120},
            )

        output = result.output or {}
        exit_code_raw = output.get("exit_code")
        exit_code = int(exit_code_raw) if exit_code_raw is not None else None
        passed = bool(result.success) and exit_code in (0, None)
        return CraftTestResult(
            passed=passed,
            skipped=False,
            command=command,
            stdout=str(output.get("stdout") or ""),
            stderr=str(output.get("stderr") or result.error or ""),
            exit_code=exit_code,
        )
