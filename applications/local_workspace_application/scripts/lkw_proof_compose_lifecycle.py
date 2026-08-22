#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Terminal Compose lifecycle for standalone public LKW proofs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lkw_host_port_preflight import (  # noqa: E402
    KnownIntergraxStackDefinition,
    known_intergrax_stack_definitions,
    non_destructive_compose_down_args,
)

_APP_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_DIR.parent.parent
_DOCKER_DIR = _APP_DIR / "docker"

_BACKGROUND_TASK_STACK_ID = "lkw-background-task-proof"
_OS_INTERACTION_STACK_ID = "lkw-os-interaction-proof"
_HOSTING_STACK_ID = "lkw-hosting-proof"
_FILE_WATCHER_STACK_ID = "lkw-file-watcher-e2e-proof"


@dataclass(frozen=True, slots=True)
class TerminalTeardownOutcome:
    attempted: bool
    result: str
    error_type: str | None = None


def resolve_known_stack(
    stack_id: str,
    *,
    docker_dir: Path | None = None,
) -> KnownIntergraxStackDefinition:
    resolved_docker_dir = docker_dir or _DOCKER_DIR
    for stack in known_intergrax_stack_definitions(resolved_docker_dir):
        if stack.stack_id == stack_id:
            return stack
    raise KeyError(f"unknown_stack:{stack_id}")


def _default_run_command(
    command: Sequence[str],
    *,
    cwd: str,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    timeout: float | None = 300,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
    )


def run_compose_teardown_command(
    command: Sequence[str],
    *,
    cwd: Path,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout: float | None = 300,
) -> None:
    runner = run_command or _default_run_command
    completed = runner(
        list(command),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError("compose_down_failed")


def teardown_known_stack(
    stack_id: str,
    *,
    docker_dir: Path | None = None,
    cwd: Path | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> None:
    stack = resolve_known_stack(stack_id, docker_dir=docker_dir)
    command = non_destructive_compose_down_args(stack)
    run_compose_teardown_command(
        command,
        cwd=cwd or _REPO_ROOT,
        run_command=run_command,
    )


def run_terminal_compose_teardown(
    *,
    compose_ownership_entered: bool,
    teardown_fn: Callable[[], None],
    kv_prefix: str = "proof",
) -> TerminalTeardownOutcome:
    if not compose_ownership_entered:
        return TerminalTeardownOutcome(attempted=False, result="SKIPPED")
    print(f"{kv_prefix}_teardown_attempted=true")
    try:
        teardown_fn()
    except Exception as exc:  # noqa: BLE001 - terminal cleanup boundary
        print(f"{kv_prefix}_teardown_result=FAIL")
        print(f"{kv_prefix}_teardown_error_type={type(exc).__name__}")
        return TerminalTeardownOutcome(
            attempted=True,
            result="FAIL",
            error_type=type(exc).__name__,
        )
    print(f"{kv_prefix}_teardown_result=PASS")
    return TerminalTeardownOutcome(attempted=True, result="PASS")


def finalize_exit_code_with_teardown(
    *,
    functional_pass: bool,
    functional_exit_code: int,
    teardown_outcome: TerminalTeardownOutcome,
) -> int:
    if functional_pass:
        if teardown_outcome.result == "FAIL":
            return 1
        return 0
    if functional_exit_code != 0:
        return functional_exit_code
    return 1


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teardown a known standalone proof Compose project.",
    )
    parser.add_argument(
        "command",
        choices=("teardown",),
        help="Lifecycle command to execute.",
    )
    parser.add_argument(
        "--stack-id",
        required=True,
        choices=(
            _BACKGROUND_TASK_STACK_ID,
            _OS_INTERACTION_STACK_ID,
            _HOSTING_STACK_ID,
            _FILE_WATCHER_STACK_ID,
            "lkw-trusted-ask-workspace-proof",
            "lkw-core-platform-proof",
            "lkw-product-quickstart",
        ),
        help="Known proof stack identifier.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    if args.command != "teardown":
        return 1
    try:
        teardown_known_stack(str(args.stack_id))
    except (KeyError, RuntimeError, subprocess.TimeoutExpired, OSError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
