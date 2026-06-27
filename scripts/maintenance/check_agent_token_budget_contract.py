#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — ACP token budget contract (ACP-TOK-CI · CI-18)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

KERNEL_METERING_PATH = REPO_ROOT / "intergrax/runtime/kernel/step_kernel.py"
ACP_RUN_PATH = REPO_ROOT / "intergrax/agents/authoring/acp_run.py"
STEP_LOOP_PATH = REPO_ROOT / "intergrax/agents/authoring/step_loop.py"
AGENTS_ROOT = REPO_ROOT / "agents"

FORBIDDEN_AGENT_IMPORTS: tuple[str, ...] = (
    "apply_llm_metering_after_step",
    "increment_budget_from_llm_calls",
    "increment_token_usage",
)

METERING_INCREMENT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/contracts/acp_token_metering.py",
        "intergrax/agents/acp_token_metering_bridge.py",
    }
)

STATE_DELTA_BUDGET_TOKENS: tuple[str, ...] = (
    '"budget"',
    "'budget'",
    "tokens_in",
    "tokens_out",
    "tokens_total",
    "tokens_limit",
    "tokens_remaining",
)

BUDGET_SMOKE_TESTS: tuple[str, ...] = (
    "tests/unit/agents/test_acp_token_usage_metering.py",
    "tests/unit/agents/test_acp_token_budget_enforcement.py",
    "tests/unit/agents/test_acp_token_budget_reactions.py",
)

_REQUESTS_DEP_WARNING = "RequestsDependencyWarning"


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _check_kernel_metering_wiring() -> list[str]:
    violations: list[str] = []
    if not KERNEL_METERING_PATH.is_file():
        return [f"missing kernel: {_rel(KERNEL_METERING_PATH)}"]

    text = KERNEL_METERING_PATH.read_text(encoding="utf-8")
    required_tokens = (
        "apply_llm_metering_after_step",
        "check_hard_budget_before_llm",
        "maybe_emit_budget_threshold",
    )
    for token in required_tokens:
        if token not in text:
            violations.append(
                f"{_rel(KERNEL_METERING_PATH)}: missing harness budget hook `{token}`"
            )
    return violations


def _check_acp_run_wiring() -> list[str]:
    violations: list[str] = []
    for path, token in (
        (ACP_RUN_PATH, "wrap_budget_enforcing_router"),
        (STEP_LOOP_PATH, "handle_hard_budget_violation"),
    ):
        if not path.is_file():
            violations.append(f"missing authoring module: {_rel(path)}")
            continue
        if token not in path.read_text(encoding="utf-8"):
            violations.append(f"{_rel(path)}: missing `{token}` wiring")
    return violations


def _check_metering_increment_scope() -> list[str]:
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "intergrax").rglob("*.py")):
        rel = _rel(path)
        if rel in METERING_INCREMENT_ALLOWLIST or rel == _rel(KERNEL_METERING_PATH):
            continue
        text = path.read_text(encoding="utf-8")
        for token in ("increment_budget_from_llm_calls", "increment_token_usage"):
            if token in text:
                violations.append(
                    f"{rel}: `{token}` only allowed in kernel bridge and metering contracts"
                )
    return violations


def _scan_agents_budget_state_delta() -> list[str]:
    violations: list[str] = []
    if not AGENTS_ROOT.is_dir():
        return violations

    for path in sorted(AGENTS_ROOT.rglob("*.py")):
        rel = _rel(path)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "state_delta" not in line:
                continue
            for token in STATE_DELTA_BUDGET_TOKENS:
                if token in line:
                    violations.append(
                        f"{rel}:{line_no}: agents must not mutate budget via state_delta "
                        f"(found `{token}`)"
                    )
    return violations


def _scan_agents_forbidden_imports() -> list[str]:
    violations: list[str] = []
    if not AGENTS_ROOT.is_dir():
        return violations

    import_pattern = re.compile(r"^\s*(?:from|import)\s+")
    for path in sorted(AGENTS_ROOT.rglob("*.py")):
        rel = _rel(path)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not import_pattern.match(line):
                continue
            for token in FORBIDDEN_AGENT_IMPORTS:
                if token in line:
                    violations.append(
                        f"{rel}:{line_no}: agents must not import harness metering `{token}`"
                    )
    return violations


def _pytest_failure_detail(stdout: str, stderr: str) -> str:
    """Prefer pytest output over noisy third-party stderr (e.g. requests pin warnings)."""
    parts: list[str] = []
    stdout = stdout.strip()
    stderr = stderr.strip()
    if stdout:
        parts.append(stdout)
    if stderr and _REQUESTS_DEP_WARNING not in stderr:
        parts.append(stderr)
    elif stderr and not stdout:
        parts.append(stderr)
    return "\n".join(parts) or "pytest failed"


def _run_budget_smoke_tests() -> list[str]:
    violations: list[str] = []
    pytest_args = (
        "-q",
        "--tb=short",
        "-W",
        "ignore::requests.exceptions.RequestsDependencyWarning",
    )
    for rel_test in BUDGET_SMOKE_TESTS:
        test_path = REPO_ROOT / rel_test
        if not test_path.is_file():
            violations.append(f"missing smoke test module: {rel_test}")
            continue
        for cmd in (
            ["uv", "run", "pytest", str(test_path), *pytest_args],
            [PYTHON, "-m", "pytest", str(test_path), *pytest_args],
        ):
            completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
            if completed.returncode == 0:
                break
        else:
            detail = _pytest_failure_detail(completed.stdout, completed.stderr)
            violations.append(f"smoke test failed for {rel_test}: {detail}")
    return violations


def main() -> int:
    violations: list[str] = []
    violations.extend(_check_kernel_metering_wiring())
    violations.extend(_check_acp_run_wiring())
    violations.extend(_check_metering_increment_scope())
    violations.extend(_scan_agents_budget_state_delta())
    violations.extend(_scan_agents_forbidden_imports())
    violations.extend(_run_budget_smoke_tests())

    if violations:
        print("ACP token budget contract violations:")
        print("\n".join(sorted(set(violations))))
        return 1

    print("ACP token budget contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
