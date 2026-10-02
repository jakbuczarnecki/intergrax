# © Artur Czarnecki. All rights reserved.

"""GOV-X2 exact proof-node replay — derived from ``gov_x2_all_proof_pytest_node_ids()`` only."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from tests.qualification.governance.gov_x2.catalog import gov_x2_all_proof_pytest_node_ids
from tests.qualification.governance.pytest_node_integrity import (
    parse_pytest_execute_q_summary,
    run_pytest_execute_q,
)

GOV_X2_ORCHESTRATION_BATCH_MODULE: str = (
    "tests/qualification/governance/gov_x2/test_gov_x2_qualification_batch.py"
)
GOV_X2_PROOF_REPLAY_GATES_MODULE: str = (
    "tests/qualification/governance/gov_x2/test_gov_x2_proof_replay_gates.py"
)


def gov_x2_orchestration_pytest_node_prefixes() -> tuple[str, ...]:
    return (
        f"{GOV_X2_ORCHESTRATION_BATCH_MODULE}::",
        f"{GOV_X2_PROOF_REPLAY_GATES_MODULE}::",
    )


def gov_x2_proof_nodes_for_exact_replay() -> tuple[str, ...]:
    """Proof inventory with mechanical exclusion of orchestration modules (no recursive replay)."""
    nodes = gov_x2_all_proof_pytest_node_ids()
    prefixes = gov_x2_orchestration_pytest_node_prefixes()
    blocked = [node_id for node_id in nodes if any(node_id.startswith(p) for p in prefixes)]
    if blocked:
        raise ValueError(
            "GOV-X2 proof catalog must not include orchestration pytest nodes: "
            + ", ".join(sorted(blocked)),
        )
    return nodes


def build_gov_x2_exact_proof_replay_argv(node_ids: tuple[str, ...]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:xdist",
        "-q",
        "--tb=short",
        *node_ids,
    ]


@dataclass(frozen=True, slots=True)
class GovX2ProofReplayReport:
    requested: int
    executed_passed: int
    failed: int
    skipped: int
    xfailed: int
    errors: int
    exit_code: int
    argv: tuple[str, ...]

    @property
    def missing(self) -> int:
        return max(0, self.requested - self.executed_passed - self.failed - self.skipped - self.errors)


def run_gov_x2_exact_proof_replay(repo_root: Path) -> GovX2ProofReplayReport:
    node_ids = gov_x2_proof_nodes_for_exact_replay()
    argv = build_gov_x2_exact_proof_replay_argv(node_ids)
    proc = run_pytest_execute_q(node_ids, repo_root)
    summary = parse_pytest_execute_q_summary(proc.stdout, proc.stderr)
    return GovX2ProofReplayReport(
        requested=len(node_ids),
        executed_passed=summary.passed,
        failed=summary.failed,
        skipped=summary.skipped,
        xfailed=summary.xfailed,
        errors=summary.errors,
        exit_code=proc.returncode,
        argv=tuple(argv),
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    report = run_gov_x2_exact_proof_replay(repo_root)
    cmd = " ".join(report.argv)
    print(
        f"gov_x2_exact_proof_replay requested={report.requested} "
        f"passed={report.executed_passed} failed={report.failed} "
        f"skipped={report.skipped} xfailed={report.xfailed} errors={report.errors} "
        f"exit_code={report.exit_code}\n"
        f"command={cmd}",
    )
    if report.exit_code != 0:
        return report.exit_code
    if report.failed or report.errors or report.xfailed:
        return 1
    if report.executed_passed != report.requested:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
