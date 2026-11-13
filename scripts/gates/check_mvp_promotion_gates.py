#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""MVP promotion readiness gates G0-G2 (MVP-EVOL.1) with governance-owned authorization.

PDC-D5-R014 / ADR-PDC-05: evaluation is advisory. This script reports readiness and can
only ever *deny*. It never authorizes a promotion by itself:

- G0 and G2 check that the required modules exist **and define** the required symbols
  (file existence alone no longer satisfies a gate);
- G1 (evaluation baseline) is an advisory input: it must be present, but it is never
  sufficient and it never contributes to authorization;
- authorization is granted only when a governance ticket under ``governance/tickets/``
  carries an operator ``Approved:`` line, passed via ``--governance-ticket``.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_APPROVED_LINE = re.compile(r"^Approved:\s*(?P<value>\S.*)$", re.MULTILINE)
_PLACEHOLDER_VALUES = ("pending", "none", "tbd", "n/a")


def _exists(relative: str, root: Path = REPO_ROOT) -> bool:
    return (root / relative).is_file()


def _defines(relative: str, names: tuple[str, ...], root: Path = REPO_ROOT) -> list[str]:
    """Return the names from ``names`` that ``relative`` does not define at module level."""
    path = root / relative
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return list(names)
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
    return [name for name in names if name not in defined]


def _check_modules(required: dict[str, tuple[str, ...]], label: str, root: Path) -> tuple[bool, str]:
    missing = [path for path in required if not _exists(path, root)]
    if missing:
        return False, f"missing {label}: {', '.join(missing)}"
    absent = [f"{path}:{name}" for path, names in required.items() for name in _defines(path, names, root)]
    if absent:
        return False, f"{label} lack required definitions: {', '.join(absent)}"
    return True, ""


def gate_g0_runnable(root: Path = REPO_ROOT) -> tuple[bool, str]:
    required: dict[str, tuple[str, ...]] = {
        "intergrax/scaffold/cli.py": (),
        "intergrax/cli/doctor.py": (),
        "intergrax/cli/run.py": (),
        "applications/lab_application/host/factory.py": (),
    }
    ok, detail = _check_modules(required, "runnable artifacts", root)
    return (True, "scaffold + lab host present") if ok else (False, detail)


def gate_g1_eval_baseline(root: Path = REPO_ROOT) -> tuple[bool, str]:
    """Advisory input: evaluation modules present and defining their contracts."""
    required: dict[str, tuple[str, ...]] = {
        "intergrax/runtime/architecture/evaluation_automation.py": ("evaluate_automated_results",),
        "intergrax/runtime/architecture/online_evaluation_registry.py": ("FileOnlineEvaluationRegistry",),
        "scripts/maintenance/check_harness_evaluation_wiring.py": (),
    }
    ok, detail = _check_modules(required, "evaluation baseline", root)
    return (True, "evaluation control plane present (advisory input)") if ok else (False, detail)


def gate_g2_policy(root: Path = REPO_ROOT) -> tuple[bool, str]:
    required: dict[str, tuple[str, ...]] = {
        "intergrax/contracts/resilience_policy.py": (),
        "intergrax/contracts/autonomy_level.py": (),
        "intergrax/runtime/policy/autonomy_resolver.py": (),
        "scripts/maintenance/check_harness_resilience_policy.py": (),
    }
    ok, detail = _check_modules(required, "policy modules", root)
    return (True, "resilience + autonomy policy present") if ok else (False, detail)


def governance_authorization(ticket: str | None, root: Path = REPO_ROOT) -> tuple[bool, str]:
    """Authorization comes only from a governance ticket carrying an operator approval."""
    if not ticket:
        return False, "no governance ticket supplied"
    path = (root / ticket).resolve()
    tickets_dir = (root / "governance" / "tickets").resolve()
    if tickets_dir not in path.parents or not path.is_file():
        return False, "governance ticket must be an existing file under governance/tickets/"
    text = path.read_text(encoding="utf-8")
    match = _APPROVED_LINE.search(text)
    if match is None:
        return False, "governance ticket has no Approved: line"
    value = match.group("value").strip()
    if value.lower().startswith(_PLACEHOLDER_VALUES):
        return False, "governance ticket Approved: line is a placeholder"
    if value.startswith("operator-delegate") and "\nDelegation:" not in text:
        return False, "delegated approval has no Delegation: line"
    return True, f"approved by governance ticket {ticket}"


@dataclass
class GateReport:
    lines: list[str] = field(default_factory=list)
    structural_failures: list[str] = field(default_factory=list)
    authorized: bool = False
    authorization_detail: str = ""

    @property
    def ready(self) -> bool:
        return not self.structural_failures


def run_gates(governance_ticket: str | None = None, root: Path = REPO_ROOT) -> GateReport:
    report = GateReport()
    gates = (
        ("G0", gate_g0_runnable, False),
        ("G1", gate_g1_eval_baseline, True),
        ("G2", gate_g2_policy, False),
    )
    for gate_id, gate_fn, advisory in gates:
        ok, detail = gate_fn(root)
        status = "OK" if ok else "FAIL"
        tag = " [advisory input]" if advisory else ""
        report.lines.append(f"{gate_id}: {status}{tag} - {detail}")
        if not ok:
            report.structural_failures.append(gate_id)
    gov_ok, gov_detail = governance_authorization(governance_ticket, root)
    report.authorization_detail = gov_detail
    # G1 is necessary evidence but never sufficient: authorization needs governance.
    report.authorized = report.ready and gov_ok
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-doctor", action="store_true")
    parser.add_argument("--governance-ticket", default=None)
    parser.add_argument("--require-authorization", action="store_true")
    args = parser.parse_args()

    report = run_gates(args.governance_ticket)
    for line in report.lines:
        print(line)

    if not report.ready:
        return 1

    if args.with_doctor:
        doctor = subprocess.run(
            [sys.executable, "-m", "intergrax.cli.doctor"],
            cwd=REPO_ROOT,
            check=False,
        )
        if doctor.returncode != 0:
            print("doctor smoke failed")
            return doctor.returncode

    if report.authorized:
        print(f"promotion authorization: GRANTED by governance - {report.authorization_detail}")
    else:
        print(
            "promotion authorization: NOT GRANTED - readiness checks never authorize a "
            f"promotion ({report.authorization_detail})"
        )
    if args.require_authorization and not report.authorized:
        return 1
    print("mvp promotion readiness G0-G2: OK (advisory; governance owns authorization)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
