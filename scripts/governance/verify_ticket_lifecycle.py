#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Enforce Delivery DevOS ticket lifecycle guards (OD-05 technical baseline)."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_LIFECYCLE = _REPO / "governance" / "agent-control-plane" / "task_lifecycle.yaml"
_STATE_RE = re.compile(r"^##\s+Lifecycle state\s*\n+\s*(\w+)\s*$", re.MULTILINE | re.IGNORECASE)
_DIGEST_RE = re.compile(r"^Plan digest:\s*([0-9a-f]{64})\s*$", re.MULTILINE | re.IGNORECASE)
_APPROVED_RE = re.compile(r"^Approved:\s*.+$", re.MULTILINE)

_POST_PLAN_STATES = frozenset({"plan_approved", "in_progress", "quality_gate", "ready_for_audit", "closed"})


def _load_fsm() -> tuple[frozenset[str], dict[str, list[str]]]:
    doc = yaml.safe_load(_LIFECYCLE.read_text(encoding="utf-8"))
    states = doc.get("states") if isinstance(doc, dict) else None
    transitions = doc.get("transitions") if isinstance(doc, dict) else None
    if not isinstance(states, list) or not isinstance(transitions, dict):
        raise ValueError("task_lifecycle.yaml: invalid states/transitions")
    state_set = frozenset(str(s) for s in states)
    norm_trans: dict[str, list[str]] = {}
    for src, targets in transitions.items():
        if isinstance(targets, list):
            norm_trans[str(src)] = [str(t) for t in targets]
    return state_set, norm_trans


def parse_ticket_state(body: str) -> str | None:
    match = _STATE_RE.search(body)
    return match.group(1).strip() if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify ticket lifecycle state and plan guards.")
    parser.add_argument("ticket", type=Path, help="Ticket markdown under governance/tickets/")
    parser.add_argument(
        "--plan",
        type=Path,
        help="Plan file (required when ticket records Plan digest and state >= plan_approved)",
    )
    args = parser.parse_args(argv)

    ticket_path = args.ticket if args.ticket.is_absolute() else _REPO / args.ticket
    if not ticket_path.is_file():
        print(f"[ERROR] ticket not found: {ticket_path}", file=sys.stderr)
        return 1

    try:
        valid_states, _ = _load_fsm()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    body = ticket_path.read_text(encoding="utf-8")
    state = parse_ticket_state(body)
    if not state:
        print("[ERROR] ticket missing ## Lifecycle state section", file=sys.stderr)
        return 1
    if state not in valid_states:
        print(f"[ERROR] invalid lifecycle state: {state}", file=sys.stderr)
        return 1

    has_digest = _DIGEST_RE.search(body) is not None
    has_approved = _APPROVED_RE.search(body) is not None

    if state in _POST_PLAN_STATES and not has_approved:
        print("[ERROR] state requires Approved: line (plan_pending_to_plan_approved guard)", file=sys.stderr)
        return 1

    if has_digest:
        plan_path = args.plan
        if plan_path is None:
            if state in _POST_PLAN_STATES:
                print("[ERROR] ticket has Plan digest — pass --plan for stale digest check", file=sys.stderr)
                return 1
        else:
            plan_resolved = plan_path if plan_path.is_absolute() else _REPO / plan_path
            completed = subprocess.run(
                [
                    sys.executable,
                    str(_REPO / "scripts/governance/verify_plan_approval.py"),
                    str(ticket_path),
                    "--plan",
                    str(plan_resolved),
                ],
                cwd=_REPO,
                check=False,
            )
            if completed.returncode != 0:
                return completed.returncode

    print("[INFO] verify_ticket_lifecycle: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
