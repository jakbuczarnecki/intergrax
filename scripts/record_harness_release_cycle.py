#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Record a harness release cycle for operational L3 evidence (W-OPS.5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", required=True, help="Release identifier, e.g. 2026.06.03-rc1")
    parser.add_argument("--notes", default="", help="Optional release board notes")
    parser.add_argument(
        "--gate-red",
        action="store_true",
        help="Mark cycle as not gate-green (default: gate_green=True)",
    )
    parser.add_argument(
        "--verify-gate",
        action="store_true",
        help="Require W-OPS code checks green before recording (runs phase_w_ops evidence checks)",
    )
    args = parser.parse_args()

    if args.verify_gate and not args.gate_red:
        import importlib.util

        evidence_path = REPO_ROOT / "scripts" / "phase_w_ops_evidence.py"
        spec = importlib.util.spec_from_file_location("phase_w_ops_evidence", evidence_path)
        if spec is None or spec.loader is None:
            print(f"Cannot load evidence module: {evidence_path}")
            return 1
        evidence_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evidence_mod)
        evidence = evidence_mod.collect_operational_checks()
        code_ok = all(
            item.passed for item in evidence.checks if item.check_id != "release_cycles"
        )
        if not code_ok:
            failed = [item.check_id for item in evidence.checks if not item.passed]
            print(f"W-OPS code checks failed: {', '.join(failed)}")
            return 1

    from intergrax.runtime.architecture.release_cycle_tracker import (
        append_release_cycle,
        default_tracker_path,
        load_release_cycle_tracker,
    )

    tracker = append_release_cycle(
        cycle_id=args.cycle_id.strip(),
        gate_green=not args.gate_red,
        notes=args.notes.strip(),
    )
    target = default_tracker_path()
    print(f"Recorded release cycle '{args.cycle_id}' ({tracker.completed_count} total)")
    print(f"Tracker: {target.as_posix()}")
    if tracker.completed_count >= 2:
        print("Operational L3 release_cycles check can pass when code checks are green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
