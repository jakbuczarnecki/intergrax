#!/usr/bin/env python3
"""Fail-closed mechanical reconciliation validator for EC-W3-11 evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parent


def die(msg: str) -> None:
    print(f"VALIDATION_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    obl = json.loads((RUN_ROOT / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    queue = json.loads((RUN_ROOT / "REMAINING_QUALIFICATION_QUEUE.json").read_text(encoding="utf-8"))
    files = json.loads((RUN_ROOT / "FILE_LEVEL_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    ids = [o["obligation_id"] for o in obl["obligations"]]
    if len(ids) != 34:
        die(f"obligation count {len(ids)} != 34")
    if len(set(ids)) != 34:
        die("duplicate obligation_id")
    qpaths = [q["path"] for q in queue["qualification_queue"]]
    if len(qpaths) != 31:
        die(f"qualification queue count {len(qpaths)} != 31")
    if len(set(qpaths)) != 31:
        die("duplicate qualification path")
    for ref in ("DES_PREREG.json", "BASELINE_BINDING.json", "ENVIRONMENT_MANIFEST.json", "RUN_MANIFEST.json"):
        if not (RUN_ROOT / ref).is_file():
            die(f"missing {ref}")
    if not (RUN_ROOT / "logs").is_dir() or not any((RUN_ROOT / "logs").iterdir()):
        die("missing raw logs")
    if not (RUN_ROOT / "junit").is_dir() or not any((RUN_ROOT / "junit").iterdir()):
        die("missing raw junit")
    for batch in files.get("batches", []):
        if batch.get("exit_code") not in (0, 1, 2, 4):
            die(f"unsupported exit code for batch {batch.get('batch_id')}")
    # structural reconciled: does not require all PASS
    obl["reconciled"] = True
    (RUN_ROOT / "OBLIGATION_EXECUTION_MATRIX.json").write_text(
        json.dumps(obl, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("VALIDATION_OK reconciled=true structural")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
