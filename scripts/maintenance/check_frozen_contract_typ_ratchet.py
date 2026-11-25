#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail on Any/object/reflection ratchet regressions vs governance/qualification/typ_ratchet_baseline.v1.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = REPO_ROOT / "governance" / "qualification" / "typ_ratchet_baseline.v1.json"


def _load_baseline(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError("baseline missing metrics")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()

    if not args.baseline.is_file():
        print(f"MISSING baseline: {args.baseline}", file=sys.stderr)
        return 2

    from scripts.maintenance.measure_frozen_contract_typ import measure_contract_surface

    baseline = _load_baseline(args.baseline)
    current = measure_contract_surface()
    b_metrics = baseline["metrics"]
    c_metrics = current["metrics"]
    regressions: list[str] = []
    for key in ("any", "object", "getattr", "setattr", "hasattr"):
        if c_metrics.get(key, 0) > b_metrics.get(key, 0):
            regressions.append(f"{key}: baseline={b_metrics.get(key, 0)} current={c_metrics.get(key, 0)}")
    typ05_baseline = baseline.get("typ05", {}).get("totals", {})
    if typ05_baseline:
        from scripts.maintenance.measure_frozen_contract_typ05 import measure_typ05

        typ05_current = measure_typ05()["totals"]
        for key, ceiling in typ05_baseline.items():
            if typ05_current.get(key, 0) > ceiling:
                regressions.append(f"typ05.{key}: baseline={ceiling} current={typ05_current.get(key, 0)}")
    if regressions:
        print("TYP_RATCHET_FAIL")
        for line in regressions:
            print(line, file=sys.stderr)
        return 1
    print("TYP_RATCHET_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
