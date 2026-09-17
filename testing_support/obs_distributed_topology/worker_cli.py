# © Artur Czarnecki. All rights reserved.

"""Subprocess entrypoint for OBS-DG005 worker roles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_qualification_import_root() -> None:
    root = os.environ.get("INTERGRAX_DG005_QUALIFICATION_ROOT", "").strip()
    if not root:
        return
    root_path = Path(root).resolve()
    root_str = str(root_path)
    filtered: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            filtered.append(entry)
            continue
        if resolved == root_path:
            continue
        if (resolved / "intergrax").is_dir():
            continue
        filtered.append(entry)
    sys.path[:] = [root_str, *filtered]


def main(argv: list[str] | None = None) -> int:
    _bootstrap_qualification_import_root()
    from testing_support.obs_distributed_topology.scenario_io import read_scenario
    from testing_support.obs_distributed_topology.worker_ops import (
        run_diagnostics_role,
        run_idempotent_retry_role,
        run_reader_role,
        run_writer_role,
        worker_result_to_json_dict,
    )

    parser = argparse.ArgumentParser(description="OBS-DG005 topology worker")
    parser.add_argument(
        "role",
        choices=("writer", "reader", "diagnostics", "idempotent_retry"),
    )
    parser.add_argument("scenario_path", type=Path)
    parser.add_argument("result_path", type=Path)
    args = parser.parse_args(argv)

    scenario = read_scenario(args.scenario_path)
    if args.role == "writer":
        result = run_writer_role(scenario)
    elif args.role == "reader":
        result = run_reader_role(scenario)
    elif args.role == "diagnostics":
        result = run_diagnostics_role(scenario)
    else:
        result = run_idempotent_retry_role(scenario)

    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(
        json.dumps(worker_result_to_json_dict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
