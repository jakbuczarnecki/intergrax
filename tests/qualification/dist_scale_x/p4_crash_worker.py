# © Artur Czarnecki. All rights reserved.

"""Crash-point worker — staged F01/F03/F07/F09 boundaries."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tests.qualification.dist_scale_x.p4_persistence_op import mint_qual_task_id
from tests.qualification.dist_scale_x.p4_staged_persistence import (
    BOUNDARY_SYNC,
    BoundaryId,
    execute_staged_boundary,
    handle_to_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--sync-file", required=True)
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()
    sync = Path(args.sync_file)
    state = Path(args.state_file)
    boundary: BoundaryId = args.boundary  # type: ignore[assignment]
    task_id = mint_qual_task_id()
    handle = execute_staged_boundary(
        dsn=args.dsn,
        schema=args.schema,
        tenant_id=args.tenant_id,
        boundary=boundary,
        task_id=task_id,
        subject=f"crash-{boundary.lower()}",
    )
    state.write_text(handle_to_json(handle), encoding="utf-8")
    sync.write_text(BOUNDARY_SYNC[boundary], encoding="utf-8")
    time.sleep(300)


if __name__ == "__main__":
    main()
