# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P4 physical qualification helpers."""

from __future__ import annotations

import os

from tests.qualification.dist_scale_x._p2_support import pg_port_open, require_p2_infra, resolve_pg_dsn
from tests.qualification.dist_scale_x._p3_support import (
    code_sha,
    new_run_id,
    write_p3_manifest,
)

P4_ENVELOPE_RECORD_ID = "DIST-SCALE-X-ENVELOPE-P4-v1"
P4_ENV_DEFINITION = (
    "PostgreSQL M; 90/10 write/read; ENVELOPE_ONLY; NOT_LOCKED; "
    f"envelope {P4_ENVELOPE_RECORD_ID}"
)

WL5_WARMUP_SECONDS = int(os.environ.get("DIST_SCALE_X_P4_WL5_WARMUP_SECONDS", "60"))
WL5_MEASURE_SECONDS = int(os.environ.get("DIST_SCALE_X_P4_WL5_MEASURE_SECONDS", "120"))
WL5_WRITERS = int(os.environ.get("DIST_SCALE_X_P4_WL5_WRITERS", "2"))
WL5_WRITE_RATE_CAP = float(os.environ.get("DIST_SCALE_X_P4_WL5_WRITE_RATE_CAP", "40"))
WL5_PAYLOAD_BYTES = int(os.environ.get("DIST_SCALE_X_P4_WL5_PAYLOAD_BYTES", "1024"))


def require_p4_pg_infra() -> None:
    require_p2_infra()
    if not pg_port_open():
        import pytest

        pytest.skip("PostgreSQL handshake failed for DIST-SCALE-X P4")


def qual_pg_schema(run_id: str) -> str:
    safe = run_id.replace("-", "_")[:32]
    return f"dist_scale_x_p4_{safe}"


__all__ = [
    "P4_ENVELOPE_RECORD_ID",
    "P4_ENV_DEFINITION",
    "WL5_MEASURE_SECONDS",
    "WL5_PAYLOAD_BYTES",
    "WL5_WARMUP_SECONDS",
    "WL5_WRITE_RATE_CAP",
    "WL5_WRITERS",
    "code_sha",
    "new_run_id",
    "qual_pg_schema",
    "require_p4_pg_infra",
    "resolve_pg_dsn",
    "write_p3_manifest",
]
