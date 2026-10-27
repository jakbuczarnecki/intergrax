# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P6 soak/chaos helpers."""

from __future__ import annotations

import os
from pathlib import Path

from tests.qualification.dist_scale_x._p2_support import LOCK_ENV_DEFINITION, code_sha
from tests.qualification.dist_scale_x._p5_support import p5_record, write_p5_manifest
from tests.qualification.dist_scale_x.evidence_manifest import (
    DistScaleXEvidenceManifest,
    catalog_integrity_digest,
    write_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

P6_ENVELOPE_RECORD_ID = "DIST-SCALE-X-ENVELOPE-P6-v1"
P6_ENV_DEFINITION = (
    f"{LOCK_ENV_DEFINITION}; P6 envelope {P6_ENVELOPE_RECORD_ID}; WL-7 soak + chaos N"
)

# Envelope target 14400 s — override for full soak qualification.
WL7_DURATION_SECONDS = int(os.environ.get("DIST_SCALE_X_WL7_DURATION_SECONDS", "120"))
WL7_RSS_SAMPLE_SECONDS = int(os.environ.get("DIST_SCALE_X_WL7_RSS_SAMPLE_SECONDS", "300"))
WL7_QUEUE_SAMPLE_SECONDS = int(os.environ.get("DIST_SCALE_X_WL7_QUEUE_SAMPLE_SECONDS", "60"))
WL7_LAT_SAMPLE_SECONDS = int(os.environ.get("DIST_SCALE_X_WL7_LAT_SAMPLE_SECONDS", "900"))
WL7_WORKERS = int(os.environ.get("DIST_SCALE_X_WL7_WORKERS", "2"))


def write_p6_manifest(
    *,
    run_id: str,
    records: tuple,
    topology_description: str,
    chaos_seed: str | None = None,
) -> Path:
    topo = topology_description
    if chaos_seed:
        topo = f"{topo}; chaos_seed={chaos_seed}"
    manifest = DistScaleXEvidenceManifest(
        schema_version=1,
        run_id=run_id,
        code_sha=code_sha(),
        catalog_integrity_sha256=catalog_integrity_digest(),
        environment_definition=P6_ENV_DEFINITION,
        topology_description=topo,
        qualification_phase="P6",
        records=records,
    )
    return write_manifest(_REPO_ROOT, manifest)


def p6_record(**kwargs):  # type: ignore[no-untyped-def]
    return p5_record(**kwargs)


__all__ = [
    "P6_ENVELOPE_RECORD_ID",
    "P6_ENV_DEFINITION",
    "WL7_DURATION_SECONDS",
    "WL7_LAT_SAMPLE_SECONDS",
    "WL7_QUEUE_SAMPLE_SECONDS",
    "WL7_RSS_SAMPLE_SECONDS",
    "WL7_WORKERS",
    "p6_record",
    "write_p6_manifest",
]
