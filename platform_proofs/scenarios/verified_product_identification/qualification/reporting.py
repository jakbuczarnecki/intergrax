"""JSON serialization for qualification evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    VpiEmbeddingQualificationReport,
)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    msg = f"unsupported type for qualification JSON: {type(value)!r}"
    raise TypeError(msg)


def qualification_report_to_json(report: VpiEmbeddingQualificationReport) -> str:
    return json.dumps(asdict(report), indent=2, default=_json_default, sort_keys=True)


def write_qualification_report(path: Path, report: VpiEmbeddingQualificationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(qualification_report_to_json(report), encoding="utf-8")
