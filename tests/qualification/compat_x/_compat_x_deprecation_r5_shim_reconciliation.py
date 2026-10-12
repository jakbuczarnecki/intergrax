# © Artur Czarnecki. All rights reserved.

"""Evidence-backed historical vs current compat.shim set reconciliation (COMPAT-X-R5-P0-R1-R1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.compat_x._compat_x_closed_world import build_closed_world_report

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_HISTORICAL_SHIM_FIXTURE: Final = (
    _REPO_ROOT / "tests/fixtures/compat_x/historical_shim_candidates_0612c2a7.json"
)
HISTORICAL_SHIM_ACCEPTED_EVIDENCE_SHA: Final = "0612c2a7a263b22a6b72ed5476e0bf0a52cdfea0"


@dataclass(frozen=True, slots=True)
class FrozenHistoricalShimBaseline:
    source_sha: str
    derivation_method: str
    candidate_identity_definition: str
    paths: frozenset[str]
    count: int
    fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class ShimSetDelta:
    historical_paths: frozenset[str]
    current_paths: frozenset[str]
    added_paths: tuple[str, ...]
    removed_paths: tuple[str, ...]
    historical_fingerprint: str
    current_fingerprint: str


def shim_candidate_path_fingerprint(paths: frozenset[str] | set[str]) -> str:
    normalized = "\n".join(sorted(paths))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def load_frozen_historical_shim_baseline() -> FrozenHistoricalShimBaseline:
    raw = json.loads(_HISTORICAL_SHIM_FIXTURE.read_text(encoding="utf-8"))
    path_list = tuple(raw["paths"])
    paths = frozenset(path_list)
    count = int(raw["count"])
    fingerprint = str(raw["fingerprint_sha256"])
    if len(paths) != count:
        raise AssertionError("historical shim fixture: count != len(paths)")
    if len(path_list) != len(paths):
        raise AssertionError("historical shim fixture: duplicate paths")
    computed = shim_candidate_path_fingerprint(paths)
    if computed != fingerprint:
        raise AssertionError("historical shim fixture: fingerprint mismatch")
    if raw["source_sha"] != HISTORICAL_SHIM_ACCEPTED_EVIDENCE_SHA:
        raise AssertionError("historical shim fixture: source_sha mismatch")
    return FrozenHistoricalShimBaseline(
        source_sha=str(raw["source_sha"]),
        derivation_method=str(raw["derivation_method"]),
        candidate_identity_definition=str(raw["candidate_identity_definition"]),
        paths=paths,
        count=count,
        fingerprint_sha256=fingerprint,
    )


def current_compat_shim_paths() -> frozenset[str]:
    closed = build_closed_world_report()
    return frozenset(
        c.path.replace("\\", "/")
        for c in closed.raw_candidates
        if c.discovery_kind == "compat.shim"
    )


def compute_shim_set_delta(
    historical_paths: frozenset[str],
    current_paths: frozenset[str],
) -> ShimSetDelta:
    added = tuple(sorted(current_paths - historical_paths))
    removed = tuple(sorted(historical_paths - current_paths))
    return ShimSetDelta(
        historical_paths=historical_paths,
        current_paths=current_paths,
        added_paths=added,
        removed_paths=removed,
        historical_fingerprint=shim_candidate_path_fingerprint(historical_paths),
        current_fingerprint=shim_candidate_path_fingerprint(current_paths),
    )


@lru_cache(maxsize=1)
def production_shim_set_delta() -> ShimSetDelta:
    baseline = load_frozen_historical_shim_baseline()
    current = current_compat_shim_paths()
    return compute_shim_set_delta(baseline.paths, current)


def shim_reconciliation_gate_violations(
    *,
    historical_paths: frozenset[str] | None = None,
    current_paths: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Fail closed on removed historical shims or shim set delta mismatch vs production."""
    baseline = load_frozen_historical_shim_baseline()
    hist = historical_paths if historical_paths is not None else baseline.paths
    cur = current_paths if current_paths is not None else current_compat_shim_paths()
    delta = compute_shim_set_delta(hist, cur)
    violations: list[str] = []
    if delta.removed_paths:
        violations.append(
            "shim reconciliation: removed historical shim paths: "
            + ", ".join(delta.removed_paths[:8])
            + ("..." if len(delta.removed_paths) > 8 else "")
        )
    prod_delta = production_shim_set_delta()
    if historical_paths is not None or current_paths is not None:
        if (
            delta.added_paths != prod_delta.added_paths
            or delta.removed_paths != prod_delta.removed_paths
            or delta.current_fingerprint != prod_delta.current_fingerprint
        ):
            violations.append("shim reconciliation: exact set/fingerprint mismatch")
    elif delta.added_paths != prod_delta.added_paths:
        violations.append(
            "shim reconciliation: unexpected added shim paths: " + ", ".join(delta.added_paths)
        )
    return tuple(violations)


def shim_growth_reconciliation_snapshot() -> ShimSetDelta:
    return production_shim_set_delta()
