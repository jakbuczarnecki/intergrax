# © Artur Czarnecki. All rights reserved.

"""W4-03 deployment safety assessor DM-01..07 (read-only; OD-EVOL-05b Option 1: no production port)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from testing_support.change_safety.contracts.enums import (
    CompatibilityAtChangeResult,
    DeployRefusalReason,
    DeploySafetyVerdict,
)
from testing_support.change_safety.contracts.verdicts import DeployDimension, DeploySafetyAssessment
from testing_support.change_safety.data.loaders import load_evol_x_data_document

SNAPSHOT_SCHEMA = "schema_snapshot_manifest.v1"


@dataclass(frozen=True)
class DeployFacts:
    operation: str  # "rollback" | "roll_forward"; anything else fails closed
    evidence_present: bool = True
    contract_compat: CompatibilityAtChangeResult = CompatibilityAtChangeResult.COMPATIBLE
    persisted_schema_compatible: bool = True
    durable_forward_only_applied: bool = False
    inflight_pins: bool = False
    drain_possible: bool = True


INPUT_NAMES: tuple[str, ...] = tuple(f.name for f in fields(DeployFacts))


@dataclass(frozen=True)
class StoreSchema:
    version: int
    max_readable_version: int
    forward_only: bool = False  # this version was reached via a non-reversible migration


@dataclass(frozen=True)
class SchemaSnapshot:
    revision_id: str
    stores: tuple[tuple[str, StoreSchema], ...]


class SidecarSnapshotPort:
    """``schema_snapshot@revision`` facts from a sidecar manifest (OD-EVOL-06); absent revision -> None."""

    def __init__(self, snapshots: dict[str, SchemaSnapshot]) -> None:
        self._snapshots = snapshots

    @classmethod
    def from_manifest(cls, payload: dict[str, object]) -> SidecarSnapshotPort:
        if payload.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported snapshot manifest schema_version")
        out: dict[str, SchemaSnapshot] = {}
        for revision_id, body in dict(payload["revisions"]).items():  # type: ignore[call-overload]
            stores = tuple(
                sorted(
                    (
                        name,
                        StoreSchema(
                            version=int(s["version"]),
                            max_readable_version=int(s.get("max_readable_version", s["version"])),
                            forward_only=bool(s.get("forward_only", False)),
                        ),
                    )
                    for name, s in dict(body["stores"]).items()
                )
            )
            out[revision_id] = SchemaSnapshot(revision_id, stores)
        return cls(out)

    @classmethod
    def from_file(cls, path: Path) -> SidecarSnapshotPort:
        return cls.from_manifest(json.loads(path.read_text(encoding="utf-8")))

    def schema_snapshot(self, revision_id: str) -> SchemaSnapshot | None:
        return self._snapshots.get(revision_id)


def facts_from_snapshots(
    *,
    operation: str,
    current: SchemaSnapshot | None,
    target: SchemaSnapshot | None,
    contract_compat: CompatibilityAtChangeResult,
    inflight_pins: bool = False,
    drain_possible: bool = True,
) -> DeployFacts:
    """Derive durable-state facts from the two revision snapshots. Missing snapshot => evidence_present False."""
    if current is None or target is None:
        return DeployFacts(
            operation=operation,
            evidence_present=False,
            contract_compat=contract_compat,
            inflight_pins=inflight_pins,
            drain_possible=drain_possible,
        )
    cur = dict(current.stores)
    tgt = dict(target.stores)
    compatible = True
    forward_only = False
    for name, cur_schema in cur.items():
        tgt_schema = tgt.get(name)
        if tgt_schema is None:
            compatible = False
            continue
        if tgt_schema.max_readable_version < cur_schema.version:
            compatible = False
        if cur_schema.forward_only and cur_schema.version > tgt_schema.version:
            forward_only = True
    return DeployFacts(
        operation=operation,
        evidence_present=True,
        contract_compat=contract_compat,
        persisted_schema_compatible=compatible,
        durable_forward_only_applied=forward_only,
        inflight_pins=inflight_pins,
        drain_possible=drain_possible,
    )


def load_deploy_matrix() -> tuple[dict[str, object], ...]:
    doc = load_evol_x_data_document("deployment_safety_matrix.v1.json")
    rows = tuple(doc["rows"])
    if tuple(doc["inputs"]) != INPUT_NAMES:
        raise ValueError("deployment matrix inputs drifted from DeployFacts fields")
    for row in rows:
        DeploySafetyVerdict(row["verdict"])
        if row["refusal"] is not None:
            DeployRefusalReason(row["refusal"])
        if set(row["when"]) - set(INPUT_NAMES):
            raise ValueError(f"{row['dm_id']}: unknown condition fields")
    if not rows or rows[-1]["when"] != {}:
        raise ValueError("deployment matrix must end with a fail-closed catch-all row")
    covered = {str(r["dm_id"]) for r in rows}
    if covered != {f"DM-0{i}" for i in range(1, 8)}:
        raise ValueError(f"deployment matrix must cover DM-01..07, got {sorted(covered)}")
    return rows


def _value(facts: DeployFacts, key: str) -> object:
    raw = getattr(facts, key)
    return raw.value if isinstance(raw, CompatibilityAtChangeResult) else raw


def assess_deploy_safety(facts: DeployFacts) -> DeploySafetyAssessment:
    dims = tuple(DeployDimension(name=k, value=str(_value(facts, k))) for k in INPUT_NAMES)
    for row in load_deploy_matrix():
        when: dict[str, object] = row["when"]  # type: ignore[assignment]
        if all(_value(facts, key) == expected for key, expected in when.items()):
            refusal = row["refusal"]
            return DeploySafetyAssessment(
                matrix_id=str(row["dm_id"]),
                verdict=DeploySafetyVerdict(row["verdict"]),
                reason=DeployRefusalReason(refusal) if refusal else None,
                dimensions=dims,
            )
    raise AssertionError("unreachable: matrix ends with catch-all")


def assess_rollback_eligibility(
    *, durable_state_forward_only: bool
) -> tuple[DeploySafetyVerdict, DeployRefusalReason | None]:
    """Backward-compatible narrow entry point."""
    result = assess_deploy_safety(
        DeployFacts(operation="rollback", durable_forward_only_applied=durable_state_forward_only)
    )
    return result.verdict, result.reason


def load_na_register() -> tuple[dict[str, object], ...]:
    return tuple(load_evol_x_data_document("deployment_na_register.v1.json")["entries"])
