# © Artur Czarnecki. All rights reserved.

"""Manifest digest and report artifact helpers."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import NamedTuple

from intergrax.platform.enterprise_audit.contracts import ManifestCheck


def canonical_manifest_bytes() -> bytes:
    pkg = resources.files("intergrax.platform.enterprise_audit")
    text = (pkg / "data" / "enterprise_audit_check_manifest.v1.json").read_text(encoding="utf-8")
    data = json.loads(text)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_digest() -> str:
    return hashlib.sha256(canonical_manifest_bytes()).hexdigest()


# Explicit check-id prefix -> FRZ family attribution (first match wins). Family-level wildcards (``FRZ-TEN-*`` ...)
# are the existing convention. ``ENT-A-DUP-`` has no FRZ family in the checklist: it is attributed to the aggregator
# ``FRZ-AUD-03`` explicitly (no new coverage claim). A check that matches no row is *not* silently attributed in
# certifying runs (``frz_default_fallback_check_ids``, ``strict``).
_FRZ_ATTRIBUTION_ROWS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("TEN-",), ("FRZ-TEN-*",)),
    (("VBN-", "ENT-A-VBN-"), ("FRZ-RPL-*", "FRZ-CFG-*")),
    (("ENT-A-BND-",), ("FRZ-BND-*",)),
    (("ENT-A-OWN-",), ("FRZ-OWN-*",)),
    (("ENT-A-CTR-",), ("FRZ-CTR-*",)),
    (("ENT-A-TYP-",), ("FRZ-TYP-*",)),
    (("ENT-A-EVOL-", "ENT-A-INV-"), ("FRZ-EVOL-*",)),
    (("ENT-A-TEN-",), ("FRZ-TEN-*",)),
    (("ENT-A-REG-",), ("FRZ-REG-*",)),
    (("ENT-A-DOC-",), ("FRZ-DOC-*",)),
    (("ENT-A-PRD-",), ("FRZ-PRD-*",)),
    (("ENT-A-CMP-",), ("FRZ-CMP-*",)),
    (("ENT-A-EXE-",), ("FRZ-EXE-*",)),
    (("ENT-A-GOV-",), ("FRZ-GOV-*",)),
    (("ENT-A-OBS-",), ("FRZ-OBS-*",)),
    (("ENT-A-PLG-",), ("FRZ-PLG-*",)),
    (("ENT-A-REC-",), ("FRZ-REC-*",)),
    (("ENT-A-REL-",), ("FRZ-REL-*",)),
    (("ENT-A-SEC-",), ("FRZ-SEC-*",)),
    (("ENT-A-STA-",), ("FRZ-STA-*",)),
    (("ENT-A-DUP-",), ("FRZ-AUD-03",)),
)

_FRZ_DEFAULT_ATTRIBUTION: tuple[str, ...] = ("FRZ-AUD-03",)


def _explicit_frz_for_check(check: ManifestCheck) -> tuple[str, ...] | None:
    for prefixes, frz in _FRZ_ATTRIBUTION_ROWS:
        if check.check_id.startswith(prefixes):
            return frz
    return None


def _frz_for_check(check: ManifestCheck) -> tuple[str, ...]:
    explicit = _explicit_frz_for_check(check)
    return explicit if explicit is not None else _FRZ_DEFAULT_ATTRIBUTION


def frz_default_fallback_check_ids(checks: tuple[ManifestCheck, ...]) -> tuple[str, ...]:
    """Check ids that match no explicit attribution row (they would silently fall through to ``FRZ-AUD-03``)."""
    return tuple(c.check_id for c in checks if _explicit_frz_for_check(c) is None)


def _family_criteria(family: str, count: int, *, skip: tuple[int, ...] = ()) -> tuple[str, ...]:
    return tuple(f"FRZ-{family}-{n:02d}" for n in range(1, count + 1) if n not in skip)


# FRZ families with no mapped ENT check (EQOA-A6-004, DP-66): their criteria are evidenced by stage-gate records of the
# owner stage in the freeze checklist, not by the enterprise audit. The block is a declaration, not a status claim: no
# FRZ row changes status because it exists. ``criteria_not_exercised`` is checked against the checklist rows by
# ``test_wd_c04`` / ``test_wd_c05``, so a checklist change forces a decision. ``FRZ-AUD-03`` is the aggregator that
# every unattributed check falls into and is not listed.
class StageGateOnlyFamily(NamedTuple):
    family: str
    owner_stage: str
    basis: str
    criteria_not_exercised: tuple[str, ...]


STAGE_GATE_ONLY_FRZ_FAMILIES: tuple[StageGateOnlyFamily, ...] = (
    StageGateOnlyFamily(
        family="AUD",
        owner_stage="ENT-AUDIT-X",
        basis=("Audit-runner criteria are evidenced by the ENT-AUDIT-X stage records; no mapped check asserts them."),
        criteria_not_exercised=_family_criteria("AUD", 10, skip=(3,)),
    ),
    StageGateOnlyFamily(
        family="CTL",
        owner_stage="CTRL-X",
        basis=("Control-plane criteria are evidenced by the CTRL-X stage records; no mapped check asserts them."),
        criteria_not_exercised=_family_criteria("CTL", 12),
    ),
    StageGateOnlyFamily(
        family="DEBT",
        owner_stage="ARCH-FREEZE",
        basis=("Architecture-debt criteria are owned by the ARCH-FREEZE stage; no mapped check asserts them."),
        criteria_not_exercised=_family_criteria("DEBT", 6),
    ),
    StageGateOnlyFamily(
        family="FRZ",
        owner_stage="ARCH-FREEZE",
        basis=("Freeze-mechanics criteria are owned by the ARCH-FREEZE stage; no mapped check asserts them."),
        criteria_not_exercised=_family_criteria("FRZ", 13),
    ),
    StageGateOnlyFamily(
        family="HRN",
        owner_stage="HARNESS-FINAL",
        basis=("Harness criteria are evidenced by the HARNESS-FINAL stage records; no mapped check asserts them."),
        criteria_not_exercised=_family_criteria("HRN", 8),
    ),
    StageGateOnlyFamily(
        family="TRC",
        owner_stage="TRACE-X, GOV-X*",
        basis=(
            "Traceability criteria are evidenced by the TRACE-X and GOV-X stage records; no mapped check asserts them."
        ),
        criteria_not_exercised=_family_criteria("TRC", 12),
    ),
)


def _stage_gate_only_block() -> list[dict[str, object]]:
    return [
        {
            "family": entry.family,
            "owner_stage": entry.owner_stage,
            "basis": entry.basis,
            "criteria_not_exercised": list(entry.criteria_not_exercised),
        }
        for entry in STAGE_GATE_ONLY_FRZ_FAMILIES
    ]


def frz_coverage_artifact(checks: tuple[ManifestCheck, ...], *, strict: bool = False) -> dict:
    if strict:
        fallthrough = frz_default_fallback_check_ids(checks)
        if fallthrough:
            raise ValueError(
                "FRZ coverage: checks without an explicit FRZ family attribution: " + ", ".join(sorted(fallthrough))
            )
    rows = []
    for check in checks:
        frz = _frz_for_check(check)
        rows.append(
            {
                "check_id": check.check_id,
                "pg_id": check.pg_id,
                "frz_aud": frz[0],
                "frz_mapping": list(frz),
                "executable": check.entrypoint_kind != "internal_pending",
            }
        )
    return {
        "schema_version": 1,
        "rows": rows,
        "row_count": len(rows),
        "stage_gate_only": _stage_gate_only_block(),
    }
