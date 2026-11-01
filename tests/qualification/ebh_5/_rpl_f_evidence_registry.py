# © Artur Czarnecki. All rights reserved.

"""RPL-F failure/recovery evidence per REPLACEABLE — QUALIFIED concern (Protocol 03)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_5._vendor_backend_types import RplGateStatus

_BEHAVIORAL_MODULE: Final[str] = "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py"


@dataclass(frozen=True, slots=True)
class RplFDisposition:
    concern_id: str
    gate: RplGateStatus
    proof_pytest_nodes: tuple[str, ...]
    normative_na_rationale: str = ""


_RPL_F_ACTIVE_PROOF_NODES: Final[dict[str, tuple[str, ...]]] = {
    "observability.otlp_transport": (
        f"{_BEHAVIORAL_MODULE}::test_ebh5_rpl_f_otlp_active_transport_export_failure_propagates",
    ),
    "tokenizer.selection": (f"{_BEHAVIORAL_MODULE}::test_ebh5_rpl_f_tokenizer_active_provider_tokenization_failure",),
    "persistence.relational_sqlite_lab": (
        f"{_BEHAVIORAL_MODULE}::test_ebh5_rpl_f_relational_active_store_operational_failure_after_connect",
    ),
    "int_config.configure_existing": (
        f"{_BEHAVIORAL_MODULE}::test_ebh5_rpl_f_int_config_active_strategy_realization_failure",
    ),
    "integration.key_value_cache": (f"{_BEHAVIORAL_MODULE}::test_ebh5_rpl_f_kv_active_plugin_operation_failure",),
}


_PACKAGE_C_RPL_F_NA: Final[str] = (
    "N/A — WITH EVIDENCE: Package C selection/reachability proof; "
    "active failure/recovery covered by category owner regression @ EBH-7"
)


def normative_rpl_f_na_rationale(concern_id: str) -> str:
    """Per-concern RPL-F N/A rationale (dual-slug integrations must not share one generic string)."""
    from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import catalog_by_concern_id

    entry = catalog_by_concern_id().get(concern_id)
    if entry is not None:
        cat = entry.category.value
        if cat == "relational_store":
            return (
                "N/A — WITH EVIDENCE: relational_store active operational failure @ "
                "persistence.relational_sqlite_lab RPL-F PASS; dual-slug proves selection/isolation only"
            )
        if entry.operational_lifecycle:
            return (
                f"N/A — WITH EVIDENCE: {cat} — connect/close parity @ Package C v2 RPL-D; "
                "no active fault-injection port on this integration resolve surface"
            )
        return (
            f"N/A — WITH EVIDENCE: {cat} — profile selection + unregistered fail-closed @ Package C v2; "
            "active failure/recovery not claimed on catalog resolve-only path"
        )
    if concern_id == "integration.external_work":
        return (
            "N/A — WITH EVIDENCE: external_work replaceability @ dual-provider qualification proof; "
            "operational failure semantics owned by contractor contract tests"
        )
    if concern_id == "integration.marketplace":
        return (
            "N/A — WITH EVIDENCE: marketplace catalog resolution @ factory one-pass; "
            "no active backend fault surface on listing resolve path"
        )
    if concern_id.startswith("integration."):
        suffix = concern_id.removeprefix("integration.")
        return (
            f"N/A — WITH EVIDENCE: {suffix} — sanctioned composition reachability; "
            "RPL-F active failure not applicable without second live backend @ EBH-7 scope"
        )
    return _PACKAGE_C_RPL_F_NA


def _build_rpl_f_by_concern() -> dict[str, RplFDisposition]:
    from tests.qualification.ebh_5._behavioral_proof_registry import QUALIFIED_CONCERN_IDS

    by: dict[str, RplFDisposition] = {
        concern_id: RplFDisposition(
            concern_id=concern_id,
            gate=RplGateStatus.PASS,
            proof_pytest_nodes=_RPL_F_ACTIVE_PROOF_NODES[concern_id],
        )
        for concern_id in _RPL_F_ACTIVE_PROOF_NODES
    }
    for concern_id in QUALIFIED_CONCERN_IDS:
        if concern_id in by:
            continue
        by[concern_id] = RplFDisposition(
            concern_id=concern_id,
            gate=RplGateStatus.NA,
            proof_pytest_nodes=(),
            normative_na_rationale=normative_rpl_f_na_rationale(concern_id),
        )
    return by


_RPL_F_BY_CONCERN: Final[dict[str, RplFDisposition]] = _build_rpl_f_by_concern()


def rpl_f_disposition(concern_id: str) -> RplFDisposition | None:
    return _RPL_F_BY_CONCERN.get(concern_id)


def rpl_f_gate_for_qualified(concern_id: str) -> RplGateStatus:
    disp = _RPL_F_BY_CONCERN.get(concern_id)
    if disp is None:
        return RplGateStatus.FAIL
    return disp.gate


def rpl_f_validation_errors() -> tuple[str, ...]:
    from tests.qualification.ebh_5._behavioral_proof_registry import QUALIFIED_CONCERN_IDS
    from tests.qualification.ebh_5._proof_validation import pytest_test_function_exists

    errors: list[str] = []
    for concern_id in sorted(QUALIFIED_CONCERN_IDS):
        disp = _RPL_F_BY_CONCERN.get(concern_id)
        if disp is None:
            errors.append(f"{concern_id}: missing RPL-F disposition")
            continue
        if disp.gate == RplGateStatus.PASS:
            if not disp.proof_pytest_nodes:
                errors.append(f"{concern_id}: RPL-F PASS requires proof_pytest_nodes")
            for node in disp.proof_pytest_nodes:
                if not pytest_test_function_exists(node):
                    errors.append(f"{concern_id}: RPL-F proof missing {node}")
        elif disp.gate == RplGateStatus.NA:
            if not disp.normative_na_rationale.strip().startswith("N/A — WITH EVIDENCE"):
                errors.append(f"{concern_id}: RPL-F NA requires normative N/A — WITH EVIDENCE rationale")
            if disp.proof_pytest_nodes:
                errors.append(f"{concern_id}: RPL-F NA must not cite proof nodes")
            from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import catalog_by_concern_id

            if concern_id in catalog_by_concern_id() and disp.normative_na_rationale == _PACKAGE_C_RPL_F_NA:
                errors.append(f"{concern_id}: dual-slug integration must use per-category RPL-F N/A rationale")
        else:
            errors.append(f"{concern_id}: RPL-F gate must be PASS or NA, got {disp.gate}")
    return tuple(errors)
