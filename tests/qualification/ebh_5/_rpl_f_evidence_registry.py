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

_RPL_F_BY_CONCERN: Final[dict[str, RplFDisposition]] = {
    concern_id: RplFDisposition(
        concern_id=concern_id,
        gate=RplGateStatus.PASS,
        proof_pytest_nodes=_RPL_F_ACTIVE_PROOF_NODES[concern_id],
    )
    for concern_id in _RPL_F_ACTIVE_PROOF_NODES
}


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
        else:
            errors.append(f"{concern_id}: RPL-F gate must be PASS or NA, got {disp.gate}")
    return tuple(errors)
