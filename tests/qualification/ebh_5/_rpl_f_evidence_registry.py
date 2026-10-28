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


_RPL_F_BY_CONCERN: Final[dict[str, RplFDisposition]] = {
    "observability.otlp_transport": RplFDisposition(
        concern_id="observability.otlp_transport",
        gate=RplGateStatus.PASS,
        proof_pytest_nodes=(
            f"{_BEHAVIORAL_MODULE}::test_ebh5_behavioral_otlp_transport_negative_invalid_distributed_configuration",
            f"{_BEHAVIORAL_MODULE}::test_ebh5_behavioral_otlp_transport_negative_otlp_without_endpoint_fail_closed",
        ),
        normative_na_rationale="",
    ),
    "tokenizer.selection": RplFDisposition(
        concern_id="tokenizer.selection",
        gate=RplGateStatus.PASS,
        proof_pytest_nodes=(
            f"{_BEHAVIORAL_MODULE}::test_ebh5_behavioral_tokenizer_selection_negative_unknown_default",
        ),
    ),
    "persistence.relational_sqlite_lab": RplFDisposition(
        concern_id="persistence.relational_sqlite_lab",
        gate=RplGateStatus.PASS,
        proof_pytest_nodes=(
            f"{_BEHAVIORAL_MODULE}::test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed",
        ),
    ),
    "int_config.configure_existing": RplFDisposition(
        concern_id="int_config.configure_existing",
        gate=RplGateStatus.PASS,
        proof_pytest_nodes=(
            f"{_BEHAVIORAL_MODULE}::test_ebh5_behavioral_int_config_negative_unsupported_strategy_fail_closed",
        ),
    ),
    "integration.key_value_cache": RplFDisposition(
        concern_id="integration.key_value_cache",
        gate=RplGateStatus.PASS,
        proof_pytest_nodes=(f"{_BEHAVIORAL_MODULE}::test_ebh5_behavioral_kv_negative_unregistered_plugin_fail_closed",),
    ),
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
