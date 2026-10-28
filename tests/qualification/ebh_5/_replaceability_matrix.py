# © Artur Czarnecki. All rights reserved.

"""CROSS_LAYER_REPLACEABILITY_MATRIX SSOT (RPL-A..H)."""

from __future__ import annotations

from functools import lru_cache
from typing import Final

from tests.qualification.ebh_5._vendor_backend_inventory import (
    concerns_with_extensibility_yes,
)
from tests.qualification.ebh_5._vendor_backend_types import (
    ExtensibilityClassification,
    ExtensibilityPromised,
    ReplaceabilityMatrixRow,
    RplGateStatus,
)


def _qualified_row(
    concern_id: str,
    *,
    behavioral: str,
    alternate: str,
    gates: tuple[str, ...],
    rpl_f: RplGateStatus = RplGateStatus.NA,
) -> ReplaceabilityMatrixRow:
    return ReplaceabilityMatrixRow(
        concern_id=concern_id,
        rpl_a=RplGateStatus.PASS,
        rpl_b=RplGateStatus.PASS,
        rpl_c=RplGateStatus.PASS,
        rpl_d=RplGateStatus.PASS,
        rpl_e=RplGateStatus.PASS,
        rpl_f=rpl_f,
        rpl_g=RplGateStatus.PASS,
        rpl_h=ExtensibilityClassification.REPLACEABLE_QUALIFIED,
        alternate_implementation_ref=alternate,
        behavioral_proof_ref=behavioral,
        regression_gate_ids=gates,
    )


def _not_yet_row(concern_id: str, *, gates: tuple[str, ...] = ("ebh5-matrix-integrity",)) -> ReplaceabilityMatrixRow:
    return ReplaceabilityMatrixRow(
        concern_id=concern_id,
        rpl_a=RplGateStatus.PASS,
        rpl_b=RplGateStatus.FAIL,
        rpl_c=RplGateStatus.PASS,
        rpl_d=RplGateStatus.PASS,
        rpl_e=RplGateStatus.FAIL,
        rpl_f=RplGateStatus.NA,
        rpl_g=RplGateStatus.PASS,
        rpl_h=ExtensibilityClassification.REPLACEABLE_NOT_YET,
        alternate_implementation_ref="PENDING",
        behavioral_proof_ref="PENDING",
        regression_gate_ids=gates,
    )


_QUALIFIED_OVERRIDES: dict[str, ReplaceabilityMatrixRow] = {
    "persistence.relational_sqlite_lab": _qualified_row(
        "persistence.relational_sqlite_lab",
        behavioral="INT-CONFIG-REAL-X @ a59744517b92847f55def1db22826d17d89ee155",
        alternate="relational_store configuration_realization strategy SPI",
        gates=("test_ebh_5_behavioral_relational_store", "test_config_x_qualification_gates"),
    ),
    "int_config.configure_existing": _qualified_row(
        "int_config.configure_existing",
        behavioral="ExistingCapabilityConfigurationRealizationService tests",
        alternate="sqlite configuration_realization reference strategy",
        gates=("test_config_x_qualification_gates",),
    ),
    "observability.otlp_transport": _qualified_row(
        "observability.otlp_transport",
        behavioral="HARNESS-W5 RecordingTransport @ 5a87046…",
        alternate="OtlpTransportPort local recording transport",
        gates=("test_ebh_5_behavioral_otlp_transport",),
    ),
    "tokenizer.selection": _qualified_row(
        "tokenizer.selection",
        behavioral="test_config_x_r1_remediation_gates tokenizer tests",
        alternate="SimpleTokenizer vs TiktokenTokenizer registration",
        gates=("test_config_x_r1_remediation_gates",),
    ),
    "integration.key_value_cache": _qualified_row(
        "integration.key_value_cache",
        behavioral="test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition",
        alternate="CustomMemoryKvPlugin via register_integration_plugin",
        gates=("test_ebh_5_behavioral_proofs", "test_external_plugin"),
    ),
}


@lru_cache(maxsize=1)
def replaceability_matrix_rows() -> tuple[ReplaceabilityMatrixRow, ...]:
    rows: list[ReplaceabilityMatrixRow] = []
    for concern in concerns_with_extensibility_yes():
        if concern.concern_id in _QUALIFIED_OVERRIDES:
            rows.append(_QUALIFIED_OVERRIDES[concern.concern_id])
            continue
        if concern.extensibility_classification == ExtensibilityClassification.REPLACEABLE_QUALIFIED:
            rows.append(
                _qualified_row(
                    concern.concern_id,
                    behavioral=concern.behavioral_qualification_evidence_id,
                    alternate=concern.structural_replaceability_evidence_id,
                    gates=("test_ebh_5_l2_integrity_gates",),
                ),
            )
            continue
        if concern.extensibility_promised == ExtensibilityPromised.YES:
            rows.append(_not_yet_row(concern.concern_id))
            continue
        if concern.extensibility_promised == ExtensibilityPromised.PARTIAL:
            rows.append(_not_yet_row(concern.concern_id))
    return tuple(rows)


CROSS_LAYER_REPLACEABILITY_MATRIX: Final[tuple[ReplaceabilityMatrixRow, ...]] = replaceability_matrix_rows()


def matrix_by_concern_id() -> dict[str, ReplaceabilityMatrixRow]:
    return {row.concern_id: row for row in CROSS_LAYER_REPLACEABILITY_MATRIX}


def undocumented_qualified_rows() -> tuple[str, ...]:
    bad: list[str] = []
    for row in CROSS_LAYER_REPLACEABILITY_MATRIX:
        if row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED:
            if row.behavioral_proof_ref in ("PENDING", "PENDING — EBH_5 Package C"):
                bad.append(row.concern_id)
    return tuple(bad)
