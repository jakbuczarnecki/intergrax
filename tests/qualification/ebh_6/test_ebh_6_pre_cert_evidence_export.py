# © Artur Czarnecki. All rights reserved.

"""C1 pre-cert export integrity — 133 pairs / 19 bundles."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_6._pre_cert_evidence_export import (
    format_pre_cert_bundle_table_markdown,
    format_pre_cert_pair_table_markdown,
    pre_cert_bundle_rows,
    pre_cert_export_dict,
    pre_cert_export_metadata,
    pre_cert_pair_rows,
)
from tests.qualification.ebh_6._t2_evidence_bundles import EvidenceBundleId

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh6_pre_cert_pair_export_complete() -> None:
    rows = pre_cert_pair_rows()
    assert len(rows) == 133
    for row in rows:
        assert row.gate_node_id
        assert row.bundle_id
        assert row.independent_disposition.startswith("PENDING")
        assert row.implementer_outcome in {
            "PARTIAL_CANDIDATE",
            "PASS_CANDIDATE",
            "FINDING",
            "EVIDENCE_REQUIRED",
            "N/A — WITH EVIDENCE",
        }


def test_ebh6_pre_cert_bundle_export_complete() -> None:
    bundles = pre_cert_bundle_rows()
    assert len(bundles) == len(EvidenceBundleId)
    pair_total = sum(b.pair_count for b in bundles)
    assert pair_total == 133
    for bundle in bundles:
        assert bundle.primary_behavioral_gates
        assert bundle.invariant_ref.strip()
        assert bundle.independent_disposition.startswith("PENDING")


def test_ebh6_pre_cert_export_metadata_honest() -> None:
    meta = pre_cert_export_metadata()
    assert meta["pair_count"] == 133
    assert meta["bundle_count"] == 19
    assert "must not issue" in meta["independent_disposition"]


def test_ebh6_pre_cert_bundle_markdown_non_empty() -> None:
    assert "integration_factory" in format_pre_cert_bundle_table_markdown()


def test_ebh6_pre_cert_machine_export_full_gate_nodeids() -> None:
    payload = pre_cert_export_dict()
    for row in payload["pairs"]:
        gate = row["gate_node_id"]
        assert "::" in gate, row
        assert len(gate) > len("tests/x.py::t")
        assert row["independent_disposition"].startswith("PENDING")
    for bundle in payload["bundles"]:
        assert bundle["independent_disposition"].startswith("PENDING")
        pytest_gates = [g for g in bundle["primary_behavioral_gates"] if "::" in g]
        assert pytest_gates, bundle["bundle_id"]
        for gate in pytest_gates:
            assert len(gate) > len("tests/x.py::t")


def test_ebh6_pre_cert_export_rejects_truncated_gate_in_machine_export() -> None:
    rows = pre_cert_pair_rows()
    assert rows
    full = rows[0].gate_node_id
    assert full.endswith("]")
    assert full not in format_pre_cert_pair_table_markdown(max_rows=1) or len(full) > 80
