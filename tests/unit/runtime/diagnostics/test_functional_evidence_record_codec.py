# © Artur Czarnecki. All rights reserved.

"""DIAG-DURABILITY-D1 functional evidence record codec tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    sample_functional_evidence,
    sample_functional_evidence_scope,
)
from intergrax.runtime.diagnostics.functional_evidence_record_codec import (
    decode_functional_evidence_record,
    decode_functional_evidence_record_bytes,
    encode_functional_evidence_record,
    encode_functional_evidence_record_bytes,
)

pytestmark = pytest.mark.unit


def test_functional_evidence_record_round_trip() -> None:
    evidence = sample_functional_evidence(scope=sample_functional_evidence_scope())
    encoded = encode_functional_evidence_record(evidence)
    decoded = decode_functional_evidence_record(encoded)
    assert decoded == evidence


def test_functional_evidence_record_bytes_round_trip() -> None:
    evidence = sample_functional_evidence(scope=sample_functional_evidence_scope())
    raw = encode_functional_evidence_record_bytes(evidence)
    decoded = decode_functional_evidence_record_bytes(raw)
    assert decoded == evidence


def test_functional_evidence_record_unknown_schema_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported functional evidence persistence schema"):
        decode_functional_evidence_record({"schema_version": "broken", "payload": {}})
