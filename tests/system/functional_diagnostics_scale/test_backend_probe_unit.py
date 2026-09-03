# © Artur Czarnecki. All rights reserved.

"""Unit tests for S1 scale backend probe contract."""

from __future__ import annotations

from tests.system.functional_diagnostics_scale.mongodb_backend import (
    MongoFunctionalDiagnosticsScaleProbe,
)
from tests.system.functional_diagnostics_scale.synthetic_backend import (
    SyntheticFunctionalDiagnosticsScaleProbe,
)


def test_synthetic_scale_probe_pluginability() -> None:
    probe = SyntheticFunctionalDiagnosticsScaleProbe()
    probe.prepare()
    store = probe.build_document_store()
    identity = probe.backend_identity()
    assert identity.provider_id == "synthetic-in-memory"
    probe.close_document_store(store)
    probe.cleanup()


def test_mongo_scale_probe_production_indexes() -> None:
    probe = MongoFunctionalDiagnosticsScaleProbe(collection_name="unit-scale-probe")
    indexes = probe.production_index_observations()
    assert len(indexes) == 1
    assert indexes[0].unique is True
    assert indexes[0].index_name == "uq_intergrax_document_key"
