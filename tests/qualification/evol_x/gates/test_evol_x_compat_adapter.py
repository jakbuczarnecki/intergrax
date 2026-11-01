# © Artur Czarnecki. All rights reserved.

import pytest
from tests.qualification.evol_x.adapters.compat_x import CompatXRegistryAdapter

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_compat_x_adapter_reads_registry() -> None:
    adapter = CompatXRegistryAdapter()
    names = adapter.list_contract_names()
    assert names
    assert adapter.diff_schemas({"properties": {}}, {"properties": {}}).value == "COMPATIBLE"
