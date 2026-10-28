# © Artur Czarnecki. All rights reserved.

"""T-CLS-A..J parametrized class gates — one pytest node per applicable layer×class."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_6._aj_class_gate_scan import (
    iter_applicable_layer_class_pairs,
    run_class_gate_scan,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_PAIRS = iter_applicable_layer_class_pairs()
_PAIR_IDS = [f"{layer}-{cls}" for layer, cls in _PAIRS]


@pytest.mark.parametrize(("layer", "detection_class"), _PAIRS, ids=_PAIR_IDS)
def test_ebh6_aj_class_gate(layer: str, detection_class: str) -> None:
    run_class_gate_scan(layer, detection_class)
