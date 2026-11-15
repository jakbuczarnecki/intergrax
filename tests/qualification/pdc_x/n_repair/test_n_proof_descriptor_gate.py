# © Artur Czarnecki. All rights reserved.

"""PDC-N (IA-GLOBAL G-05, residual c) — every platform proof descriptor is valid.

``platform_proofs/scenarios/indirect_prompt_injection/proof.json`` declared ``safety_class``
``LOCAL_NETWORK``, which is not a ``ProofSafetyClass`` member, so the proof suite failed with
``FAIL_MANIFEST`` and every test that runs the suite failed with it. This gate validates every
``proof.json`` below ``platform_proofs/`` against the descriptor model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from scripts.proof.intergrax_platform_proof_descriptor import PlatformProofDescriptor

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DESCRIPTORS = sorted((_REPO_ROOT / "platform_proofs").rglob("proof.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_descriptors_are_found() -> None:
    assert len(_DESCRIPTORS) >= 4


@pytest.mark.parametrize("path", _DESCRIPTORS, ids=lambda p: p.parent.name)
def test_every_proof_descriptor_validates(path: Path) -> None:
    PlatformProofDescriptor.model_validate(_load(path))


def test_the_indirect_prompt_injection_descriptor_is_local_mutating() -> None:
    path = _REPO_ROOT / "platform_proofs" / "scenarios" / "indirect_prompt_injection" / "proof.json"
    assert _load(path)["safety_class"] == "LOCAL_MUTATING"


def test_a_safety_class_outside_the_enum_is_rejected() -> None:
    path = _REPO_ROOT / "platform_proofs" / "scenarios" / "ai_incident_investigation" / "proof.json"
    broken = {**_load(path), "safety_class": "LOCAL_NETWORK"}
    with pytest.raises(ValidationError, match="safety_class"):
        PlatformProofDescriptor.model_validate(broken)
