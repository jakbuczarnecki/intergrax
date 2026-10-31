# © Artur Czarnecki. All rights reserved.

"""Independent audit corrective findings — false PASS gates (corrected)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_7._gate_strength import GateStrength


@dataclass(frozen=True, slots=True)
class GateFinding:
    gate_id: str
    prior_claim: str
    actual_strength: GateStrength
    required_strength: GateStrength
    correction: str


EBH7_CORRECTIVE_GATE_FINDINGS: Final[tuple[GateFinding, ...]] = (
    GateFinding(
        gate_id="W6-ID-REGISTRY-ONLY",
        prior_claim="26/26 PASS via scenario_id tuple",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.E2E,
        correction="Split structural registry vs adversarial proof execution; pending scenarios stay OPEN",
    ),
    GateFinding(
        gate_id="W6-TENANT-X-FINAL-READY",
        prior_claim="Single TENANT-X readiness test = platform E2E",
        actual_strength=GateStrength.INTEGRITY,
        required_strength=GateStrength.E2E,
        correction="Removed as E2E substitute; mapped proofs only where semantically aligned",
    ),
    GateFinding(
        gate_id="W7-HASATTR-DUP-X",
        prior_claim="hasattr(dup_exit, test_name) PASS",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.BEHAVIORAL,
        correction="Execute registered proof callables; pending XDM scenarios remain OPEN",
    ),
    GateFinding(
        gate_id="W5-RPL-ROW-COUNT",
        prior_claim="56-row disposition record = RPL-H qualified",
        actual_strength=GateStrength.INTEGRITY,
        required_strength=GateStrength.BEHAVIORAL,
        correction="Structural inventory separated; 51 NOT YET remain QUALIFICATION_PENDING",
    ),
    GateFinding(
        gate_id="W5-VBN-DOC-PRESENCE",
        prior_claim="Seven markdown artifacts = VBN PASS",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.BEHAVIORAL,
        correction="Artifact presence is MATERIALIZED only",
    ),
    GateFinding(
        gate_id="TRACE-466-COUNT-ONLY",
        prior_claim="466 unique IDs = behavioral coverage",
        actual_strength=GateStrength.INTEGRITY,
        required_strength=GateStrength.CONTRACT,
        correction="Extended rows with owner, evidence, acceptance, dual status fields",
    ),
    GateFinding(
        gate_id="READINESS-TABLE-PASS",
        prior_claim="W6/W7/W5 PASS in readiness doc",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.ACCEPTANCE,
        correction="Readiness reconciled to materialization vs qualification columns",
    ),
    GateFinding(
        gate_id="GAP-TEN-UNK-05-CLOSED",
        prior_claim="N/A disposition = tenant E2E closed",
        actual_strength=GateStrength.CONTRACT,
        required_strength=GateStrength.E2E,
        correction="Structural N/A evidence only; not platform E2E proof",
    ),
    GateFinding(
        gate_id="AUD-01-RPL-AUTO-QUALIFIED",
        prior_claim="56/56 semantic QUALIFIED via registry membership",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.BEHAVIORAL,
        correction="Normative RPL layer + appendix; factory/blocked/dual-slug demoted until OD or real proof",
    ),
    GateFinding(
        gate_id="B6-NORMATIVE-OPEN-ROWS",
        prior_claim="L3 green = B6 ready",
        actual_strength=GateStrength.INTEGRITY,
        required_strength=GateStrength.ACCEPTANCE,
        correction=(
            "23 RPL concerns remain (17 factory PARTIAL + 6 vendor BLOCKED); "
            "28 dual-slug qualified via RPL-C/D/E; EBH_7_RPL_FINAL_CONVERGENCE_DECISION_MATRIX.md"
        ),
    ),
)


def assert_ebh7_corrective_findings_registered() -> None:
    assert len(EBH7_CORRECTIVE_GATE_FINDINGS) >= 10
