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
    GateFinding(
        gate_id="B6-F01-EXTERNAL-WORK-DUAL-INSTANCE",
        prior_claim="Two InMemoryExternalWorkIntegration instances = replaceability",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.BEHAVIORAL,
        correction=(
            "QualExternalWorkAlpha/Beta integrations + IntegrationProfile selection; "
            "contract ops, failure isolation, profile fail-closed"
        ),
    ),
    GateFinding(
        gate_id="B6-F02-PRODQ-TEXT-PREFIX-NEGATIVE",
        prior_claim="factory.py startswith('prodq gate') = PROD-Q negative",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.BEHAVIORAL,
        correction=(
            "_internal_rpl_behavioral_proofs: VenvBundleColocated positive; "
            "FakeInMemory adapter → StrictProductionServingLauncherError"
        ),
    ),
    GateFinding(
        gate_id="B6-F03-TEN-XDM-DOC-CONFLICT",
        prior_claim="Corrective findings doc listed 12+18 OPEN as if current",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.INTEGRITY,
        correction=(
            "Historical superseded table + B6_EVIDENCE_CONSISTENCY_MATRIX; SSOT 26/0 TEN and 34/0 XDM @ registry"
        ),
    ),
    GateFinding(
        gate_id="B6-F04-P03-TEST-CLOSURE-SHA",
        prior_claim="TEST_EXECUTION_SHA must equal EBH7_CLOSURE_IMPLEMENTATION_SHA",
        actual_strength=GateStrength.INTEGRITY,
        required_strength=GateStrength.INTEGRITY,
        correction=(
            "TEST_EXECUTION_SHA = EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA; "
            "closure SHA is ancestor only — _ebh7_p03_evidence_integrity.py"
        ),
    ),
    GateFinding(
        gate_id="B6-F07-RPL-EF-PROOF-STRENGTH",
        prior_claim="Dual-slug RPL-E = distinct provider_id + unregistered reject",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.CONTRACT,
        correction=(
            "Contract-class typing, profile/provider identity, wrong-slug fail-closed, "
            "relational execute(SELECT 1); per-category RPL-F N/A — _dual_slug_rpl_ef_contract_evidence.py"
        ),
    ),
    GateFinding(
        gate_id="B6-F08-ECAP-P03-DERIVED-GATES",
        prior_claim="G07/G08 closure rows unconditional PASS",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.INTEGRITY,
        correction="_derive_ecap_gate_status / _derive_p03_gate_status fail-closed @ certification + P03 SSOT",
    ),
    GateFinding(
        gate_id="B6-F09-FINDING-RESOLUTION-EVIDENCE",
        prior_claim="_RESOLVED_FINDING_IDS allowlist = G09 PASS",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.ACCEPTANCE,
        correction="_gate_finding_resolution.py — pytest, assert hook, or documented supersession per finding",
    ),
    GateFinding(
        gate_id="B6-F10-ECAP-NORMATIVE-C1-C5",
        prior_claim="Checklist counters + PROHIBITED zero = full ECAP C1–C5",
        actual_strength=GateStrength.INTEGRITY,
        required_strength=GateStrength.ACCEPTANCE,
        correction=(
            "Computed certification over closed obligation sets (_ebh7_f10_canonical_inputs.py + "
            "_ebh7_f10_ecap_normative_matrix.py); matrix and §14 block generated, verified read-only"
        ),
    ),
    GateFinding(
        gate_id="B6-F10-C1-STALE-GATE-REG",
        prior_claim="FRZ-REG family gate node exists because its file exists",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.INTEGRITY,
        correction="Gate rebound to renamed successor; C1 requires every family gate node to resolve and execute",
    ),
    GateFinding(
        gate_id="B6-F10-C4-STATE-REG-01",
        prior_claim="SQLite conversational memory round-trip repaired by B6 corrective",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.ACCEPTANCE,
        correction="load_memory validates closed MessageRole set instead of calling typing.Literal",
    ),
    GateFinding(
        gate_id="B6-F10-P03-PROFILE-SHADOW",
        prior_claim="MANIFEST profile argument tests/qualification/ebh_7/ executes the whole directory",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.INTEGRITY,
        correction="Shadowing member files removed; coverage helper rejects directory shadowed by member files",
    ),
    GateFinding(
        gate_id="B6-F10-P03-DIRTY-TREE",
        prior_claim="EBH-7 W1 mechanism registry = 685 disjoint MECH IDs",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.INTEGRITY,
        correction=(
            "685 counted 447 composition paths from the gitignored docker/runtime-context copy; "
            "wiring discovery excludes it — 238 on a clean checkout"
        ),
    ),
    GateFinding(
        gate_id="B6-F10-CFG-ISO-01",
        prior_claim="CONFIG-X FRZ-CFG-07 gates are order-independent",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.INTEGRITY,
        correction="Fixture restores default integration catalog and bootstrap state after clear_catalog()",
    ),
    GateFinding(
        gate_id="B6-F11-DOSSIER-STALE-REPLAY",
        prior_claim="Dossier replay profile and HEAD matched pre-B5 closure",
        actual_strength=GateStrength.STRUCTURAL,
        required_strength=GateStrength.INTEGRITY,
        correction=(
            "Sync dossier, traceability, checklist with MANIFEST + B5 GHA 41214574614; "
            "preserve historical SHA provenance"
        ),
    ),
)


def assert_ebh7_corrective_findings_registered() -> None:
    assert len(EBH7_CORRECTIVE_GATE_FINDINGS) >= 10
