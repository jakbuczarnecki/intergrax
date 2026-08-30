# © Artur Czarnecki. All rights reserved.

"""Documentation contract tests for PP-2 Platform Proof System."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_PROOFS = REPO_ROOT / "platform_proofs"
PROTOCOL_PATH = PLATFORM_PROOFS / "PLATFORM_PROOF_PROTOCOL.md"
README_PATH = PLATFORM_PROOFS / "README.md"
AUTHORING_PATH = PLATFORM_PROOFS / "PLATFORM_PROOF_AUTHORING_GUIDE.md"
MANIFEST_PATH = REPO_ROOT / "scripts" / "proof" / "intergrax_proof_manifest.py"
RUNNER_PATH = REPO_ROOT / "scripts" / "proof" / "intergrax_proof_runner.py"
PUBLIC_PROOFS = REPO_ROOT / "docs" / "project" / "proofs" / "PROOFS.md"
PUBLIC_LIBRARY = REPO_ROOT / "docs" / "project" / "proofs" / "PROOF_LIBRARY.md"
SCENARIOS_ROOT = PLATFORM_PROOFS / "scenarios"

_DOCS_SCRIPTS = REPO_ROOT / "scripts" / "docs"
if str(_DOCS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_DOCS_SCRIPTS))
from docs_domain_common import canonical_domain_ids  # noqa: E402

_ADDITIONAL_DOMAINS = (
    "DECISION_SYSTEM",
    "GOVERNED_EXECUTION",
    "AGENT_DISTRIBUTION",
    "PLATFORM_PLUGINS",
    "PROOF_RECEIPTS",
)
ALL_CANONICAL_DOMAINS: tuple[str, ...] = canonical_domain_ids() + _ADDITIONAL_DOMAINS


@pytest.fixture
def protocol_text() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


@pytest.fixture
def readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


@pytest.fixture
def authoring_text() -> str:
    return AUTHORING_PATH.read_text(encoding="utf-8")


def test_platform_proofs_canonical_docs_exist() -> None:
    required = (
        README_PATH,
        PROTOCOL_PATH,
        AUTHORING_PATH,
    )
    missing = [path for path in required if not path.is_file()]
    assert not missing, f"missing platform_proofs canonical docs: {missing}"


def test_platform_proof_map_removed() -> None:
    assert not (PLATFORM_PROOFS / "PLATFORM_PROOF_MAP.md").is_file()


def test_scenario_conformance_are_only_top_level_taxonomy(protocol_text: str) -> None:
    assert "## B. Proof Library classes" in protocol_text
    assert "**SCENARIO**" in protocol_text
    assert "**CONFORMANCE**" in protocol_text
    assert "DOMAIN_PROOF" not in protocol_text
    assert "FEATURE_PROOF" not in protocol_text


def test_scenario_design_root_documented(readme_text: str) -> None:
    assert "platform_proofs/scenarios/" in readme_text
    assert "ai_incident_investigation" in readme_text
    assert SCENARIOS_ROOT.is_dir()


def test_public_scenario_catalog_linked_from_gateway(readme_text: str) -> None:
    assert PUBLIC_LIBRARY.is_file()
    assert "docs/project/proofs/PROOF_LIBRARY.md" in readme_text


def test_no_manual_scenario_registry_in_gateway(readme_text: str) -> None:
    assert "no" in readme_text.lower() and "manual scenario registry" in readme_text.lower()


def test_lkw_not_platform_proof(protocol_text: str, readme_text: str) -> None:
    assert "LKW is a product" in protocol_text
    assert "LKW" in readme_text
    assert "`LKW`" not in protocol_text.split("## B. Proof Library classes", 1)[0]
    lkw_domain_row = re.search(r"^\| `LKW` \|", protocol_text, re.MULTILINE)
    assert lkw_domain_row is None


def test_applications_product_only_ownership_in_protocol_and_readme(
    protocol_text: str,
    readme_text: str,
) -> None:
    normative = (
        "product-specific execution **MUST NOT** substitute for an independently owned Platform Proof"
    )
    assert normative in protocol_text
    assert normative in readme_text
    assert "applications/" in protocol_text
    assert "product proofs" in protocol_text.lower()
    assert "LKW" in protocol_text
    assert "applications/" in readme_text


def test_scenario_application_not_product_boundary(protocol_text: str) -> None:
    assert "Scenario application ≠ Product" in protocol_text
    assert "platform_proofs/scenarios/" in protocol_text


def test_scripts_proof_remains_canonical_execution_infrastructure(
    protocol_text: str,
    readme_text: str,
) -> None:
    assert MANIFEST_PATH.is_file()
    assert RUNNER_PATH.is_file()
    assert "scripts/proof/intergrax_proof_manifest.py" in protocol_text
    assert "scripts/proof/intergrax_proof_runner.py" in protocol_text
    assert "SuiteReceipt" in protocol_text
    assert "scripts/proof/" in readme_text


def test_public_proofs_dashboard_ownership_preserved(protocol_text: str) -> None:
    assert PUBLIC_PROOFS.is_file()
    assert "docs/project/proofs/PROOFS.md" in protocol_text
    assert "PUBLIC_PROOF_AND_CLAIMS_MODEL" in protocol_text


def test_no_duplicate_proof_runner_or_manifest_paths() -> None:
    """PP-2 must not introduce competing manifest/runner under platform_proofs/."""
    platform_tree = PLATFORM_PROOFS
    forbidden_names = {
        "intergrax_proof_manifest.py",
        "intergrax_proof_runner.py",
        "intergrax_proof_contracts.py",
        "proof_manifest.py",
        "proof_runner.py",
    }
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in platform_tree.rglob("*.py")
        if path.name in forbidden_names
    ]
    assert not offenders, f"duplicate proof infrastructure under platform_proofs/: {offenders}"

    for name in ("intergrax_proof_manifest.py", "intergrax_proof_runner.py"):
        matches = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in REPO_ROOT.rglob(name)
            if path.relative_to(REPO_ROOT).as_posix() != f"scripts/proof/{name}"
        ]
        assert not matches, f"competing canonical {name} outside scripts/proof/: {matches}"


def test_authoring_guide_defines_production_capable_scenario_application(
    authoring_text: str,
) -> None:
    assert "production-capable autonomous application component" in authoring_text
    assert "it does not replace the application itself" in authoring_text
    assert "production-validated" in authoring_text
    assert "Do **not** write \"production-proven\" or \"production-validated\"" in authoring_text


def test_authoring_guide_contains_application_survival_test(
    authoring_text: str,
) -> None:
    assert "### Application Survival Test" in authoring_text
    assert "If proof infrastructure, evaluator, evidence packaging, and report generation" in (
        authoring_text
    )
    assert "**NO** | **Not acceptable** as SCENARIO" in authoring_text


def test_authoring_guide_contains_observability_contract(
    authoring_text: str,
) -> None:
    assert "### Application Observability Test" in authoring_text
    assert "### Mandatory observability, explainability, and diagnostics" in authoring_text
    assert "MUST NOT** be a black box" in authoring_text
    assert "Hidden chain-of-thought is neither required nor accepted" in authoring_text
    assert "MUST NOT** invent execution explanations" in authoring_text
    assert "RuntimeState.trace_events" in authoring_text
    assert "ToolCallTrace" in authoring_text


def test_protocol_requires_scenario_observability(
    protocol_text: str,
) -> None:
    assert "## G2. SCENARIO observability, explainability, and diagnostics" in protocol_text
    assert "MUST NOT** be a black box" in protocol_text
    assert "Application Observability Test" in protocol_text
    assert "MUST NOT** invent post-hoc execution explanations" in protocol_text
    assert "PASS cannot rely solely on final answer" in protocol_text
    assert "renderer **MUST NOT** invent tool calls" in protocol_text


def test_report_standard_scenario_source_of_truth_rule() -> None:
    report_text = (
        REPO_ROOT / "docs" / "project" / "proofs" / "PLATFORM_PROOF_REPORT_STANDARD.md"
    ).read_text(encoding="utf-8")
    assert "### 7.2 SCENARIO report presentation profile (normative)" in report_text
    assert "MUST NOT** invent tool calls" in report_text
    assert "intergrax.platform_proof_evidence.v3" in report_text
    assert "PLATFORM_PROOF_MAP" not in report_text


def test_authoring_guide_prohibits_fake_llm_in_canonical_scenario_path(
    authoring_text: str,
) -> None:
    assert "FakeLLMAdapter" in authoring_text
    assert "**Scenario canonical application path**" in authoring_text
    assert "**PROHIBITED**" in authoring_text


def test_protocol_distinguishes_scenario_vs_conformance_mock_policy(
    protocol_text: str,
) -> None:
    assert "**Scenario canonical application path**" in protocol_text
    assert "**Conformance proof**" in protocol_text
    assert "Allowed only **outside** claimed boundary" in protocol_text
    assert "FakeLLM" in protocol_text


def test_gateway_defines_scenario_as_production_capable_component(
    readme_text: str,
) -> None:
    assert "Production-capable autonomous application component" in readme_text
    assert "falsification" in readme_text.lower() or "falsifies" in readme_text.lower()
    assert "PLATFORM_PROOF_AUTHORING_GUIDE.md" in readme_text
    assert "does **not** substitute for the application" in readme_text
    assert "create_scenario_proof.py" in readme_text


def test_protocol_scenario_proof_identity_convention(protocol_text: str) -> None:
    assert "SCENARIO-<SCENARIO-ID>" in protocol_text
    assert "SCENARIO-AI-INCIDENT-INVESTIGATION" in protocol_text
