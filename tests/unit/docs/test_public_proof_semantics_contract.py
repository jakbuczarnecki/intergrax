# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROOFS = _REPO_ROOT / "docs/project/proofs/PROOFS.md"
_MAINTAINER_MODEL = (
    _REPO_ROOT
    / "docs/project/maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md"
)
_DOC_ARCH = (
    _REPO_ROOT
    / "docs/project/maintainers/public-adoption/PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
)
_TOKEN_GUIDE = _REPO_ROOT / "docs/project/capabilities/token_optimization/README.md"
_GOVERNED_PROOF = (
    _REPO_ROOT
    / "applications/local_workspace_application/docs/proof/GOVERNED_HYBRID_KNOWLEDGE_PROOF.md"
)
_LKW_PRODUCT_TOUR = (
    _REPO_ROOT
    / "applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md"
)
_PUBLIC_DOCUMENTATION_MAP = (
    _REPO_ROOT / "docs/project/community/PUBLIC_DOCUMENTATION_MAP.md"
)


@pytest.fixture
def proofs_text() -> str:
    return _PROOFS.read_text(encoding="utf-8")


@pytest.fixture
def maintainer_model_text() -> str:
    return _MAINTAINER_MODEL.read_text(encoding="utf-8")


@pytest.fixture
def doc_arch_text() -> str:
    return _DOC_ARCH.read_text(encoding="utf-8")


@pytest.fixture
def governed_proof_text() -> str:
    return _GOVERNED_PROOF.read_text(encoding="utf-8")


def test_governed_flagship_proof_distinguishes_controlled_live_from_external_saas(
    governed_proof_text: str,
    proofs_text: str,
) -> None:
    """Flagship proof must claim controlled Docker-backed runtime, not external SaaS."""
    for phrase in (
        "controlled live provider",
        "external live provider",
        "not external SaaS",
        "real runtime",
        "Docker-backed",
    ):
        assert phrase in governed_proof_text, f"Missing governed proof phrase: {phrase}"

    assert "four independent live providers" not in governed_proof_text.lower()
    assert "four live organizational sources" not in governed_proof_text.lower()
    assert "four independent live provider connections" not in governed_proof_text.lower()

    lkw_section = proofs_text.split("## LKW — Active reference product evidence", 1)[1].split(
        "\n---\n", 1
    )[0]
    governed_row = re.search(
        r"^\| \*\*Governed Evidence Decision Proof\*\* \|.*$",
        lkw_section,
        re.MULTILINE,
    )
    assert governed_row is not None
    row = governed_row.group(0).lower()
    assert "controlled live provider" in row
    assert "not external saas" in row

    not_established = proofs_text.split("### Not established by the accepted public proof", 1)[1]
    lkw_not_established = not_established.split("## Token Optimization", 1)[0]
    assert "external" in lkw_not_established.lower()
    assert "external saas validation" in lkw_not_established.lower() or (
        "controlled live provider" in lkw_not_established.lower()
        and "not external saas" in lkw_not_established.lower()
    )


def test_public_lkw_projection_maintains_controlled_live_boundary(
    governed_proof_text: str,
    proofs_text: str,
) -> None:
    """Product Tour and Public Map must not widen flagship proof beyond controlled live."""
    tour_text = _LKW_PRODUCT_TOUR.read_text(encoding="utf-8").lower()
    map_text = _PUBLIC_DOCUMENTATION_MAP.read_text(encoding="utf-8").lower()

    for phrase in (
        "controlled live provider",
        "not external saas",
        "real runtime",
        "docker-backed",
    ):
        assert phrase in governed_proof_text.lower(), f"canonical proof missing: {phrase}"

    lkw_section = proofs_text.split("## LKW — Active reference product evidence", 1)[1].split(
        "\n---\n", 1
    )[0]
    governed_row = re.search(
        r"^\| \*\*Governed Evidence Decision Proof\*\* \|.*$",
        lkw_section,
        re.MULTILINE,
    )
    assert governed_row is not None
    row = governed_row.group(0).lower()
    assert "controlled live provider" in row
    assert "not external saas" in row

    assert "four independent live providers" not in tour_text
    assert "four independent live providers" not in map_text

    for projection_text, label in ((tour_text, "product tour"), (map_text, "public map")):
        assert "controlled live provider" in projection_text, f"{label} missing controlled live"
        assert "not external saas" in projection_text, f"{label} missing SaaS boundary"
        assert "real runtime" in projection_text or "real http" in projection_text, (
            f"{label} missing runtime/HTTP execution marker"
        )

    assert "complete external live-provider access" in tour_text


def test_token_offline_proof_not_masquerading_as_vllm_claim(proofs_text: str) -> None:
    """RUNTIME-TOKEN-OPTIMIZATION-OFFLINE must not be the cited proof for vLLM rows."""
    token_section = proofs_text.split("## Token Optimization", 1)[1]
    proof_line = re.search(
        r"^\*\*Proof:\*\* `RUNTIME-TOKEN-OPTIMIZATION-OFFLINE`$",
        token_section,
        re.MULTILINE,
    )
    assert proof_line is not None

    before_proof = token_section[: proof_line.start()]
    assert "Bounded offline smoke proof" in before_proof
    assert "offline_smoke" in before_proof or "offline smoke" in before_proof.lower()

    vllm_row = re.search(
        r"^\| vLLM prefix-cache reuse \|.*$",
        token_section,
        re.MULTILINE,
    )
    assert vllm_row is not None
    assert "RUNTIME-TOKEN-OPTIMIZATION-OFFLINE" not in vllm_row.group(0)

    guide = _TOKEN_GUIDE.read_text(encoding="utf-8")
    assert "Bounded offline smoke proof" in guide
    assert "manual vLLM" in guide or "Manual live evidence" in guide


def test_web_url_pass_wording_protects_insufficient_evidence_boundary(
    proofs_text: str,
) -> None:
    web_row = re.search(
        r"^\| Web URL knowledge intake \|.*$",
        proofs_text,
        re.MULTILINE,
    )
    assert web_row is not None
    row = web_row.group(0).lower()
    assert "insufficient_evidence" in row
    assert "indexed retrieval" in row
    assert "status=completed" in row or "completed" in row
    assert "grounded" in row
    assert "citation" in row or "evidence" in row


def test_proofs_surface_lkw_product_quickstart_taxonomy(proofs_text: str) -> None:
    lkw_section = proofs_text.split("## LKW — Active reference product evidence", 1)[1].split(
        "\n---\n", 1
    )[0]

    assert "Product Quick Start" in lkw_section
    assert "Active reference product" in lkw_section
    assert "Primary product proof" not in lkw_section
    assert "accepted bounded proof paths" in lkw_section.lower()
    assert "indexed Ask V1" in lkw_section or "Ask V1" in lkw_section
    assert "../../../applications/local_workspace_application/docs/product/QUICKSTART.md" in lkw_section

    for proof_id in (
        "LKW-PRODUCT-QUICKSTART-WINDOWS",
        "LKW-PRODUCT-QUICKSTART-LINUX",
        "LKW-PRODUCT-QUICKSTART-MACOS",
    ):
        assert proof_id in lkw_section

    quickstart_row = re.search(
        r"^\| \*\*Product Quick Start / indexed Ask V1\*\* \|.*$",
        lkw_section,
        re.MULTILINE,
    )
    assert quickstart_row is not None
    row = quickstart_row.group(0).lower()
    assert "not the hybrid ask verification path" in row
    assert "hybrid ask verification path" not in row.replace(
        "not the hybrid ask verification path", ""
    )

    assert "🟡 **PARTIAL**" in proofs_text
    assert "**Trusted Ask / durable indexed workspace Ask**" in lkw_section
    assert "**Core Platform Proof**" in lkw_section
    assert "Mixed indexed + authorized live Hybrid Ask remains incomplete" in lkw_section

    quickstart_pos = lkw_section.index("| **Product Quick Start / indexed Ask V1** |")
    hybrid_pos = lkw_section.index("| **Indexed Hybrid Ask** |")
    trusted_pos = lkw_section.index("| **Trusted Ask / durable indexed workspace Ask** |")
    core_pos = lkw_section.index("| **Core Platform Proof** |")
    assert quickstart_pos < hybrid_pos < trusted_pos < core_pos


def test_proofs_additional_bounded_lkw_proof_table_structure(proofs_text: str) -> None:
    lkw_section = proofs_text.split("## LKW — Active reference product evidence", 1)[1].split(
        "\n---\n", 1
    )[0]
    assert "### Additional bounded LKW proof paths" in lkw_section
    supporting_table = lkw_section.split("### Additional bounded LKW proof paths", 1)[1]
    assert "| Capability | Status | What it demonstrates | Limitation |" in supporting_table
    assert "| Web URL knowledge intake |" in supporting_table
    assert "| Ollama / vLLM model runtime portability |" in supporting_table


def test_maintainer_lkw_classification_is_active_reference_product(
    maintainer_model_text: str,
) -> None:
    lkw_section = maintainer_model_text.split("## 6. LKW classification rules", 1)[1].split(
        "## 7.", 1
    )[0]
    assert "Active reference product" in lkw_section
    assert "Primary product proof" not in lkw_section
    assert "LKW is the Primary product proof" not in lkw_section
    assert "PARTIAL" in lkw_section
    assert "Backend Product Alpha / MVP" in lkw_section
    assert "LKW_PLATFORM_PROOF.md" in lkw_section
    assert "IMPLEMENTATION_PLAN.md" in lkw_section
    assert "PROOFS.md" in lkw_section
    assert "LKW product identity" in lkw_section
    assert "proof/evidence paths" in lkw_section
    assert "bounded live proof" in maintainer_model_text
    assert "product workflow proof" in maintainer_model_text
    assert "implemented code" in maintainer_model_text
    assert "≠ live proof" in maintainer_model_text


def test_doc_arch_lkw_product_separated_from_proof_identity(doc_arch_text: str) -> None:
    lkw_placement = doc_arch_text.split("## 4. LKW placement contract", 1)[1].split(
        "## 4a.", 1
    )[0]
    layer3 = doc_arch_text.split("### Layer 3 — Proofs and capability spotlights", 1)[1].split(
        "### Layer 4", 1
    )[0]

    assert "active reference product" in lkw_placement.lower()
    assert "primary product-development and product-proof path" not in lkw_placement.lower()
    assert "LKW_PRODUCT_TOUR.md" in lkw_placement
    assert "QUICKSTART.md" in lkw_placement
    assert "LKW_PLATFORM_PROOF.md" in lkw_placement
    assert "product quickstart ≠ platform proof" in lkw_placement.lower()

    assert "Active reference product" in layer3
    assert "Bounded technical LKW proof" in layer3
    assert "Primary product proof" not in layer3
    assert "Featured platform-capability proof" in layer3
    assert "does not define LKW product identity" in layer3

    assert "PROOFS.md" in doc_arch_text
    assert "public evidence dashboard" in doc_arch_text.lower()
    assert "PDOC-5B" in doc_arch_text


def test_maintainer_contract_contains_executable_binding_rules(
    maintainer_model_text: str,
) -> None:
    assert "## 12. Executable public proof binding" in maintainer_model_text
    for phrase in (
        "public_evidence_eligible=True",
        "structural",
        "Never invent `proof_id`",
        "SuiteReceipt",
        "ProofReceipt",
        "Duplicate proof references",
        "execution-selection profiles",
        "reader-oriented",
    ):
        assert phrase in maintainer_model_text, f"Missing maintainer contract phrase: {phrase}"
