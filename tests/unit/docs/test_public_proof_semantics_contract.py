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
_TOKEN_GUIDE = _REPO_ROOT / "docs/project/capabilities/token_optimization/README.md"


@pytest.fixture
def proofs_text() -> str:
    return _PROOFS.read_text(encoding="utf-8")


@pytest.fixture
def maintainer_model_text() -> str:
    return _MAINTAINER_MODEL.read_text(encoding="utf-8")


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
    lkw_section = proofs_text.split("## LKW — Primary product proof", 1)[1].split(
        "\n---\n", 1
    )[0]

    assert "Product Quick Start" in lkw_section
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
    lkw_section = proofs_text.split("## LKW — Primary product proof", 1)[1].split(
        "\n---\n", 1
    )[0]
    assert "### Additional bounded LKW proof paths" in lkw_section
    supporting_table = lkw_section.split("### Additional bounded LKW proof paths", 1)[1]
    assert "| Capability | Status | What it demonstrates | Limitation |" in supporting_table
    assert "| Web URL knowledge intake |" in supporting_table
    assert "| Ollama / vLLM model runtime portability |" in supporting_table


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
