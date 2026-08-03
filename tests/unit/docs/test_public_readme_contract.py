# © Artur Czarnecki. All rights reserved.

"""PUBLIC-DOCS-COMMERCIALIZATION-5: root README product-first landing contract tests."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
HERO_LIGHT_PATH = REPO_ROOT / "docs" / "assets" / "public" / "intergrax-hero-light.svg"
HERO_DARK_PATH = REPO_ROOT / "docs" / "assets" / "public" / "intergrax-hero-dark.svg"

_PRIMARY_SENTENCE = (
    "Intergrax helps teams build specialized agent applications without "
    "rebuilding the same policy, knowledge, evidence, integration, and "
    "execution foundations for every product."
)
_CATEGORY_DESCRIPTOR = (
    "Intergrax is a reusable Harness AI foundation for governed agent applications."
)

_PROBLEM_HEADING = "## Building the agent is not the hard part"

_SECTION_HEADINGS_ORDER = (
    _PROBLEM_HEADING,
    "## What Intergrax changes",
    "## Product proof: Local Knowledge Workspace",
    "## What is proven today",
    "## Featured platform capability: Token Optimization",
    "## How Intergrax works",
    "## Quick start",
    "## Choose your path",
    "## License and collaboration",
)

_REQUIRED_PUBLIC_LINKS = (
    "docs/public-adoption/LKW_PLATFORM_PROOF.md",
    "docs/features/token_optimization/README.md",
    "PROOFS.md",
    "docs/PUBLIC_DOCUMENTATION_MAP.md",
    "docs/DOCUMENTATION_MAP.md",
    "EVALUATION_GUIDE.md",
    "PARTNERS.md",
    "COLLABORATION.md",
    "LICENSE",
)

_COMPATIBILITY_ANCHORS = (
    "quick-start",
    "proof-of-platform",
    "start-here",
    "harness-ai--the-core-idea",
    "the-agent-model--why-architects-choose-intergrax",
)

_FORBIDDEN_SAVINGS_PHRASES = (
    "production-proven savings",
    "universal token reduction",
    "guaranteed token savings",
)

_FORBIDDEN_SAVINGS_PATTERN = re.compile(r"reduces token usage by\s*\d+\s*%", re.IGNORECASE)
_PERCENT_PATTERN = re.compile(r"\d+\s*%")

_MERMAID_FENCE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def test_canonical_positioning(readme_text: str) -> None:
    assert _PRIMARY_SENTENCE in readme_text
    assert _CATEGORY_DESCRIPTOR in readme_text


def test_section_order(readme_text: str) -> None:
    positions = [readme_text.index(heading) for heading in _SECTION_HEADINGS_ORDER]
    assert positions == sorted(positions), "README section headings are out of required order"


def test_required_public_links(readme_text: str) -> None:
    for link in _REQUIRED_PUBLIC_LINKS:
        assert link in readme_text, f"Missing required public link: {link}"


def test_hero_contract(readme_text: str) -> None:
    assert "docs/assets/public/intergrax-hero-light.svg" in readme_text
    assert "docs/assets/public/intergrax-hero-dark.svg" in readme_text
    assert "<picture>" in readme_text
    assert 'alt="Intergrax connects specialized agent applications' in readme_text
    assert HERO_LIGHT_PATH.is_file(), "Hero light SVG is missing"
    assert HERO_DARK_PATH.is_file(), "Hero dark SVG is missing"


def _parse_svg(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _collect_svg_violations(root: ET.Element) -> list[str]:
    violations: list[str] = []
    tag = root.tag.rsplit("}", 1)[-1] if "}" in root.tag else root.tag
    if tag == "script":
        violations.append("script element")
    if tag == "foreignObject":
        violations.append("foreignObject element")

    for elem in root.iter():
        local_tag = elem.tag.rsplit("}", 1)[-1] if "}" in elem.tag else elem.tag
        if local_tag in {"script", "foreignObject"}:
            violations.append(f"{local_tag} element")
        for attr_name, attr_value in elem.attrib.items():
            local_attr = attr_name.rsplit("}", 1)[-1] if "}" in attr_name else attr_name
            if local_attr.lower().startswith("on"):
                violations.append(f"event handler attribute: {local_attr}")
            if isinstance(attr_value, str):
                if re.search(r"https?://", attr_value):
                    violations.append(f"external URL in attribute {local_attr}")
                if "data:" in attr_value and "base64" in attr_value:
                    violations.append("base64 data URI")

    if root.get("viewBox") is None:
        violations.append("missing viewBox")
    if root.find(".//{*}title") is None and root.find("title") is None:
        violations.append("missing title")
    if root.find(".//{*}desc") is None and root.find("desc") is None:
        violations.append("missing desc")

    return violations


@pytest.mark.parametrize("svg_path", [HERO_LIGHT_PATH, HERO_DARK_PATH])
def test_svg_safety(svg_path: Path) -> None:
    root = _parse_svg(svg_path)
    violations = _collect_svg_violations(root)
    assert not violations, f"{svg_path.name}: {violations}"


def test_visual_contract(readme_text: str) -> None:
    blocks = _MERMAID_FENCE.findall(readme_text)
    assert len(blocks) >= 3, "README must contain at least three Mermaid blocks"
    forbidden_tokens = ("classDef", "style", "%%{init", "theme", "http://", "https://")
    for block in blocks:
        for token in forbidden_tokens:
            assert token not in block, f"Forbidden Mermaid token {token!r} found"


def test_public_maturity_boundary(readme_text: str) -> None:
    normalized = re.sub(r"[*_`]", "", readme_text).lower()
    for phrase in (
        "source-available",
        "active r&d",
        "backend product alpha",
        "real-user validation",
        "commercial validation",
    ):
        assert phrase in normalized, f"Missing maturity boundary phrase: {phrase}"
    assert "incomplete" in normalized


def test_token_optimization_claim_boundary(readme_text: str) -> None:
    for phrase in (
        "PARTIAL",
        "bounded vLLM",
        "durable in-cache compaction incomplete",
        "universal savings not claimed",
    ):
        assert phrase in readme_text, f"Missing Token Optimization boundary: {phrase}"

    lower = readme_text.lower()
    for phrase in _FORBIDDEN_SAVINGS_PHRASES:
        if phrase in lower:
            idx = lower.index(phrase)
            context = lower[max(0, idx - 40) : idx + len(phrase) + 40]
            assert "not claimed" in context or "not complete" in context, (
                f"Forbidden phrase {phrase!r} used positively"
            )

    assert not _FORBIDDEN_SAVINGS_PATTERN.search(readme_text)
    assert not _PERCENT_PATTERN.search(readme_text), "Numeric savings percentage found in README"


def test_evidence_script_compatibility() -> None:
    from scripts.maintenance.check_evidence_artifacts import _check_readme

    missing, proof_path_ok, boundaries_ok, links_ok = _check_readme(REPO_ROOT)
    assert proof_path_ok, missing
    assert boundaries_ok, missing
    assert links_ok, missing


def test_relative_links(readme_text: str) -> None:
    for _label, target in _MD_LINK.findall(readme_text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if target.startswith("#"):
            continue
        clean = target.split("#", 1)[0].strip()
        if not clean:
            continue
        if clean.startswith("http"):
            continue
        resolved = (REPO_ROOT / clean).resolve()
        assert resolved.exists(), f"Broken relative link target: {target}"


def test_quick_start_anchor(readme_text: str) -> None:
    assert "## Quick start" in readme_text
    for anchor in _COMPATIBILITY_ANCHORS:
        if anchor == "quick-start":
            assert "## Quick start" in readme_text
        else:
            assert f'id="{anchor}"' in readme_text, f"Missing compatibility anchor: {anchor}"


def test_brevity() -> None:
    line_count = len(README_PATH.read_text(encoding="utf-8").splitlines())
    assert line_count <= 450, f"README has {line_count} lines (max 450)"


def test_public_architecture_sync() -> None:
    text = PUBLIC_ARCHITECTURE_PATH.read_text(encoding="utf-8")
    assert "PUBLIC-DOCS-COMMERCIALIZATION-5" in text
    assert "real product screenshots remain deferred" in text.lower()
