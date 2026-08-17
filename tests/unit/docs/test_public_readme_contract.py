# © Artur Czarnecki. All rights reserved.

"""Root README product-first landing contract tests."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
HERO_LIGHT_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-hero-light.svg"
HERO_DARK_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-hero-dark.svg"
LKW_LIGHT_PATH = (
    REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "assets"
    / "lkw-grounded-result-light.svg"
)
LKW_DARK_PATH = (
    REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "assets"
    / "lkw-grounded-result-dark.svg"
)

_SECTION_HEADINGS_ORDER = (
    "## Choose your path",
    "## Local Knowledge Workspace (LKW)",
    "## Try LKW",
    "## Why this matters",
    "## Responsibility model",
    "## AI execution should not be a black box",
    "## What exists today",
    "## Platform capabilities and directions",
    "## License and collaboration",
)

_REMOVED_STANDALONE_CAPABILITY_HEADINGS = (
    "## Token Optimization",
    "## Multiplayer AI",
    "## Platform Extensibility",
    "## Agent Marketplace — future ecosystem concept",
)

_REQUIRED_PUBLIC_LINKS = (
    "applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md",
    "docs/project/architecture/GOVERNED_EXECUTION.md",
    "docs/project/capabilities/token_optimization/README.md",
    "PROOFS.md",
    "docs/project/community/PUBLIC_DOCUMENTATION_MAP.md",
    "WHY_INTERGRAX.md",
    "ARCHITECTURE_OVERVIEW.md",
    "LKW_PRODUCT_TOUR.md",
    "applications/local_workspace_application/docs/product/QUICKSTART.md",
    "PARTNERS.md",
    "LICENSE",
)

_COMPATIBILITY_ANCHORS = (
    "start-here",
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
    """Product-first first screen: Intergrax/LKW purpose and reusable foundations."""
    assert "Local Knowledge Workspace" in readme_text
    normalized = re.sub(r"[*_`]", "", readme_text).lower()
    assert (
        "primary daily-use conversational interface" in normalized
        or "try lkw" in normalized
    )
    assert "reusable" in normalized
    assert "policy" in normalized
    assert "evidence" in normalized
    assert "Intergrax helps teams build" in readme_text


def test_section_order(readme_text: str) -> None:
    positions = [readme_text.index(heading) for heading in _SECTION_HEADINGS_ORDER]
    assert positions == sorted(positions), "README section headings are out of required order"


def test_platform_capabilities_table_contract(readme_text: str) -> None:
    assert "## Platform capabilities and directions" in readme_text
    for capability in (
        "Governed Execution",
        "Observability & Auditability",
        "Token Optimization",
        "Multiplayer AI",
        "Platform Extensibility",
        "Agent Marketplace",
    ):
        assert capability in readme_text, f"Missing platform capability row: {capability}"
    for heading in _REMOVED_STANDALONE_CAPABILITY_HEADINGS:
        assert heading not in readme_text, f"Duplicated standalone section returned: {heading}"


def test_platform_capability_claim_boundaries(readme_text: str) -> None:
    normalized = re.sub(r"[*_`]", "", readme_text).lower()
    for phrase in (
        "implemented core — coverage / qualification ongoing",
        "complete platform-wide governance and production qualification not established",
        "implemented core + bounded proof",
        "universal every-path production observability not claimed",
        "partial — bounded",
        "universal savings",
        "production-proven savings",
        "architecture / roadmap stage",
        "runtime proof not yet established",
        "canonical architecture frozen",
        "complete third-party install-to-runtime e2e proof not yet established",
        "future product — not shipped today",
    ):
        assert phrase in normalized, f"Missing platform capability boundary: {phrase}"


def test_required_public_links(readme_text: str) -> None:
    for link in _REQUIRED_PUBLIC_LINKS:
        assert link in readme_text, f"Missing required public link: {link}"


def test_hero_contract(readme_text: str) -> None:
    assert "applications/local_workspace_application/docs/assets/lkw-grounded-result-light.svg" in readme_text
    assert "applications/local_workspace_application/docs/assets/lkw-grounded-result-dark.svg" in readme_text
    assert readme_text.count("<picture>") == 1
    assert 'alt="LKW quickstart flow' in readme_text
    assert "docs/project/assets/public/intergrax-hero-light.svg" not in readme_text
    assert "docs/project/assets/public/intergrax-hero-dark.svg" not in readme_text
    assert HERO_LIGHT_PATH.is_file(), "Hero light SVG is missing"
    assert HERO_DARK_PATH.is_file(), "Hero dark SVG is missing"
    assert LKW_LIGHT_PATH.is_file(), "LKW light SVG is missing"
    assert LKW_DARK_PATH.is_file(), "LKW dark SVG is missing"


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


@pytest.mark.parametrize(
    "svg_path",
    [HERO_LIGHT_PATH, HERO_DARK_PATH, LKW_LIGHT_PATH, LKW_DARK_PATH],
)
def test_svg_safety(svg_path: Path) -> None:
    root = _parse_svg(svg_path)
    violations = _collect_svg_violations(root)
    assert not violations, f"{svg_path.name}: {violations}"


def test_visual_contract(readme_text: str) -> None:
    """The concrete LKW proof visual is the only README visual."""
    blocks = _MERMAID_FENCE.findall(readme_text)
    assert not blocks, "README should route conceptual architecture to its owner"
    assert readme_text.count("<picture>") == 1


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


def test_governed_execution_claim_boundary(readme_text: str) -> None:
    """G1B closeout: implemented core exists; coverage/qualification remain open."""
    assert "Governed Execution" in readme_text
    assert "IMPLEMENTED CORE — coverage / qualification ongoing" in readme_text
    normalized = re.sub(r"[*_`]", "", readme_text)
    assert (
        "complete platform-wide governance and production qualification not established"
        in normalized
    )
    lower = readme_text.lower()
    assert "production ready" not in lower
    assert "fully implemented" not in lower


def test_token_optimization_claim_boundary(readme_text: str) -> None:
    for phrase in (
        "PARTIAL",
        "bounded offline proof",
        "universal savings",
        "production-proven savings",
    ):
        assert phrase in readme_text, f"Missing Token Optimization boundary: {phrase}"

    lower = readme_text.lower()
    assert "not established" in lower or "not claimed" in lower or "not complete" in lower
    assert "universal token-savings claim" in lower or "universal savings" in lower

    for phrase in _FORBIDDEN_SAVINGS_PHRASES:
        if phrase in lower:
            idx = lower.index(phrase)
            context = lower[max(0, idx - 160) : idx + len(phrase) + 40]
            assert (
                "not claimed" in context
                or "not complete" in context
                or "not established" in context
                or "not currently" in context
                or "incomplete" in context
            ), f"Forbidden phrase {phrase!r} used positively"

    assert not _FORBIDDEN_SAVINGS_PATTERN.search(readme_text)
    assert not _PERCENT_PATTERN.search(readme_text), "Numeric savings percentage found in README"


def test_lkw_proof_boundary(readme_text: str) -> None:
    """PX-12 claim boundary: indexed Hybrid Ask proven; mixed indexed+live still incomplete."""
    lower = readme_text.lower()
    assert "hybrid ask" in lower
    assert "indexed" in lower
    assert "authorized live" in lower or "live evidence" in lower
    assert (
        "not complete" in lower
        or "not yet proven" in lower
        or "incomplete" in lower
    )
    assert "real application code path" in lower or "production code path" in lower


def test_proof_dashboard_route(readme_text: str) -> None:
    """Proof-path CLI details live outside the product-first README; dashboard must remain linked."""
    assert "PROOFS.md" in readme_text
    assert "applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md" in readme_text


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
    assert "## Try LKW" in readme_text
    assert "run-lkw-product-quickstart" in readme_text
    for anchor in _COMPATIBILITY_ANCHORS:
        assert f'id="{anchor}"' in readme_text, f"Missing compatibility anchor: {anchor}"


def test_brevity() -> None:
    line_count = len(README_PATH.read_text(encoding="utf-8").splitlines())
    assert line_count <= 450, f"README has {line_count} lines (max 450)"


def test_public_architecture_sync() -> None:
    text = PUBLIC_ARCHITECTURE_PATH.read_text(encoding="utf-8")
    assert "PX-2" in text
    assert "product-first" in text.lower()
    assert "real product screenshots remain deferred" in text.lower()
