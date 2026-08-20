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
README_STRATEGIC_ASSETS_DIR = REPO_ROOT / "docs" / "project" / "assets" / "public" / "readme"
ECOSYSTEM_HERO_LIGHT_PATH = README_STRATEGIC_ASSETS_DIR / "intergrax-ecosystem-hero-light.png"
ECOSYSTEM_HERO_DARK_PATH = README_STRATEGIC_ASSETS_DIR / "intergrax-ecosystem-hero-dark.png"
PLATFORM_MAP_LIGHT_PATH = README_STRATEGIC_ASSETS_DIR / "intergrax-platform-map-light.png"
PLATFORM_MAP_DARK_PATH = README_STRATEGIC_ASSETS_DIR / "intergrax-platform-map-dark.png"
WHY_LIGHT_PATH = README_STRATEGIC_ASSETS_DIR / "intergrax-why-light.png"
WHY_DARK_PATH = README_STRATEGIC_ASSETS_DIR / "intergrax-why-dark.png"
GOVERNED_EXECUTION_LIGHT_PATH = (
    README_STRATEGIC_ASSETS_DIR / "intergrax-governed-execution-light.png"
)
GOVERNED_EXECUTION_DARK_PATH = (
    README_STRATEGIC_ASSETS_DIR / "intergrax-governed-execution-dark.png"
)
_STRATEGIC_PNG_PAIRS = (
    (ECOSYSTEM_HERO_LIGHT_PATH, ECOSYSTEM_HERO_DARK_PATH),
    (PLATFORM_MAP_LIGHT_PATH, PLATFORM_MAP_DARK_PATH),
    (WHY_LIGHT_PATH, WHY_DARK_PATH),
    (GOVERNED_EXECUTION_LIGHT_PATH, GOVERNED_EXECUTION_DARK_PATH),
)
HERO_LIGHT_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-hero-light.svg"
HERO_DARK_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-hero-dark.svg"
LKW_ASSETS_PREFIX = "applications/local_workspace_application/docs/assets/"
README_STRATEGIC_PREFIX = "docs/project/assets/public/readme/"
_README_VISUAL_OWNERSHIP_ROOTS = (
    REPO_ROOT / "docs" / "project" / "assets" / "public" / "readme",
    REPO_ROOT / "applications" / "local_workspace_application" / "docs" / "assets",
)
_MIN_README_PICTURES = 5
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
_PICTURE_BLOCK = re.compile(r"<picture>(.*?)</picture>", re.DOTALL | re.IGNORECASE)
_SOURCE_TAG = re.compile(r"<source\b([^>]*)>", re.IGNORECASE)
_IMG_TAG = re.compile(r"<img\b([^>]*)>", re.IGNORECASE)
_ATTR_VALUE = re.compile(r'(\w+)="([^"]*)"')


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


def _normalize_light_dark_stem(filename: str) -> str | None:
    for suffix in ("-light.svg", "-dark.svg", "-light.png", "-dark.png"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def _validate_light_dark_pair(light_path: Path, dark_path: Path) -> list[str]:
    violations: list[str] = []
    if not light_path.is_file():
        violations.append(f"missing light variant: {light_path.name}")
    if not dark_path.is_file():
        violations.append(f"missing dark variant: {dark_path.name}")
    if light_path.is_file() and dark_path.is_file():
        light_stem = _normalize_light_dark_stem(light_path.name)
        dark_stem = _normalize_light_dark_stem(dark_path.name)
        if light_stem is None or dark_stem is None:
            violations.append("pair must use *-light.* / *-dark.* naming")
        elif light_stem != dark_stem:
            violations.append("light/dark stem mismatch")
        elif light_path.suffix != dark_path.suffix:
            violations.append("light/dark extension mismatch")
    return violations


def _light_dark_pair_paths(stem: str, directory: Path, extension: str) -> tuple[Path, Path]:
    return (
        directory / f"{stem}-light{extension}",
        directory / f"{stem}-dark{extension}",
    )


def _normalize_visual_path(path_str: str) -> str:
    return path_str.replace("\\", "/")


def _resolve_approved_readme_visual(path_str: str) -> Path:
    """Resolve a README visual path and ensure it stays within approved ownership roots."""
    if Path(path_str).is_absolute():
        raise ValueError(f"README visual must be repo-relative: {path_str}")

    repo_root = REPO_ROOT.resolve()
    resolved = (repo_root / path_str).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"README visual resolves outside repository: {path_str}") from exc

    for ownership_root in _README_VISUAL_OWNERSHIP_ROOTS:
        try:
            resolved.relative_to(ownership_root.resolve())
            return resolved
        except ValueError:
            continue

    raise ValueError(
        "README visual outside approved ownership roots: "
        f"{path_str!r} (allowed: {README_STRATEGIC_PREFIX!r}, {LKW_ASSETS_PREFIX!r})"
    )


def _tag_attributes(tag_body: str) -> dict[str, str]:
    return {match.group(1).lower(): match.group(2) for match in _ATTR_VALUE.finditer(tag_body)}


def _parse_picture_block(block: str) -> dict[str, str]:
    light_src = dark_src = img_src = img_alt = ""
    for source_match in _SOURCE_TAG.finditer(block):
        attrs = _tag_attributes(source_match.group(1))
        srcset = attrs.get("srcset", "").strip()
        media = attrs.get("media", "").lower()
        if "prefers-color-scheme: dark" in media:
            dark_src = srcset
        elif "prefers-color-scheme: light" in media:
            light_src = srcset
    for img_match in _IMG_TAG.finditer(block):
        attrs = _tag_attributes(img_match.group(1))
        img_src = attrs.get("src", "").strip()
        img_alt = attrs.get("alt", "").strip()
    return {
        "light": light_src,
        "dark": dark_src,
        "img_src": img_src,
        "img_alt": img_alt,
    }


def _extract_picture_blocks(readme_text: str) -> list[str]:
    return _PICTURE_BLOCK.findall(readme_text)


def test_ecosystem_hero_contract(readme_text: str) -> None:
    """Root README ecosystem hero appears before Choose your path with controlled PNG assets."""
    choose_idx = readme_text.index("## Choose your path")
    hero_light_ref = "docs/project/assets/public/readme/intergrax-ecosystem-hero-light.png"
    hero_dark_ref = "docs/project/assets/public/readme/intergrax-ecosystem-hero-dark.png"
    assert "intergrax-ecosystem-hero-light.svg" not in readme_text
    assert "intergrax-ecosystem-hero-dark.svg" not in readme_text
    assert readme_text.index(hero_light_ref) < choose_idx
    assert readme_text.index(hero_dark_ref) < choose_idx
    assert readme_text.index("<picture>") < choose_idx
    assert ECOSYSTEM_HERO_LIGHT_PATH.is_file(), "Ecosystem hero light PNG is missing"
    assert ECOSYSTEM_HERO_DARK_PATH.is_file(), "Ecosystem hero dark PNG is missing"
    pair_violations = _validate_light_dark_pair(ECOSYSTEM_HERO_LIGHT_PATH, ECOSYSTEM_HERO_DARK_PATH)
    assert not pair_violations, f"Ecosystem hero light/dark pair: {pair_violations}"
    assert 'alt="Specialized AI products share the Intergrax governed foundation' in readme_text


def test_platform_map_visual_contract(readme_text: str) -> None:
    section_idx = readme_text.index("## Explore the Intergrax Platform")
    table_marker = "| Platform area | What it provides | Explore |"
    table_idx = readme_text.index(table_marker, section_idx)
    light_ref = "docs/project/assets/public/readme/intergrax-platform-map-light.png"
    dark_ref = "docs/project/assets/public/readme/intergrax-platform-map-dark.png"
    assert section_idx < readme_text.index(light_ref) < table_idx
    assert section_idx < readme_text.index(dark_ref) < table_idx
    assert PLATFORM_MAP_LIGHT_PATH.is_file()
    assert PLATFORM_MAP_DARK_PATH.is_file()
    pair_violations = _validate_light_dark_pair(PLATFORM_MAP_LIGHT_PATH, PLATFORM_MAP_DARK_PATH)
    assert not pair_violations, f"Platform map light/dark pair: {pair_violations}"
    assert "Intergrax platform architecture map showing execution core" in readme_text


def test_why_visual_contract(readme_text: str) -> None:
    section_idx = readme_text.index("## Why this matters")
    next_section_idx = readme_text.index("## Responsibility model", section_idx)
    light_ref = "docs/project/assets/public/readme/intergrax-why-light.png"
    dark_ref = "docs/project/assets/public/readme/intergrax-why-dark.png"
    assert section_idx < readme_text.index(light_ref) < next_section_idx
    assert section_idx < readme_text.index(dark_ref) < next_section_idx
    assert WHY_LIGHT_PATH.is_file()
    assert WHY_DARK_PATH.is_file()
    pair_violations = _validate_light_dark_pair(WHY_LIGHT_PATH, WHY_DARK_PATH)
    assert not pair_violations, f"Why Intergrax light/dark pair: {pair_violations}"
    assert "rebuilding duplicated AI foundations per product" in readme_text


def test_governed_execution_visual_contract(readme_text: str) -> None:
    section_idx = readme_text.index("## AI execution should not be a black box")
    light_ref = "docs/project/assets/public/readme/intergrax-governed-execution-light.png"
    dark_ref = "docs/project/assets/public/readme/intergrax-governed-execution-dark.png"
    assert section_idx < readme_text.index(light_ref)
    assert section_idx < readme_text.index(dark_ref)
    assert GOVERNED_EXECUTION_LIGHT_PATH.is_file()
    assert GOVERNED_EXECUTION_DARK_PATH.is_file()
    pair_violations = _validate_light_dark_pair(
        GOVERNED_EXECUTION_LIGHT_PATH,
        GOVERNED_EXECUTION_DARK_PATH,
    )
    assert not pair_violations, f"Governed execution light/dark pair: {pair_violations}"
    assert "governed agentic execution loop" in readme_text
    assert "request → context → agent / plan / decision → policy / approval" not in readme_text


def test_lkw_visual_contract(readme_text: str) -> None:
    assert "applications/local_workspace_application/docs/assets/lkw-grounded-result-light.svg" in readme_text
    assert "applications/local_workspace_application/docs/assets/lkw-grounded-result-dark.svg" in readme_text
    assert 'alt="LKW quickstart flow' in readme_text
    assert LKW_LIGHT_PATH.is_file(), "LKW light SVG is missing"
    assert LKW_DARK_PATH.is_file(), "LKW dark SVG is missing"
    pair_violations = _validate_light_dark_pair(LKW_LIGHT_PATH, LKW_DARK_PATH)
    assert not pair_violations, f"LKW light/dark pair: {pair_violations}"


def test_readme_controlled_multi_visual_contract(readme_text: str) -> None:
    """Root README may host multiple strategic visuals under controlled ownership."""
    picture_blocks = _extract_picture_blocks(readme_text)
    assert len(picture_blocks) >= _MIN_README_PICTURES, (
        f"README must contain at least {_MIN_README_PICTURES} controlled <picture> block(s); "
        f"found {len(picture_blocks)}"
    )

    referenced_svgs: set[Path] = set()
    referenced_pngs: set[Path] = set()
    for block in picture_blocks:
        parsed = _parse_picture_block(block)
        assert parsed["light"], "picture block missing light <source>"
        assert parsed["dark"], "picture block missing dark <source>"
        assert parsed["img_src"], "picture block missing <img src>"
        assert parsed["img_alt"], "picture block missing non-empty <img alt>"

        light_path = _normalize_visual_path(parsed["light"])
        img_path = _normalize_visual_path(parsed["img_src"])
        assert light_path == img_path, (
            "picture <img src> fallback must match light <source srcset> exactly: "
            f"light={parsed['light']!r}, img={parsed['img_src']!r}"
        )

        for path_str in (parsed["light"], parsed["dark"], parsed["img_src"]):
            normalized = _normalize_visual_path(path_str)
            assert normalized.endswith((".svg", ".png")), (
                f"README visual must be local SVG or PNG: {path_str}"
            )
            resolved = _resolve_approved_readme_visual(normalized)
            if normalized.endswith(".svg"):
                referenced_svgs.add(resolved)
            else:
                referenced_pngs.add(resolved)

        light_stem = _normalize_light_dark_stem(Path(parsed["light"]).name)
        dark_stem = _normalize_light_dark_stem(Path(parsed["dark"]).name)
        assert light_stem and dark_stem and light_stem == dark_stem, (
            "picture block light/dark srcset must share the same *-light/*-dark stem"
        )

    for png_path in referenced_pngs:
        assert png_path.is_file(), f"Referenced README PNG is missing: {png_path}"

    for svg_path in referenced_svgs:
        assert svg_path.is_file(), f"Referenced README SVG is missing: {svg_path}"
        violations = _collect_svg_violations(_parse_svg(svg_path))
        assert not violations, f"{svg_path.name}: {violations}"


def test_strategic_light_dark_pair_convention() -> None:
    """Reusable pair validation for strategic PNG families and module-owned SVG families."""
    for light_path, dark_path in _STRATEGIC_PNG_PAIRS:
        violations = _validate_light_dark_pair(light_path, dark_path)
        assert not violations, f"{light_path.parent.name}/{light_path.stem}: {violations}"
        assert light_path.suffix == ".png"
        assert dark_path.suffix == ".png"

    for light_path, dark_path in ((LKW_LIGHT_PATH, LKW_DARK_PATH),):
        violations = _validate_light_dark_pair(light_path, dark_path)
        assert not violations, f"{light_path.parent.name}/{light_path.stem}: {violations}"


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
    [
        LKW_LIGHT_PATH,
        LKW_DARK_PATH,
    ],
)
def test_svg_safety(svg_path: Path) -> None:
    root = _parse_svg(svg_path)
    violations = _collect_svg_violations(root)
    assert not violations, f"{svg_path.name}: {violations}"


def test_visual_contract(readme_text: str) -> None:
    """Strategic root README visuals use controlled PNG pairs; Mermaid is not a substitute."""
    blocks = _MERMAID_FENCE.findall(readme_text)
    assert not blocks, "README should route conceptual architecture to its owner"


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
    assert "controlled multi-visual" in text.lower()
    assert "docs/project/assets/public/readme/" in text
