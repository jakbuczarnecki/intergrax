# © Artur Czarnecki. All rights reserved.

"""Root README product-first landing contract tests."""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
import zlib
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
THREE_ENTRY_POINTS_LIGHT_PATH = (
    README_STRATEGIC_ASSETS_DIR / "intergrax-three-entry-points-light.png"
)
THREE_ENTRY_POINTS_DARK_PATH = (
    README_STRATEGIC_ASSETS_DIR / "intergrax-three-entry-points-dark.png"
)
SCENARIOS_OVERVIEW_LIGHT_PATH = (
    README_STRATEGIC_ASSETS_DIR / "intergrax-scenarios-overview-light.png"
)
SCENARIOS_OVERVIEW_DARK_PATH = (
    README_STRATEGIC_ASSETS_DIR / "intergrax-scenarios-overview-dark.png"
)
SCENARIO_INCIDENT_LIGHT_PATH = (
    README_STRATEGIC_ASSETS_DIR / "scenario-ai-incident-investigation-light.png"
)
SCENARIO_INCIDENT_DARK_PATH = (
    README_STRATEGIC_ASSETS_DIR / "scenario-ai-incident-investigation-dark.png"
)
_FULL_SIZE_LINK_LABEL = "View full-size diagram"
_STRATEGIC_FULL_SIZE_LINKS = (
    (
        "## Explore the Intergrax Platform",
        "| Platform area | What it provides | Explore |",
        "docs/project/assets/public/readme/fullsize/intergrax-platform-map.md",
        "docs/project/assets/public/readme/intergrax-platform-map-light.png",
    ),
    (
        "## Why this matters",
        "---",
        "docs/project/assets/public/readme/fullsize/intergrax-why.md",
        "docs/project/assets/public/readme/intergrax-why-light.png",
    ),
    (
        "## AI execution should not be a black box",
        "A governed run can leave correlated runtime events",
        "docs/project/assets/public/readme/fullsize/intergrax-governed-execution.md",
        "docs/project/assets/public/readme/intergrax-governed-execution-light.png",
    ),
    (
        "## Explore Intergrax",
        "**[Explore Proof Library]",
        "docs/project/assets/public/readme/fullsize/intergrax-three-entry-points.md",
        "docs/project/assets/public/readme/intergrax-three-entry-points-light.png",
    ),
    (
        "## Real problems. Executable evidence.",
        "**[Explore Proof Library]",
        "docs/project/assets/public/readme/fullsize/intergrax-scenarios-overview.md",
        "docs/project/assets/public/readme/intergrax-scenarios-overview-light.png",
    ),
)
_STRATEGIC_PNG_PAIRS = (
    (ECOSYSTEM_HERO_LIGHT_PATH, ECOSYSTEM_HERO_DARK_PATH),
    (PLATFORM_MAP_LIGHT_PATH, PLATFORM_MAP_DARK_PATH),
    (WHY_LIGHT_PATH, WHY_DARK_PATH),
    (GOVERNED_EXECUTION_LIGHT_PATH, GOVERNED_EXECUTION_DARK_PATH),
    (THREE_ENTRY_POINTS_LIGHT_PATH, THREE_ENTRY_POINTS_DARK_PATH),
    (SCENARIOS_OVERVIEW_LIGHT_PATH, SCENARIOS_OVERVIEW_DARK_PATH),
    (SCENARIO_INCIDENT_LIGHT_PATH, SCENARIO_INCIDENT_DARK_PATH),
)
HERO_LIGHT_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-hero-light.svg"
HERO_DARK_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-hero-dark.svg"
LKW_ASSETS_PREFIX = "applications/local_workspace_application/docs/assets/"
README_STRATEGIC_PREFIX = "docs/project/assets/public/readme/"
_README_VISUAL_OWNERSHIP_ROOTS = (
    REPO_ROOT / "docs" / "project" / "assets" / "public" / "readme",
    REPO_ROOT / "applications" / "local_workspace_application" / "docs" / "assets",
)
_MIN_README_PICTURES = 8
LKW_LIGHT_PATH = (
    REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "assets"
    / "lkw-governed-evidence-gate-light.png"
)
LKW_DARK_PATH = (
    REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "assets"
    / "lkw-governed-evidence-gate-dark.png"
)

_SECTION_HEADINGS_ORDER = (
    "## Choose your path",
    "## Why this matters",
    "## Explore Intergrax",
    "## Real problems. Executable evidence.",
    "## Local Knowledge Workspace (LKW)",
    "## Try LKW",
    "## Explore the Intergrax Platform",
    "## AI execution should not be a black box",
    "## Responsibility model",
    "## Platform capabilities",
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
    "applications/local_workspace_application/docs/proof/GOVERNED_HYBRID_KNOWLEDGE_PROOF.md",
    "docs/project/architecture/GOVERNED_EXECUTION.md",
    "docs/project/capabilities/token_optimization/README.md",
    "docs/project/proofs/PROOF_LIBRARY.md",
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

_STALE_INCIDENT_SCENARIO_LOGISTICS_MARKERS = (
    "warehouse overload",
    "warehouse",
    "parcel",
    "sorter",
    "logistics operator",
    "heavy parcel",
)

_FORBIDDEN_SAVINGS_PATTERN = re.compile(r"reduces token usage by\s*\d+\s*%", re.IGNORECASE)
_PERCENT_PATTERN = re.compile(r"\d+\s*%")

_MERMAID_FENCE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
_PICTURE_BLOCK = re.compile(r"<picture>(.*?)</picture>", re.DOTALL | re.IGNORECASE)
_SOURCE_TAG = re.compile(r"<source\b([^>]*)>", re.IGNORECASE)
_IMG_TAG = re.compile(r"<img\b([^>]*)>", re.IGNORECASE)
_ATTR_VALUE = re.compile(r'(\w+)="([^"]*)"')
_LKW_GOVERNED_HERO_LIGHT = (
    "applications/local_workspace_application/docs/assets/lkw-governed-evidence-gate-light.png"
)
_ECOSYSTEM_HERO_LIGHT = "docs/project/assets/public/readme/intergrax-ecosystem-hero-light.png"
_QUICKSTART_SCRIPT_PATTERN = re.compile(
    r"run-lkw-product-quickstart-(windows|linux|macos)\.(bat|sh)"
)


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


def test_opening_architecture_differentiator(readme_text: str) -> None:
    """Opening states policy/evidence/admissibility differentiator without brittle prose."""
    opening = readme_text[: readme_text.index("## Choose your path")]
    normalized = re.sub(r"[*_`]", "", opening).lower()
    assert "policy" in normalized
    assert "evidence" in normalized
    assert (
        "allowed to proceed" in normalized
        or "admissib" in normalized
        or "determine whether" in normalized
    )
    assert "instead of leaving" in normalized or "entirely to the model" in normalized
    forbidden = (
        "guaranteed correctness",
        "hallucination prevention",
        "production ready",
        "complete governance everywhere",
        "complete hybrid ask",
    )
    for phrase in forbidden:
        assert phrase not in normalized, f"Forbidden opening claim: {phrase}"


def test_section_order(readme_text: str) -> None:
    positions = [readme_text.index(heading) for heading in _SECTION_HEADINGS_ORDER]
    assert positions == sorted(positions), "README section headings are out of required order"
    why_idx = readme_text.index("## Why this matters")
    explore_idx = readme_text.index("## Explore Intergrax")
    scenario_idx = readme_text.index("## Real problems. Executable evidence.")
    lkw_idx = readme_text.index("## Local Knowledge Workspace (LKW)")
    platform_idx = readme_text.index("## Explore the Intergrax Platform")
    assert why_idx < explore_idx < scenario_idx < lkw_idx < platform_idx, (
        "Required flow: Why → Explore Intergrax → Scenario Proof Library → LKW → Platform"
    )
    ai_execution_idx = readme_text.index("## AI execution should not be a black box")
    responsibility_idx = readme_text.index("## Responsibility model")
    assert ai_execution_idx < responsibility_idx, (
        "Platform differentiation (AI execution) must precede Responsibility model"
    )


def test_featured_incident_scenario_no_stale_logistics_framing(readme_text: str) -> None:
    """Featured AI Incident Investigation prose must not regress to logistics fixture wording."""
    featured_section = _section_slice(
        readme_text,
        "### Featured scenario in development",
        "---",
    )
    normalized = re.sub(r"[*_`]", "", featured_section).lower()
    for marker in _STALE_INCIDENT_SCENARIO_LOGISTICS_MARKERS:
        assert marker not in normalized, (
            f"Stale logistics framing in featured incident scenario: {marker!r}"
        )
    assert "workload overload" in normalized or "production signals" in normalized


def test_scenario_public_positioning(readme_text: str) -> None:
    """Scenario Proof Library is first-class: before LKW, distinct from products."""
    scenario_idx = readme_text.index("## Real problems. Executable evidence.")
    lkw_idx = readme_text.index("## Local Knowledge Workspace (LKW)")
    assert scenario_idx < lkw_idx, "Scenario section must precede main LKW section"
    scenario_section = _section_slice(
        readme_text,
        "## Real problems. Executable evidence.",
        "## Local Knowledge Workspace (LKW)",
    )
    normalized = re.sub(r"[*_`]", "", scenario_section).lower()
    assert "scenario proof library" in normalized
    assert "not a marketing demo" in normalized or "not a marketing" in normalized
    assert "featured scenario in development" in normalized
    assert "full-1" in normalized
    assert "implemented" in normalized
    assert "executable" in normalized
    assert (
        "public proof" in normalized
        or "not accepted" in normalized
        or "remain pending" in normalized
    )
    assert "no executable proof yet" not in normalized
    assert "no executable evidence or report yet" not in normalized
    assert "ai_incident_investigation" in scenario_section
    explore_section = _section_slice(
        readme_text,
        "## Explore Intergrax",
        "## Real problems. Executable evidence.",
    )
    explore_normalized = re.sub(r"[*_`]", "", explore_section).lower()
    for path_label in ("scenario proofs", "products", "platform"):
        assert path_label in explore_normalized, f"Missing three-entry path: {path_label}"
    assert readme_text.count("## Real problems. Executable evidence.") == 1, (
        "Duplicate scenario narrative sections are forbidden"
    )


def test_scenario_visual_contract(readme_text: str) -> None:
    """Three new scenario entry-point visuals use theme-aware pairs and preview routes."""
    for light_path, dark_path in (
        (THREE_ENTRY_POINTS_LIGHT_PATH, THREE_ENTRY_POINTS_DARK_PATH),
        (SCENARIOS_OVERVIEW_LIGHT_PATH, SCENARIOS_OVERVIEW_DARK_PATH),
        (SCENARIO_INCIDENT_LIGHT_PATH, SCENARIO_INCIDENT_DARK_PATH),
    ):
        assert light_path.is_file(), f"Missing scenario visual: {light_path.name}"
        assert dark_path.is_file(), f"Missing scenario visual: {dark_path.name}"
        violations = _validate_light_dark_pair(light_path, dark_path)
        assert not violations, f"{light_path.stem}: {violations}"
    for preview_ref in (
        "docs/project/assets/public/readme/fullsize/intergrax-three-entry-points.md",
        "docs/project/assets/public/readme/fullsize/intergrax-scenarios-overview.md",
        "docs/project/assets/public/readme/fullsize/scenario-ai-incident-investigation.md",
    ):
        assert (REPO_ROOT / preview_ref).is_file(), f"Missing fullsize preview: {preview_ref}"


def test_top_cta_proof_library(readme_text: str) -> None:
    """Top CTA exposes the public Proof Library gateway."""
    top_block = readme_text[: readme_text.index("## Choose your path")]
    assert "Explore Proof Library" in top_block
    assert "docs/project/proofs/PROOF_LIBRARY.md" in top_block


def test_governed_evidence_first_contact_projection(readme_text: str) -> None:
    """Governed Evidence flagship wording matches controlled-live canonical semantics."""
    governed_section = _section_slice(
        readme_text,
        "#### B. Governed Evidence Decision Proof",
        "**Other bounded paths:**",
    )
    normalized = re.sub(r"[*_`]", "", governed_section).lower()
    assert "controlled live" in normalized
    assert (
        "not external saas" in normalized
        or "not external saas validation" in normalized
    )
    assert "four live organizational sources" not in normalized
    assert "external live-provider access" in re.sub(r"[*_`]", "", readme_text).lower()


def test_lkw_routes_governed_evidence_proof(readme_text: str) -> None:
    routes_section = _section_slice(readme_text, "### LKW routes", "**Core Platform Proof**")
    assert "Governed Evidence Decision Proof" in routes_section
    assert (
        "applications/local_workspace_application/docs/proof/GOVERNED_HYBRID_KNOWLEDGE_PROOF.md"
        in routes_section
    )
    product_tour_idx = routes_section.index("Product Tour")
    quick_start_idx = routes_section.index("Quick Start")
    governed_idx = routes_section.index("Governed Evidence Decision Proof")
    core_idx = routes_section.index("Core Platform Proof")
    assert product_tour_idx < quick_start_idx < governed_idx < core_idx


def test_lkw_product_positioning(readme_text: str) -> None:
    """LKW section is product-first; Primary Product Proof terminology removed."""
    lkw_section = _section_slice(readme_text, "## Local Knowledge Workspace (LKW)", "## Try LKW")
    normalized = re.sub(r"[*_`]", "", lkw_section).lower()
    assert "primary product proof" not in normalized
    assert "reference product application" not in normalized
    assert "governed ai knowledge workspace" in normalized or "approved organizational knowledge" in normalized
    assert "backend product alpha / mvp" in normalized
    assert "accepted bounded proof paths" in normalized


def test_try_lkw_compressed(readme_text: str) -> None:
    """Gateway README delegates OS commands to Quick Start doc."""
    try_section = _section_slice(readme_text, "## Try LKW", "### LKW routes")
    assert "AURORA-17" in try_section
    assert "applications/local_workspace_application/docs/product/QUICKSTART.md" in try_section
    assert _QUICKSTART_SCRIPT_PATTERN.search(try_section) is None, (
        "Root README must not duplicate per-OS quick start command blocks"
    )
    assert try_section.count("```") == 0, "Try LKW must not include fenced command blocks"


def test_what_exists_today_removed(readme_text: str) -> None:
    assert "## What exists today" not in readme_text


def test_docs_taxonomy_compressed(readme_text: str) -> None:
    assert "### How documentation is organized" not in readme_text
    assert "docs/project/architecture/<DOMAIN>.md" not in readme_text
    assert "docs/project/capabilities/architecture/<FEATURE>.md" not in readme_text
    assert "24 architecture" not in readme_text.lower()
    assert "architecture ↔ plan pairs" not in readme_text
    assert "Full technical domain index" in readme_text
    for route in (
        "Builder Quick Start",
        "Architecture Overview",
        "PROOFS",
        "Public Documentation Map",
        "Technical Documentation Map",
    ):
        assert route in readme_text, f"Missing compact docs route: {route}"


def test_platform_capabilities_table_contract(readme_text: str) -> None:
    assert "## Platform capabilities" in readme_text
    assert "### Future strategic directions" in readme_text
    current_section = _section_slice(
        readme_text,
        "## Platform capabilities",
        "### Future strategic directions",
    )
    future_section = _section_slice(
        readme_text,
        "### Future strategic directions",
        "## License and collaboration",
    )
    for capability in (
        "Governed Execution",
        "Observability & Auditability",
        "Token Optimization",
    ):
        assert capability in current_section, f"Missing current capability row: {capability}"
    for direction in (
        "Multiplayer AI",
        "Platform Extensibility",
        "Agent Marketplace",
    ):
        assert direction in future_section, f"Missing future direction row: {direction}"
        assert direction not in current_section, f"Future direction leaked into current table: {direction}"
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


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not a PNG: {path}")
        length = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR":
            raise ValueError(f"missing IHDR: {path}")
        data = handle.read(length)
    width, height = struct.unpack(">II", data[:8])
    return width, height


def _png_row_luma_avg(path: Path, row: int, sample_step: int = 16) -> int:
    with path.open("rb") as handle:
        handle.read(8)
        chunks: list[tuple[bytes, bytes]] = []
        while True:
            length = struct.unpack(">I", handle.read(4))[0]
            chunk_type = handle.read(4)
            data = handle.read(length)
            handle.read(4)
            chunks.append((chunk_type, data))
            if chunk_type == b"IEND":
                break
    ihdr = next(data for chunk_type, data in chunks if chunk_type == b"IHDR")
    width, height, bit_depth, color_type, *_ = struct.unpack(">IIBBBBB", ihdr)
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError(f"unsupported PNG color layout: {path}")
    bytes_per_pixel = 3 if color_type == 2 else 4
    idat = b"".join(data for chunk_type, data in chunks if chunk_type == b"IDAT")
    raw = zlib.decompress(idat)
    stride = width * bytes_per_pixel + 1
    row = max(0, min(height - 1, row))
    row_start = row * stride + 1
    total = 0
    samples = 0
    for x in range(0, width, sample_step):
        index = row_start + x * bytes_per_pixel
        total += (raw[index] + raw[index + 1] + raw[index + 2]) // 3
        samples += 1
    return total // samples


def _png_has_vertical_theme_composite(path: Path) -> bool:
    """Detect light+dark variants stacked vertically in one PNG."""
    _, height = _png_dimensions(path)
    top = _png_row_luma_avg(path, height // 16)
    mid = _png_row_luma_avg(path, height // 2)
    bottom = _png_row_luma_avg(path, height - height // 16)
    light_top_dark_bottom = top > 280 and mid < 160 and top - mid > 200
    dark_top_light_bottom = top < 160 and mid > 220 and bottom - top > 200
    return light_top_dark_bottom or dark_top_light_bottom


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
        else:
            light_w, light_h = _png_dimensions(light_path)
            dark_w, dark_h = _png_dimensions(dark_path)
            if light_w != dark_w or light_h != dark_h:
                violations.append("light/dark dimension mismatch")
            if _png_has_vertical_theme_composite(light_path):
                violations.append("light PNG looks like a vertical light+dark composite")
            if _png_has_vertical_theme_composite(dark_path):
                violations.append("dark PNG looks like a vertical light+dark composite")
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


def _section_slice(readme_text: str, start_marker: str, end_marker: str) -> str:
    start = readme_text.index(start_marker)
    end = readme_text.index(end_marker, start + len(start_marker))
    return readme_text[start:end]


def _full_size_link_markdown(preview_ref: str) -> str:
    return f"[{_FULL_SIZE_LINK_LABEL}]({preview_ref})"


def _assert_full_size_link_after_picture(section: str, preview_ref: str) -> None:
    link = _full_size_link_markdown(preview_ref)
    assert link in section, f"Missing full-size link to {preview_ref}"
    picture_end = section.index("</picture>")
    link_idx = section.index(link)
    assert picture_end < link_idx, (
        f"Full-size link must immediately follow <picture> block for {preview_ref}"
    )


def test_product_visual_order(readme_text: str) -> None:
    """Gateway visuals lead with ecosystem hero; scenario visuals precede LKW product proof."""
    ecosystem_idx = readme_text.index(_ECOSYSTEM_HERO_LIGHT)
    scenario_overview_idx = readme_text.index(
        "docs/project/assets/public/readme/intergrax-scenarios-overview-light.png"
    )
    lkw_idx = readme_text.index(_LKW_GOVERNED_HERO_LIGHT)
    why_light_idx = readme_text.index("docs/project/assets/public/readme/intergrax-why-light.png")
    explore_idx = readme_text.index("## Explore the Intergrax Platform")
    assert ecosystem_idx < scenario_overview_idx, "Ecosystem hero must appear before scenario overview"
    assert scenario_overview_idx < lkw_idx, "Scenario overview must appear before LKW governed hero"
    assert why_light_idx < explore_idx, "Why visual must appear before Explore the Intergrax Platform"
    ecosystem_picture_blocks = [
        block for block in _extract_picture_blocks(readme_text) if _ECOSYSTEM_HERO_LIGHT in block
    ]
    assert len(ecosystem_picture_blocks) == 1, "Ecosystem hero must appear exactly once"
    lkw_picture_blocks = [
        block for block in _extract_picture_blocks(readme_text) if _LKW_GOVERNED_HERO_LIGHT in block
    ]
    assert len(lkw_picture_blocks) == 1, "LKW governed hero must appear exactly once"


def test_ecosystem_hero_contract(readme_text: str) -> None:
    """Ecosystem hero is the first large strategic visual in the gateway block."""
    choose_idx = readme_text.index("## Choose your path")
    lkw_idx = readme_text.index("## Local Knowledge Workspace (LKW)")
    capabilities_idx = readme_text.index("## Platform capabilities")
    hero_light_ref = _ECOSYSTEM_HERO_LIGHT
    hero_dark_ref = "docs/project/assets/public/readme/intergrax-ecosystem-hero-dark.png"
    hero_light_idx = readme_text.index(hero_light_ref)
    hero_dark_idx = readme_text.index(hero_dark_ref)
    assert "intergrax-ecosystem-hero-light.svg" not in readme_text
    assert "intergrax-ecosystem-hero-dark.svg" not in readme_text
    assert hero_light_idx < choose_idx, "Ecosystem hero must appear before Choose your path"
    assert hero_light_idx < lkw_idx, "Ecosystem hero must appear before LKW section"
    assert hero_light_idx < capabilities_idx, "Ecosystem hero must appear before Platform capabilities"
    assert hero_dark_idx < choose_idx
    assert ECOSYSTEM_HERO_LIGHT_PATH.is_file(), "Ecosystem hero light PNG is missing"
    assert ECOSYSTEM_HERO_DARK_PATH.is_file(), "Ecosystem hero dark PNG is missing"
    pair_violations = _validate_light_dark_pair(ECOSYSTEM_HERO_LIGHT_PATH, ECOSYSTEM_HERO_DARK_PATH)
    assert not pair_violations, f"Ecosystem hero light/dark pair: {pair_violations}"
    assert 'alt="Specialized AI products share the Intergrax governed foundation' in readme_text
    future_idx = readme_text.index("### Future strategic directions")
    assert hero_light_idx < future_idx, "Ecosystem hero must not remain in future-directions tail"


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
    next_section_idx = readme_text.index("---", section_idx)
    light_ref = "docs/project/assets/public/readme/intergrax-why-light.png"
    dark_ref = "docs/project/assets/public/readme/intergrax-why-dark.png"
    assert section_idx < readme_text.index(light_ref) < next_section_idx
    assert section_idx < readme_text.index(dark_ref) < next_section_idx
    assert WHY_LIGHT_PATH.is_file()
    assert WHY_DARK_PATH.is_file()
    pair_violations = _validate_light_dark_pair(WHY_LIGHT_PATH, WHY_DARK_PATH)
    assert not pair_violations, f"Why Intergrax light/dark pair: {pair_violations}"
    assert "rebuilding duplicated AI foundations per product" in readme_text


def test_strategic_diagram_full_size_links(readme_text: str) -> None:
    """Detailed strategic diagrams expose a theme-aware full-size link within their section."""
    for start_marker, end_marker, preview_ref, light_png_ref in _STRATEGIC_FULL_SIZE_LINKS:
        section = _section_slice(readme_text, start_marker, end_marker)
        _assert_full_size_link_after_picture(section, preview_ref)
        assert (REPO_ROOT / preview_ref).is_file(), f"Missing full-size preview: {preview_ref}"
        assert (REPO_ROOT / light_png_ref).is_file(), f"Missing full-size PNG: {light_png_ref}"

    hero_section = readme_text[: readme_text.index("## Choose your path")]
    assert _FULL_SIZE_LINK_LABEL not in hero_section, "Hero must not include a full-size diagram link"

    lkw_section = _section_slice(readme_text, "## Local Knowledge Workspace (LKW)", "---")
    assert _FULL_SIZE_LINK_LABEL not in lkw_section, "LKW section must not include full-size diagram links"


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
    assert _LKW_GOVERNED_HERO_LIGHT in readme_text
    assert (
        "applications/local_workspace_application/docs/assets/lkw-governed-evidence-gate-dark.png"
        in readme_text
    )
    assert "lkw-grounded-result" not in readme_text
    assert 'alt="LKW advanced governed proof' in readme_text
    assert LKW_LIGHT_PATH.is_file(), "LKW governed evidence gate light PNG is missing"
    assert LKW_DARK_PATH.is_file(), "LKW governed evidence gate dark PNG is missing"
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
    assert "live_only" in lower


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
    assert "applications/local_workspace_application/docs/product/QUICKSTART.md" in readme_text
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
