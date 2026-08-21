# © Artur Czarnecki. All rights reserved.

"""Public documentation clickable physical visual contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
LKW_DOCS_PREFIX = "applications/local_workspace_application/docs/"
LKW_ASSET_PREFIX = "applications/local_workspace_application/docs/assets/"

_PUBLIC_SCOPES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "project" / "overview",
    REPO_ROOT / "docs" / "project" / "architecture",
    REPO_ROOT / "docs" / "project" / "capabilities" / "architecture",
    REPO_ROOT / "docs" / "project" / "community",
    REPO_ROOT / "docs" / "project" / "builders",
    REPO_ROOT / "docs" / "project" / "technical" / "guides",
)

# Reader-facing LKW docs linked from README, product tour, quickstart, and proof routes.
# Internal LKW architecture/maintainer docs (for example HYBRID_ASK_ARCHITECTURE.md) stay out of scope.
_READER_FACING_LKW_SCOPES = (
    REPO_ROOT / "README.md",
    REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "product"
    / "LKW_PRODUCT_TOUR.md",
    REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "product"
    / "QUICKSTART.md",
    REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "proof"
    / "GOVERNED_HYBRID_KNOWLEDGE_PROOF.md",
    REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "proof"
    / "LKW_PLATFORM_PROOF.md",
)

_PICTURE_BLOCK = re.compile(r"<picture>.*?</picture>", re.DOTALL | re.IGNORECASE)
_IMG_SRC = re.compile(r"""<img\b[^>]*\bsrc=["']([^"']+)["']""", re.IGNORECASE)
_ANCHOR_OPEN = re.compile(r"""<a\b[^>]*\bhref=["']([^"']+)["'][^>]*>""", re.IGNORECASE)
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_BADGE_SHIELD = re.compile(r"img\.shields\.io|badge", re.IGNORECASE)
_REMOTE_PREFIXES = ("http://", "https://", "//")


def _collect_markdown_files(scopes: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for scope in scopes:
        if scope.is_file():
            files.append(scope)
        elif scope.is_dir():
            files.extend(sorted(scope.glob("*.md")))
    return files


def _collect_public_markdown_files() -> list[Path]:
    return _collect_markdown_files(_PUBLIC_SCOPES)


def _collect_reader_facing_lkw_markdown_files() -> list[Path]:
    return _collect_markdown_files(_READER_FACING_LKW_SCOPES)


def _strip_fenced_code(text: str) -> str:
    return _FENCED_CODE.sub("", text)


def _is_already_wrapped(text: str, start: int) -> bool:
    before = text[:start].rstrip()
    if not before.endswith(">"):
        return False
    last_a = before.rfind("<a ")
    if last_a == -1:
        return False
    between = text[last_a:start]
    return "</a>" not in between


def _anchor_href_before_picture(text: str, start: int) -> str | None:
    if not _is_already_wrapped(text, start):
        return None
    before = text[:start].rstrip()
    last_a = before.rfind("<a ")
    if last_a == -1:
        return None
    anchor_match = _ANCHOR_OPEN.search(text, last_a, start)
    return anchor_match.group(1).strip() if anchor_match else None


def _is_qualifying_picture_src(src: str, *, allow_lkw_assets: bool) -> bool:
    normalized = src.strip().replace("\\", "/")
    if not normalized or normalized.startswith(_REMOTE_PREFIXES):
        return False
    if not allow_lkw_assets and LKW_ASSET_PREFIX in normalized:
        return False
    if _BADGE_SHIELD.search(normalized):
        return False
    return normalized.endswith((".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp"))


def _is_qualifying_markdown_image_target(target: str, *, allow_lkw_assets: bool) -> bool:
    normalized = target.strip().replace("\\", "/")
    if not normalized or normalized.startswith(_REMOTE_PREFIXES):
        return False
    if not allow_lkw_assets and LKW_ASSET_PREFIX in normalized:
        return False
    if _BADGE_SHIELD.search(normalized):
        return False
    return normalized.endswith((".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp"))


def _resolve_asset_path(doc_path: Path, asset_ref: str) -> Path:
    normalized = asset_ref.strip().replace("\\", "/")
    if normalized.startswith("/"):
        resolved = REPO_ROOT / normalized.lstrip("/")
    else:
        resolved = (doc_path.parent / normalized).resolve()
    return resolved


def _is_clickable_markdown_image(text: str, start: int, end: int, target: str) -> bool:
    if start == 0 or text[start - 1] != "[":
        return False
    suffix = f"]({target})"
    return text[end : end + len(suffix)] == suffix


def _picture_clickability_violations(
    doc_paths: list[Path],
    *,
    allow_lkw_assets: bool,
) -> list[str]:
    violations: list[str] = []
    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8")
        scan_text = _strip_fenced_code(text)
        for match in _PICTURE_BLOCK.finditer(scan_text):
            block = match.group(0)
            img_match = _IMG_SRC.search(block)
            if not img_match:
                continue
            src = img_match.group(1).strip()
            if not _is_qualifying_picture_src(src, allow_lkw_assets=allow_lkw_assets):
                continue
            href = _anchor_href_before_picture(scan_text, match.start())
            rel = doc_path.relative_to(REPO_ROOT).as_posix()
            if href is None:
                violations.append(f"{rel}: <picture> block with src={src!r} is not wrapped in <a href>")
                continue
            if href != src:
                violations.append(
                    f"{rel}: <a href={href!r}> must match fallback <img src={src!r}>"
                )
                continue
            resolved = _resolve_asset_path(doc_path, src)
            if not resolved.is_file():
                violations.append(f"{rel}: missing asset for clickable picture target {src!r}")
    return violations


def _markdown_image_clickability_violations(
    doc_paths: list[Path],
    *,
    allow_lkw_assets: bool,
) -> list[str]:
    violations: list[str] = []
    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8")
        scan_text = _strip_fenced_code(text)
        for match in _MD_IMAGE.finditer(scan_text):
            alt = match.group(1)
            target = match.group(2).strip()
            if not _is_qualifying_markdown_image_target(target, allow_lkw_assets=allow_lkw_assets):
                continue
            start = match.start()
            end = match.end()
            rel = doc_path.relative_to(REPO_ROOT).as_posix()
            if not _is_clickable_markdown_image(scan_text, start, end, target):
                violations.append(
                    f"{rel}: markdown image ![{alt}]({target}) must be wrapped as [![{alt}]({target})]({target})"
                )
                continue
            resolved = _resolve_asset_path(doc_path, target)
            if not resolved.is_file():
                violations.append(f"{rel}: missing asset for clickable markdown image target {target!r}")
    return violations


@pytest.fixture(scope="module")
def public_markdown_files() -> list[Path]:
    return [
        path
        for path in _collect_public_markdown_files()
        if LKW_DOCS_PREFIX not in path.relative_to(REPO_ROOT).as_posix()
    ]


@pytest.fixture(scope="module")
def reader_facing_lkw_markdown_files() -> list[Path]:
    return _collect_reader_facing_lkw_markdown_files()


def test_qualifying_picture_blocks_are_directly_clickable(public_markdown_files: list[Path]) -> None:
    violations = _picture_clickability_violations(
        public_markdown_files,
        allow_lkw_assets=False,
    )
    assert not violations, "Clickable picture contract violations:\n" + "\n".join(violations)


def test_qualifying_markdown_images_are_directly_clickable(public_markdown_files: list[Path]) -> None:
    violations = _markdown_image_clickability_violations(
        public_markdown_files,
        allow_lkw_assets=False,
    )
    assert not violations, "Clickable markdown image contract violations:\n" + "\n".join(violations)


def test_reader_facing_lkw_qualifying_picture_blocks_are_directly_clickable(
    reader_facing_lkw_markdown_files: list[Path],
) -> None:
    violations = _picture_clickability_violations(
        reader_facing_lkw_markdown_files,
        allow_lkw_assets=True,
    )
    assert not violations, "Reader-facing LKW clickable picture contract violations:\n" + "\n".join(
        violations
    )


def test_reader_facing_lkw_qualifying_markdown_images_are_directly_clickable(
    reader_facing_lkw_markdown_files: list[Path],
) -> None:
    violations = _markdown_image_clickability_violations(
        reader_facing_lkw_markdown_files,
        allow_lkw_assets=True,
    )
    assert not violations, (
        "Reader-facing LKW clickable markdown image contract violations:\n"
        + "\n".join(violations)
    )
