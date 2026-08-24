# © Artur Czarnecki. All rights reserved.

"""Bounded public documentation link integrity helpers."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

PUBLIC_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs/project/overview/WHY_INTERGRAX.md",
    REPO_ROOT / "docs/project/overview/USE_CASES.md",
    REPO_ROOT / "docs/project/overview/ALTERNATIVES_AND_TRADEOFFS.md",
    REPO_ROOT / "docs/project/architecture/ARCHITECTURE_OVERVIEW.md",
    REPO_ROOT / "docs/project/proofs/PROOFS.md",
    REPO_ROOT / "docs/project/proofs/PROOF_LIBRARY.md",
    REPO_ROOT / "docs/project/builders/BUILDER_QUICKSTART.md",
    REPO_ROOT / "docs/project/builders/EVALUATION_GUIDE.md",
    REPO_ROOT / "docs/project/community/PARTNERS.md",
    REPO_ROOT / "docs/project/community/PUBLIC_DOCUMENTATION_MAP.md",
    REPO_ROOT / "applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md",
    REPO_ROOT / "applications/local_workspace_application/docs/product/QUICKSTART.md",
)

_TRAVERSE_ALLOW_PREFIXES = (
    "docs/project/overview/",
    "docs/project/architecture/",
    "docs/project/proofs/",
    "docs/project/builders/",
    "docs/project/community/",
    "docs/project/capabilities/",
    "docs/project/integrations/",
    "docs/project/assets/public/",
    "docs/project/technical/",
    "applications/local_workspace_application/docs/product/",
    "applications/local_workspace_application/docs/proof/",
    "applications/local_workspace_application/docs/assets/fullsize/",
)

_TRAVERSE_ALLOW_FILES = frozenset({"README.md", "SECURITY.md", "LICENSE"})

_NO_TRAVERSE_PARTS = (
    "/archive/",
    "/architecture/satellites/",
    "/maintainers/",
    "/audit_results/",
    "/capabilities/plan/",
    "/technical/adr/",
    "/technical/guides/",
    "/overview/case-studies/",
    "/platform_proofs/",
)

_MD_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_IMG_SRC = re.compile(r"""<img\b[^>]*\bsrc=["']([^"']+)["']""", re.IGNORECASE)
_SOURCE = re.compile(r"""<source\b[^>]*\bsrcset=["']([^"']+)["']""", re.IGNORECASE)
_ANCHOR = re.compile(r"""<a\b[^>]*\bhref=["']([^"']+)["']""", re.IGNORECASE)
_FENCED = re.compile(r"```.*?```", re.DOTALL)
_REMOTE_PREFIXES = ("http://", "https://", "mailto:", "//")


@dataclass(frozen=True, slots=True)
class BrokenLocalLink:
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class PublicLinkIntegrityReport:
    roots_checked: int
    documents_checked: int
    local_links_checked: int
    assets_checked: int
    broken_links: tuple[BrokenLocalLink, ...]


def should_traverse(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if not rel.endswith(".md"):
        return False
    if any(part in rel for part in _NO_TRAVERSE_PARTS):
        return False
    if rel in _TRAVERSE_ALLOW_FILES:
        return True
    return rel.startswith(_TRAVERSE_ALLOW_PREFIXES)


def extract_local_refs(text: str) -> list[str]:
    scan = _FENCED.sub("", text)
    refs: list[str] = []
    refs.extend(target.strip() for _, target in _MD_LINK.findall(scan))
    refs.extend(target.strip() for _, target in _MD_IMAGE.findall(scan))
    refs.extend(match.strip() for match in _IMG_SRC.findall(scan))
    refs.extend(match.strip() for match in _SOURCE.findall(scan))
    refs.extend(match.strip() for match in _ANCHOR.findall(scan))
    return refs


def is_local_ref(ref: str) -> bool:
    if not ref or ref.startswith("#"):
        return False
    if ref.startswith(_REMOTE_PREFIXES):
        return False
    return not (ref.startswith("<") and ref.endswith(">"))


def resolve_local_target(base_dir: Path, ref: str) -> Path | None:
    path_part = ref.split("#", 1)[0].strip()
    if not path_part:
        return None
    if path_part.startswith("/"):
        return (REPO_ROOT / path_part.lstrip("/")).resolve()
    return (base_dir / path_part).resolve()


def collect_public_link_integrity_report() -> PublicLinkIntegrityReport:
    visited: set[Path] = set()
    queue: deque[Path] = deque(root.resolve() for root in PUBLIC_ROOTS if root.is_file())

    broken: list[BrokenLocalLink] = []
    local_links_checked = 0
    assets_checked = 0

    while queue:
        doc_path = queue.popleft()
        if doc_path in visited or doc_path.suffix.lower() != ".md":
            continue
        if not should_traverse(doc_path):
            continue

        visited.add(doc_path)
        source = doc_path.relative_to(REPO_ROOT).as_posix()
        text = doc_path.read_text(encoding="utf-8")

        for ref in extract_local_refs(text):
            if not is_local_ref(ref):
                continue

            local_links_checked += 1
            target = resolve_local_target(doc_path.parent, ref)
            if target is None:
                continue

            if target.suffix.lower() == ".md" and target.is_file():
                if should_traverse(target) and target not in visited:
                    queue.append(target)
                continue

            if target.exists():
                assets_checked += 1
                continue

            broken.append(BrokenLocalLink(source=source, target=ref))

    unique_broken = tuple(dict.fromkeys(broken).keys())
    return PublicLinkIntegrityReport(
        roots_checked=len(PUBLIC_ROOTS),
        documents_checked=len(visited),
        local_links_checked=local_links_checked,
        assets_checked=assets_checked,
        broken_links=unique_broken,
    )
