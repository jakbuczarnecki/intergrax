# © Artur Czarnecki. All rights reserved.
"""Shared architecture hub split utilities (F4)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "docs" / "architecture"
SAT_DIR = ARCH_DIR / "satellites"

H1_SECTION = re.compile(r"^# (\d+)\.\s+(.+)$", re.MULTILINE)
NUMBERED_H2 = re.compile(r"^## (\d+)\.\s", re.MULTILINE)
SUBSECTION_H2 = re.compile(r"^## (\d+)\.(\d+)\s", re.MULTILINE)
SATELLITE_HEADER = re.compile(
    r"^# .+ — .+\n\n\*\*Parent hub:\*\* \[`.+`\]\(\../.+\.md\)\n\n",
    re.MULTILINE,
)
SATELLITE_INDEX_START = "## Architecture satellites (read on demand)"

# Canonical merge order when reassembling hub + satellites before re-split.
SATELLITE_SUFFIX_ORDER = (
    "extended_depth",
    "runtime_extended",
    "production_gates",
    "scenario_catalog",  # legacy name — removed on re-split
    "provider_catalog",
    "provider_index",
    "invocation_patterns",
    "selection_and_plugins",
    "invocation_patterns",
    "runtime_config_reference",
    "providers_catalog",
    "audit_register",
    "pipelines_detail",
    "graph_rag",
    "skill_catalog",
    "tool_surface_detail",
    "selection_and_plugins",
)


def tokens(text: str) -> int:
    return len(text) // 4


def parse_h1_sections(text: str) -> tuple[str, list[tuple[int, str, str]]]:
    matches = list(H1_SECTION.finditer(text))
    if not matches:
        return text, []
    preamble = text[: matches[0].start()]
    sections: list[tuple[int, str, str]] = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((num, m.group(0), text[start:end]))
    return preamble, sections


def parse_numbered_h2(text: str) -> tuple[str, list[tuple[int, str]]]:
    matches = list(NUMBERED_H2.finditer(text))
    if not matches:
        return text, []
    preamble = text[: matches[0].start()]
    sections: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((num, text[start:end]))
    return preamble, sections


def parse_subsection_h2(text: str, major: int) -> tuple[str, list[tuple[int, str]]]:
    pat = re.compile(rf"^## {major}\.(\d+)", re.MULTILINE)
    matches = list(pat.finditer(text))
    if not matches:
        return text, []
    preamble = text[: matches[0].start()]
    sections: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        minor = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((minor, text[start:end]))
    return preamble, sections


def render_arch_satellite(domain: str, label: str, body: str) -> str:
    return (
        f"# {domain} — {label}\n\n"
        f"**Parent hub:** [`{domain}.md`](../{domain}.md)\n\n"
        f"{body.strip()}\n"
    )


def satellite_index_rows(domain: str, files: dict[str, str]) -> list[str]:
    rows = [
        "## Architecture satellites (read on demand)",
        "",
        "Large § blocks moved out of the architecture hub to reduce Cursor context use.",
        "Load **only** the satellite matching your task or cited §.",
        "",
        "| Satellite | Contents |",
        "|-----------|----------|",
    ]
    for fname in sorted(files):
        label = fname.replace(f"{domain}_", "").replace(".md", "").replace("_", " ")
        rows.append(f"| [`satellites/{fname}`](satellites/{fname}) | {label} |")
    rows.extend(
        [
            "",
            "> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.",
            "",
        ]
    )
    return rows


def strip_satellite_header(text: str) -> str:
    return SATELLITE_HEADER.sub("", text, count=1).strip()


def remove_arch_satellite_index(hub_text: str) -> str:
    if SATELLITE_INDEX_START not in hub_text:
        return hub_text
    start = hub_text.index(SATELLITE_INDEX_START)
    rest = hub_text[start:]
    m = re.search(
        r"\n\n(?:## (?!Architecture satellites)|# \d+\.)",
        rest,
    )
    end = start + m.start() if m else len(hub_text)
    return hub_text[:start].rstrip() + "\n" + hub_text[end:].lstrip("\n")


def satellite_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    for i, suffix in enumerate(SATELLITE_SUFFIX_ORDER):
        if stem.endswith(suffix):
            return (i, stem)
    return (len(SATELLITE_SUFFIX_ORDER), stem)


def merge_arch_satellites(domain: str) -> str:
    """Reconstruct full canon from hub + arch/ satellites (lossless, canonical § order)."""
    hub_path = ARCH_DIR / f"{domain}.md"
    hub = remove_arch_satellite_index(hub_path.read_text(encoding="utf-8")).rstrip()
    sats = sorted(SAT_DIR.glob(f"{domain}_*.md"), key=satellite_sort_key)
    parts = [hub] + [strip_satellite_header(p.read_text(encoding="utf-8")) for p in sats]
    combined = "\n\n".join(p.strip() for p in parts if p.strip()) + "\n"

    preamble, h1 = parse_h1_sections(combined)
    if h1:
        ordered = sorted(h1, key=lambda x: x[0])
        return preamble.rstrip() + "\n\n" + "\n\n".join(b.rstrip() for _, _, b in ordered) + "\n"

    preamble, h2 = parse_numbered_h2(combined)
    if h2:
        ordered = sorted(h2, key=lambda x: x[0])
        return preamble.rstrip() + "\n\n" + "\n\n".join(b.rstrip() for _, b in ordered) + "\n"

    for major in (42,):
        preamble, sub = parse_subsection_h2(combined, major)
        if sub:
            ordered = sorted(sub, key=lambda x: x[0])
            return preamble.rstrip() + "\n\n" + "\n\n".join(b.rstrip() for _, b in ordered) + "\n"

    return combined


def insert_arch_satellite_index(hub_text: str, index_rows: list[str]) -> str:
    if SATELLITE_INDEX_START in hub_text:
        hub_text = remove_arch_satellite_index(hub_text)
    marker = "## Table of contents"
    if marker in hub_text:
        idx = hub_text.index(marker)
        return hub_text[:idx] + "\n".join(index_rows) + "\n\n" + hub_text[idx:]
    marker2 = "## Cursor read scope (token budget)"
    if marker2 in hub_text:
        end = hub_text.find("\n---\n", hub_text.index(marker2))
        if end != -1:
            return hub_text[: end + 5] + "\n\n" + "\n".join(index_rows) + hub_text[end + 5 :]
    return "\n".join(index_rows) + "\n\n" + hub_text


def split_heading_markers(text: str, domain: str, markers: tuple[tuple[str, str], ...]) -> tuple[str, dict[str, str]]:
    satellites: dict[str, str] = {}
    hub = text
    for marker, slug in reversed(markers):
        idx = hub.find(marker)
        if idx == -1:
            continue
        tail = hub[idx:]
        hub = hub[:idx].rstrip() + "\n"
        fname = f"{domain}_{slug}.md"
        satellites[fname] = render_arch_satellite(domain, slug.replace("_", " "), tail)
    return hub, satellites
