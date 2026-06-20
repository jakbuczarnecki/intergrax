# © Artur Czarnecki. All rights reserved.
"""Shared architecture hub split utilities (F4)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "docs" / "architecture"
SAT_DIR = ARCH_DIR / "arch"

H1_SECTION = re.compile(r"^# (\d+)\.\s+(.+)$", re.MULTILINE)
NUMBERED_H2 = re.compile(r"^## (\d+)\.\s", re.MULTILINE)
SUBSECTION_H2 = re.compile(r"^## (\d+)\.(\d+)\s", re.MULTILINE)


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
        rows.append(f"| [`arch/{fname}`](arch/{fname}) | {label} |")
    rows.extend(
        [
            "",
            "> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.",
            "",
        ]
    )
    return rows


def insert_arch_satellite_index(hub_text: str, index_rows: list[str]) -> str:
    if "## Architecture satellites (read on demand)" in hub_text:
        return hub_text
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
