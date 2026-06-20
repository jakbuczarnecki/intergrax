# © Artur Czarnecki. All rights reserved.
"""Shared plan hub split utilities."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "docs" / "plan"
SAT_DIR = PLAN_DIR / "plan"

CLOSED_SIGNALS = (
    "(closed",
    " closed",
    "(complete",
    " complete",
    "archived",
    "(done",
    "— done",
    "— complete",
)

REGISTER_SIGNALS = (
    "Master register",
    "Paydown log",
    "Suggested PR order",
    "Remediation register",
)


def tokens(text: str) -> int:
    return len(text) // 4


def strip_leading_satellite_index(hub_text: str) -> str:
    marker = "## Satellite registers (read on demand)"
    text = hub_text.lstrip("\n")
    while text.startswith(marker):
        idx = text.index(marker)
        end = text.find("\n---\n", idx)
        if end == -1:
            text = text[:idx]
            break
        text = text[:idx] + text[end + 5 :]
        text = text.lstrip("\n")
    return text


def is_closed_heading(title: str) -> bool:
    low = title.lower()
    return any(s in low for s in CLOSED_SIGNALS)


def is_register_h3(title: str) -> bool:
    return any(k in title for k in REGISTER_SIGNALS)


def split_h2(lines: list[str]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    title = "__preamble__"
    chunk: list[str] = []
    for line in lines:
        if line.startswith("## "):
            out.append((title, chunk))
            title = line[3:].strip()
            chunk = [line]
        else:
            chunk.append(line)
    out.append((title, chunk))
    return out


def split_h3(body: list[str]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    title = "__intro__"
    chunk: list[str] = []
    for line in body:
        if line.startswith("### "):
            out.append((title, chunk))
            title = line[4:].strip()
            chunk = [line]
        else:
            chunk.append(line)
    out.append((title, chunk))
    return out


def render_satellite(domain: str, name: str, body: str) -> str:
    return (
        f"# {domain} — {name}\n\n"
        f"**Parent hub:** [`{domain}.md`](../{domain}.md)\n\n"
        f"{body.strip()}\n"
    )


def satellite_index_rows(domain: str, files: dict[str, str]) -> list[str]:
    rows = [
        "## Satellite registers (read on demand)",
        "",
        "Large historical registers moved out of the hub to reduce Cursor context use.",
        "Load **only** the satellite matching your task or cited gap ID.",
        "",
        "| Satellite | Contents |",
        "|-----------|----------|",
    ]
    for fname in sorted(files):
        label = fname.replace(f"{domain}_", "").replace(".md", "").replace("_", " ")
        rows.append(f"| [`plan/plan/{fname}`](plan/plan/{fname}) | {label} |")
    rows.extend(
        [
            "",
            "> **Cursor context budget:** read this hub + **at most one** satellite per session.",
            "",
        ]
    )
    return rows


def dedupe_satellite_index(hub_text: str) -> str:
    marker = "## Satellite registers (read on demand)"
    if hub_text.count(marker) <= 1:
        return hub_text
    first = hub_text.index(marker)
    tail = hub_text[first + len(marker) :]
    second = tail.find(marker)
    if second == -1:
        return hub_text
    abs_second = first + len(marker) + second
    end = hub_text.find("\n---\n", abs_second + len(marker))
    if end == -1:
        end = len(hub_text)
    return hub_text[:abs_second].rstrip() + "\n\n" + hub_text[end:].lstrip("\n")


def insert_satellite_index(hub_text: str, index_rows: list[str]) -> str:
    marker = "**Note on audit source documents:**"
    if marker in hub_text:
        end = hub_text.find("\n---\n", hub_text.index(marker))
        if end != -1:
            return hub_text[:end] + "\n\n" + "\n".join(index_rows) + hub_text[end:]
    marker2 = "## Phase AUDIT-IDEAL"
    if marker2 in hub_text:
        idx = hub_text.index(marker2)
        return hub_text[:idx] + "\n".join(index_rows) + "\n\n---\n\n" + hub_text[idx:]
    return "\n".join(index_rows) + "\n\n" + hub_text
