# © Artur Czarnecki. All rights reserved.
"""Shared plan hub split utilities."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "docs" / "project" / "maintainers" / "plans"
SAT_DIR = PLAN_DIR / "satellites"

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
        rows.append(f"| [`plan/satellites/{fname}`](plan/satellites/{fname}) | {label} |")
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


PLAN_READ_SCOPE_MARKER = "## Cursor read scope (token budget)"
PLAN_SCOPE_BLOCK_RE = re.compile(
    rf"{re.escape(PLAN_READ_SCOPE_MARKER)}.*?\n---\n",
    re.DOTALL,
)

SKIP_PLAN_HUBS = frozenset({"AUDIT_IDEAL_2026.md", "IDEAL_HARNESS_L3.md"})


def satellite_links(hub_text: str) -> list[str]:
    marker = "## Satellite registers (read on demand)"
    if marker not in hub_text:
        return []
    section = hub_text.split(marker, 1)[1].split("\n---\n", 1)[0]
    return re.findall(r"\[`plan/satellites/([^`]+)`\]", section)


def render_plan_read_scope_block(domain: str, scope: str) -> str:
    return (
        f"{PLAN_READ_SCOPE_MARKER}\n\n"
        f"**Do not read this entire file in one session** ({domain} plan).\n\n"
        f"- **Implement / audit default:** {scope}\n"
        f"- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.\n"
        f"- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.\n"
        f"- **Architecture hub:** [`architecture/{domain}.md`](../architecture/{domain}.md) read-scope block only.\n"
        f"- **Audit slice:** [`docs/audit_results/{domain}.md`](../docs/audit_results/{domain}.md).\n"
        f"- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.\n\n"
        f"---\n"
    )


def upsert_plan_read_scope(text: str, domain: str, scope: str) -> str:
    new_block = render_plan_read_scope_block(domain, scope)
    if PLAN_READ_SCOPE_MARKER in text:
        return PLAN_SCOPE_BLOCK_RE.sub(new_block, text, count=1)
    insert_at = text.find("\n---\n")
    if insert_at == -1:
        return text
    insert_at += len("\n---\n")
    return text[:insert_at] + "\n" + new_block + "\n" + text[insert_at:].lstrip("\n")


def normalize_plan_satellite_budget_line(text: str) -> str:
    old = "> **Cursor context budget:** read this hub + **at most one** satellite per session."
    new = "> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session."
    return text.replace(old, new)
