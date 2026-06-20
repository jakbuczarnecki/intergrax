#!/usr/bin/env python3
"""Split docs/plan/PLATFORM_FOUNDATION.md into hub + plan/plan/ satellites."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/plan/PLATFORM_FOUNDATION.md"
OUT_DIR = ROOT / "docs/plan/plan"

HUB_H3_PREFIXES = (
    "6.1 Harness platform maintenance",
    "6.1av ",
    "6.1p ",
    "6.2af ",  # M.6 P5 Planned
    "6.3 ",
    "6.3a ",
)


def is_hub_h3(title: str) -> bool:
    return any(title.startswith(p) for p in HUB_H3_PREFIXES)


def is_closed_h3(title: str) -> bool:
    if is_hub_h3(title):
        return False
    low = title.lower()
    return any(
        s in low
        for s in (
            "(closed",
            " closed",
            "(complete",
            " complete",
            "archived",
            "(done",
            "— done",
            "— complete",
        )
    )


def remove_duplicate_block(lines: list[str]) -> list[str]:
    """Remove duplicated §6 block appended after 2026-06-07 sync footer."""
    marker = "*Plan synced (2026-06-07)."
    start_idx = None
    for i, line in enumerate(lines):
        if marker in line:
            # find ### 6.1s duplicate after --- following this
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].startswith("### 6.1s Harness implementation queue"):
                    start_idx = j
                    break
            break
    if start_idx is None:
        return lines
    end_idx = None
    for j in range(start_idx, len(lines)):
        if lines[j].startswith("## Phase V-REM"):
            end_idx = j
            break
    if end_idx is None:
        return lines
    return lines[:start_idx] + lines[end_idx:]


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


def is_register_h3(title: str) -> bool:
    return any(
        k in title
        for k in (
            "Master register",
            "Paydown log",
            "Suggested PR order",
            "Remediation register",
        )
    )


def render_satellite(name: str, body: str) -> str:
    return (
        f"# Platform Foundation — {name}\n\n"
        f"**Parent hub:** [`PLATFORM_FOUNDATION.md`](../PLATFORM_FOUNDATION.md)\n\n"
        f"{body.strip()}\n"
    )


def main() -> None:
    raw = SOURCE.read_text(encoding="utf-8")
    lines = remove_duplicate_block(raw.splitlines())
    h2s = split_h2(lines)

    hub: list[str] = []
    sat_closed_06: list[str] = []
    sat_registers: list[str] = []
    sat_06_embedded: list[str] = []
    sat_phases: list[str] = []
    sat_appendices: list[str] = []

    for h2_title, h2_body in h2s:
        if h2_title == "__preamble__":
            hub.extend(h2_body)
            continue

        if h2_title.startswith("Appendix"):
            sat_appendices.extend(h2_body)
            continue

        if h2_title.startswith("Phase ") and not h2_title.startswith(
            "Phase AUDIT-IDEAL"
        ):
            sat_phases.extend(h2_body)
            continue

        if h2_title == "6. What to implement next":
            hub.append("## 6. What to implement next")
            subs = split_h3(h2_body[1:])
            for sub_title, sub_body in subs:
                if sub_title == "__intro__":
                    hub.extend(sub_body)
                elif is_hub_h3(sub_title):
                    hub.extend(sub_body)
                elif is_closed_h3(sub_title):
                    sat_closed_06.extend(sub_body)
                else:
                    sat_06_embedded.extend(sub_body)
            continue

        if h2_title.startswith(("2.", "3.", "4.", "5.")):
            hub.append(h2_body[0])  # ## header
            subs = split_h3(h2_body)
            for sub_title, sub_body in subs:
                if sub_title == "__intro__":
                    hub.extend(sub_body[1:] if sub_body and sub_body[0].startswith("## ") else sub_body)
                elif is_register_h3(sub_title):
                    sat_registers.extend(sub_body)
                elif "Layer scorecard" in sub_title:
                    hub.extend(sub_body)  # keep FAUDIT summary in hub
                else:
                    hub.extend(sub_body)
            continue

        hub.extend(h2_body)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    satellites: dict[str, str] = {}

    if sat_closed_06:
        satellites["PLATFORM_FOUNDATION_06_closed_queues.md"] = render_satellite(
            "§6.1/§6.2 closed queues",
            "\n".join(sat_closed_06),
        )
    if sat_registers:
        satellites["PLATFORM_FOUNDATION_master_registers.md"] = render_satellite(
            "§5 master registers (ORCH, FLOW, TS, …)",
            "\n".join(sat_registers),
        )
    if sat_06_embedded:
        satellites["PLATFORM_FOUNDATION_06_phase_detail.md"] = render_satellite(
            "§6 embedded phase detail (appendices L/M/N, historical)",
            "\n".join(sat_06_embedded),
        )
    if sat_phases:
        satellites["PLATFORM_FOUNDATION_phase_closeout.md"] = render_satellite(
            "phase closeout (V-REM, FAUDIT-32, …)",
            "\n".join(sat_phases),
        )
    if sat_appendices:
        satellites["PLATFORM_FOUNDATION_appendices.md"] = render_satellite(
            "appendices B–M",
            "\n".join(sat_appendices),
        )

    for fname, content in satellites.items():
        (OUT_DIR / fname).write_text(content, encoding="utf-8")

    satellite_index = [
        "## Satellite registers (read on demand)",
        "",
        "Large historical registers moved out of the hub to reduce Cursor token use.",
        "Load **only** the satellite matching your task or cited gap ID.",
        "",
        "| Satellite | Contents |",
        "|-----------|----------|",
    ]
    labels = {
        "PLATFORM_FOUNDATION_06_closed_queues.md": "Closed §6.1/§6.2 implementation queues",
        "PLATFORM_FOUNDATION_master_registers.md": "§5 domain master registers + paydown logs",
        "PLATFORM_FOUNDATION_06_phase_detail.md": "§6 embedded phase/appendix detail (L/M/N, …)",
        "PLATFORM_FOUNDATION_phase_closeout.md": "Phase V-REM, FAUDIT-32 closeout",
        "PLATFORM_FOUNDATION_appendices.md": "Appendices B–M",
    }
    for fname in sorted(satellites):
        satellite_index.append(
            f"| [`plan/{fname}`](plan/{fname}) | {labels.get(fname, fname)} |"
        )
    satellite_index.extend(
        [
            "",
            "> **Cursor context budget:** read this hub + **at most one** satellite per session.",
            "> Closed queues and appendices are **audit-on-demand** only.",
            "",
        ]
    )

    hub_text = "\n".join(hub)
    hub_text = hub_text.replace(
        "Detailed phase registers live under [`plan/`](plan/). Appendices: [`plan/`](plan/).",
        "Detailed registers and appendices: [`plan/plan/`](plan/plan/) — **load on demand**.",
    )
    hub_text = hub_text.replace(
        "Historical phase registers (A–V) and closeout phases are decomposed under [`plan/`](plan/).",
        "Historical phase registers: [`plan/plan/PLATFORM_FOUNDATION_phase_closeout.md`](plan/PLATFORM_FOUNDATION_phase_closeout.md).",
    )

    insert_at = hub_text.find("**Note on audit source documents:**")
    if insert_at != -1:
        end = hub_text.find("\n---\n", insert_at)
        if end != -1:
            hub_text = (
                hub_text[:end]
                + "\n\n"
                + "\n".join(satellite_index)
                + hub_text[end:]
            )

    # Fix common internal links
    replacements = [
        (r"\(plan/PLATFORM_FOUNDATION\.md\)", "(plan/PLATFORM_FOUNDATION_phase_closeout.md)"),
        ("**Appendix B**", "**[Appendix B](plan/PLATFORM_FOUNDATION_appendices.md)**"),
        ("**Appendix C**", "**[Appendix C](plan/PLATFORM_FOUNDATION_appendices.md)**"),
        ("**Appendix D**", "**[Appendix D](plan/PLATFORM_FOUNDATION_appendices.md)**"),
        ("**Appendix E**", "**[Appendix E](plan/PLATFORM_FOUNDATION_appendices.md)**"),
        ("**Appendix F**", "**[Appendix F](plan/PLATFORM_FOUNDATION_appendices.md)**"),
        ("**Appendix G**", "**[Appendix G](plan/PLATFORM_FOUNDATION_appendices.md)**"),
        ("**Appendix M**", "**[Appendix M](plan/PLATFORM_FOUNDATION_appendices.md)**"),
        ("[Appendix J](plan/PLATFORM_FOUNDATION.md)", "[Appendix J](plan/PLATFORM_FOUNDATION_06_phase_detail.md)"),
        ("[Appendix B](plan/PLATFORM_FOUNDATION.md)", "[Appendix B](plan/PLATFORM_FOUNDATION_appendices.md)"),
        (
            "[Phase FAUDIT-32](plan/PLATFORM_FOUNDATION.md)",
            "[Phase FAUDIT-32](plan/PLATFORM_FOUNDATION_phase_closeout.md)",
        ),
        (
            "canonical phase narrative:** [Phase FAUDIT-32](plan/PLATFORM_FOUNDATION.md)",
            "canonical phase narrative:** [Phase FAUDIT-32](plan/PLATFORM_FOUNDATION_phase_closeout.md)",
        ),
    ]
    for old, new in replacements:
        hub_text = hub_text.replace(old, new)

    SOURCE.write_text(hub_text.rstrip() + "\n", encoding="utf-8")

    orig_tokens = len(raw) // 4
    hub_tokens = len(hub_text) // 4
    sat_tokens = sum(len(v) // 4 for v in satellites.values())
    print(f"Removed duplicate block: {len(raw.splitlines()) - len(lines)} lines")
    print(f"Hub: {len(hub_text.splitlines())} lines, ~{hub_tokens:,} tokens")
    for fname in sorted(satellites):
        c = satellites[fname]
        print(f"  {fname}: {len(c.splitlines())} lines, ~{len(c)//4:,} tokens")
    print(f"Original: ~{orig_tokens:,} | Hub: ~{hub_tokens:,} ({100*hub_tokens/orig_tokens:.0f}%)")
    print(f"Satellites: ~{sat_tokens:,} tokens (on-demand)")


if __name__ == "__main__":
    main()
