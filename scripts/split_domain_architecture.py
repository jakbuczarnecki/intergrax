#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Split large architecture domain docs into token-efficient hubs + arch/ satellites (F4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arch_hub_config import CONFIGS, ArchSplitConfig
from arch_hub_lib import (
    ARCH_DIR,
    SAT_DIR,
    insert_arch_satellite_index,
    parse_h1_sections,
    render_arch_satellite,
    satellite_index_rows,
    tokens,
)

ROOT = Path(__file__).resolve().parents[1]


def split_h2_markers(text: str, cfg: ArchSplitConfig) -> tuple[str, dict[str, str]]:
    """Split TOOLS-like docs on ## marker headings."""
    satellites: dict[str, str] = {}
    hub = text
    markers = list(cfg.h2_satellite_markers)
    # Split from bottom to top so earlier markers are not swallowed by later cuts.
    for marker, slug in reversed(markers):
        idx = hub.find(marker)
        if idx == -1:
            continue
        tail = hub[idx:]
        hub = hub[:idx].rstrip() + "\n"
        fname = f"{cfg.domain}_{slug}.md"
        satellites[fname] = render_arch_satellite(cfg.domain, slug.replace("_", " "), tail)
    return hub, satellites


def split_h1_sections(cfg: ArchSplitConfig) -> dict[str, int]:
    source = ARCH_DIR / f"{cfg.domain}.md"
    raw = source.read_text(encoding="utf-8")
    preamble, sections = parse_h1_sections(raw)
    if not sections and not cfg.h2_satellite_markers:
        raise ValueError(f"{cfg.domain}: no # N. sections found")

    hub_parts: list[str] = [preamble.rstrip()]
    sat_depth: list[str] = []
    sat_prod: list[str] = []
    satellites: dict[str, str] = {}

    if sections:
        for num, _title, body in sections:
            if num <= cfg.hub_section_max or num in cfg.extra_hub_sections:
                hub_parts.append(body.rstrip())
            elif num >= cfg.production_section_min:
                sat_prod.append(body.rstrip())
            else:
                sat_depth.append(body.rstrip())

        d = cfg.domain
        if sat_depth:
            fn = f"{d}_extended_depth.md"
            satellites[fn] = render_arch_satellite(
                d, "extended depth (§22–§39)", "\n\n".join(sat_depth)
            )
        if sat_prod:
            fn = f"{d}_production_gates.md"
            satellites[fn] = render_arch_satellite(
                d, "production gates (§40+)", "\n\n".join(sat_prod)
            )

    hub_text = "\n\n".join(hub_parts).rstrip() + "\n"
    if cfg.h2_satellite_markers:
        hub_text, h2_sats = split_h2_markers(hub_text, cfg)
        satellites.update(h2_sats)

    SAT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, content in satellites.items():
        (SAT_DIR / fname).write_text(content, encoding="utf-8")

    if satellites:
        hub_text = insert_arch_satellite_index(hub_text, satellite_index_rows(cfg.domain, satellites))

    source.write_text(hub_text.rstrip() + "\n", encoding="utf-8")

    stats = {
        "hub_tokens": tokens(hub_text),
        "hub_lines": len(hub_text.splitlines()),
        "orig_tokens": tokens(raw),
        "sat_tokens": sum(tokens(c) for c in satellites.values()),
    }
    print(
        f"{cfg.domain}: hub {stats['hub_lines']} lines ~{stats['hub_tokens']:,} tok "
        f"(was ~{stats['orig_tokens']:,}, satellites ~{stats['sat_tokens']:,})"
    )
    for fname in sorted(satellites):
        print(f"  wrote architecture/arch/{fname}")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split architecture domain into hub + satellites")
    parser.add_argument(
        "domains",
        nargs="*",
        default=list(CONFIGS.keys()),
        help=f"Domain basenames (default: {', '.join(CONFIGS)})",
    )
    args = parser.parse_args(argv)
    for name in args.domains:
        if name not in CONFIGS:
            print(f"unknown domain config: {name}", file=sys.stderr)
            return 1
        split_h1_sections(CONFIGS[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
