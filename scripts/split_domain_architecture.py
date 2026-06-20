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
    parse_numbered_h2,
    parse_subsection_h2,
    render_arch_satellite,
    satellite_index_rows,
    split_heading_markers,
    tokens,
)

ROOT = Path(__file__).resolve().parents[1]


def split_arch_domain(cfg: ArchSplitConfig) -> dict[str, int]:
    source = ARCH_DIR / f"{cfg.domain}.md"
    raw = source.read_text(encoding="utf-8")
    hub_text = raw
    satellites: dict[str, str] = {}
    d = cfg.domain

    preamble, h1_sections = parse_h1_sections(hub_text)
    if h1_sections and cfg.hub_section_max > 0:
        hub_parts = [preamble.rstrip()]
        sat_depth: list[str] = []
        sat_prod: list[str] = []
        for num, _title, body in h1_sections:
            if num <= cfg.hub_section_max or num in cfg.extra_hub_sections:
                hub_parts.append(body.rstrip())
            elif num >= cfg.production_section_min:
                sat_prod.append(body.rstrip())
            else:
                sat_depth.append(body.rstrip())
        hub_text = "\n\n".join(hub_parts).rstrip() + "\n"
        if sat_depth:
            fn = f"{d}_extended_depth.md"
            satellites[fn] = render_arch_satellite(d, "extended depth", "\n\n".join(sat_depth))
        if sat_prod:
            fn = f"{d}_production_gates.md"
            satellites[fn] = render_arch_satellite(d, "production gates (§40+)", "\n\n".join(sat_prod))

    if cfg.subsection_major is not None and cfg.subsection_minor_max is not None:
        pre, sub_secs = parse_subsection_h2(hub_text, cfg.subsection_major)
        if sub_secs:
            hub_parts = [pre.rstrip()]
            tail: list[str] = []
            for minor, body in sub_secs:
                if minor <= cfg.subsection_minor_max:
                    hub_parts.append(body.rstrip())
                else:
                    tail.append(body.rstrip())
            hub_text = "\n\n".join(hub_parts).rstrip() + "\n"
            if tail:
                fn = f"{d}_runtime_extended.md"
                satellites[fn] = render_arch_satellite(
                    d,
                    f"§{cfg.subsection_major}.{cfg.subsection_minor_max + 1}+ runtime depth",
                    "\n\n".join(tail),
                )

    if cfg.numbered_h2_max is not None:
        pre, num_secs = parse_numbered_h2(hub_text)
        if num_secs:
            hub_parts = [pre.rstrip()]
            tail = []
            for num, body in num_secs:
                if num <= cfg.numbered_h2_max:
                    hub_parts.append(body.rstrip())
                else:
                    tail.append(body.rstrip())
            hub_text = "\n\n".join(hub_parts).rstrip() + "\n"
            if tail:
                fn = f"{d}_scenario_catalog.md"
                satellites[fn] = render_arch_satellite(
                    d, f"§{cfg.numbered_h2_max + 1}+ scenarios & control", "\n\n".join(tail)
                )

    if cfg.h2_satellite_markers:
        hub_text, marker_sats = split_heading_markers(hub_text, d, cfg.h2_satellite_markers)
        satellites.update(marker_sats)

    SAT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, content in satellites.items():
        (SAT_DIR / fname).write_text(content, encoding="utf-8")

    if satellites:
        hub_text = insert_arch_satellite_index(hub_text, satellite_index_rows(d, satellites))

    hub_text = re_sub_blank_runs(hub_text)
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


def re_sub_blank_runs(text: str) -> str:
    import re

    return re.sub(r"\n{4,}", "\n\n\n", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split architecture domain into hub + satellites")
    parser.add_argument("domains", nargs="*", default=list(CONFIGS.keys()))
    args = parser.parse_args(argv)
    for name in args.domains:
        if name not in CONFIGS:
            print(f"unknown domain config: {name}", file=sys.stderr)
            return 1
        split_arch_domain(CONFIGS[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
