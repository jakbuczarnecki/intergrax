#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Split domain plan files into token-efficient hubs + plan/satellites/ satellites."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from plan_hub_config import CONFIGS, PlanSplitConfig
from plan_hub_lib import (
    SAT_DIR,
    insert_satellite_index,
    is_closed_heading,
    is_register_h3,
    render_satellite,
    satellite_index_rows,
    split_h2,
    split_h3,
    tokens,
    dedupe_satellite_index,
    strip_leading_satellite_index,
)

ROOT = Path(__file__).resolve().parents[2]


def remove_duplicate_sync_block(lines: list[str]) -> list[str]:
    marker = "*Plan synced (2026-06-07)."
    start_idx = end_idx = None
    for i, line in enumerate(lines):
        if marker in line:
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].startswith("### 6.1s Harness implementation queue"):
                    start_idx = j
                    break
            break
    if start_idx is None:
        return lines
    for j in range(start_idx, len(lines)):
        if lines[j].startswith("## Phase V-REM"):
            end_idx = j
            break
    if end_idx is None:
        return lines
    return lines[:start_idx] + lines[end_idx:]


def remove_foreign_block(lines: list[str], cfg: PlanSplitConfig) -> list[str]:
    if not cfg.foreign_block_start or not cfg.foreign_block_end:
        return lines
    start = end = None
    for i, line in enumerate(lines):
        if start is None and cfg.foreign_block_start in line:
            start = i
        if start is not None and cfg.foreign_block_end in line:
            end = i
            break
    if start is None or end is None:
        return lines
    stub = (cfg.foreign_stub or "").splitlines()
    return lines[:start] + stub + lines[end:]


def is_hub_h3(title: str, cfg: PlanSplitConfig) -> bool:
    return any(title.startswith(p) for p in cfg.hub_h3_prefixes)


def should_keep_h2(title: str, cfg: PlanSplitConfig) -> bool:
    if title == "__preamble__":
        return True
    if title.startswith("Appendix"):
        return False
    if cfg.move_h2_phase_closeout and title.startswith("Phase ") and title not in (
        "Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)",
    ):
        if not any(title.startswith(p) for p in cfg.keep_h2_prefixes):
            return False
    if any(title.startswith(p) for p in cfg.keep_h2_prefixes):
        return True
    if title == "6. What to implement next":
        return True
    if title.startswith(("0.", "1.", "2.", "3.", "4.", "5.")):
        return True
    if title.startswith("Phase AUDIT-IDEAL"):
        return True
    if title.startswith("Documentation model"):
        return True
    if title.startswith("Satellite registers"):
        return True
    # CVL audit sections
    if "CVL-" in title or "CRITIC_VERIFICATION-LC" in title:
        return True
    return cfg.domain != "PLATFORM_FOUNDATION" and not title.startswith("Phase ")


def split_plan(cfg: PlanSplitConfig) -> dict[str, int]:
    source = ROOT / "docs/plan" / f"{cfg.domain}.md"
    raw = source.read_text(encoding="utf-8")
    raw = strip_leading_satellite_index(raw)
    lines = raw.splitlines()
    if cfg.dedupe_sync_footer:
        lines = remove_duplicate_sync_block(lines)
    lines = remove_foreign_block(lines, cfg)

    h2s = split_h2(lines)
    hub: list[str] = []
    sat_closed: list[str] = []
    sat_registers: list[str] = []
    sat_embedded: list[str] = []
    sat_phases: list[str] = []
    sat_appendices: list[str] = []
    sat_audit_history: list[str] = []

    for h2_title, h2_body in h2s:
        if h2_title == "__preamble__":
            # Also split ### in preamble for domains without ## 6
            subs = split_h3(h2_body)
            if len(subs) <= 1:
                hub.extend(h2_body)
                continue
            for sub_title, sub_body in subs:
                if sub_title == "__intro__":
                    hub.extend(sub_body)
                elif is_hub_h3(sub_title, cfg):
                    hub.extend(sub_body)
                elif is_closed_heading(sub_title):
                    sat_closed.extend(sub_body)
                elif is_register_h3(sub_title):
                    sat_registers.extend(sub_body)
                else:
                    sat_embedded.extend(sub_body)
            continue

        if any(h2_title.startswith(p) for p in cfg.move_h2_prefixes):
            sat_audit_history.extend(h2_body)
            continue

        if any(h2_title.startswith(p) for p in cfg.move_h2_detail_prefixes):
            sat_embedded.extend(h2_body)
            continue

        if any(h2_title.startswith(p) for p in cfg.split_h3_in_h2_prefixes):
            hub.append(h2_body[0])
            for sub_title, sub_body in split_h3(h2_body[1:]):
                if sub_title == "__intro__":
                    hub.extend(sub_body)
                elif is_hub_h3(sub_title, cfg):
                    hub.extend(sub_body)
                elif is_closed_heading(sub_title) or is_register_h3(sub_title):
                    sat_closed.extend(sub_body)
                else:
                    sat_embedded.extend(sub_body)
            continue

        if h2_title.startswith("Appendix") and cfg.move_h2_appendix:
            sat_appendices.extend(h2_body)
            continue

        if h2_title.startswith("Phase ") and cfg.move_h2_phase_closeout:
            if not any(h2_title.startswith(p) for p in cfg.keep_h2_prefixes):
                sat_phases.extend(h2_body)
                continue

        if not should_keep_h2(h2_title, cfg):
            if h2_title.startswith("Phase "):
                sat_phases.extend(h2_body)
            else:
                sat_embedded.extend(h2_body)
            continue

        if h2_title == "6. What to implement next":
            hub.append("## 6. What to implement next")
            for sub_title, sub_body in split_h3(h2_body[1:]):
                if sub_title == "__intro__":
                    hub.extend(sub_body)
                elif is_hub_h3(sub_title, cfg):
                    hub.extend(sub_body)
                elif is_closed_heading(sub_title):
                    sat_closed.extend(sub_body)
                else:
                    sat_embedded.extend(sub_body)
            continue

        if h2_title.startswith(("2.", "3.", "4.", "5.")) or "Definition of Done" in h2_title:
            hub.append(h2_body[0])
            for sub_title, sub_body in split_h3(h2_body):
                if sub_title == "__intro__":
                    hub.extend(sub_body[1:] if sub_body and sub_body[0].startswith("## ") else sub_body)
                elif is_register_h3(sub_title):
                    sat_registers.extend(sub_body)
                elif "Layer scorecard" in sub_title:
                    hub.extend(sub_body)
                else:
                    hub.extend(sub_body)
            continue

        hub.extend(h2_body)

    SAT_DIR.mkdir(parents=True, exist_ok=True)
    satellites: dict[str, str] = {}
    d = cfg.domain

    # Preamble lines starting with # Audit Result → audit history
    if cfg.domain == "CRITIC_VERIFICATION":
        cleaned: list[str] = []
        audit_chunk: list[str] = []
        in_audit = False
        for line in hub:
            if line.startswith("# Audit Result:"):
                in_audit = True
                audit_chunk.append(line)
            elif in_audit and line.startswith("## "):
                in_audit = False
                sat_audit_history.extend(audit_chunk)
                audit_chunk = []
                cleaned.append(line)
            elif in_audit:
                audit_chunk.append(line)
            else:
                cleaned.append(line)
        if audit_chunk:
            sat_audit_history.extend(audit_chunk)
        hub = cleaned

    if sat_audit_history:
        fn = f"{d}_audit_history.md"
        satellites[fn] = render_satellite(d, "audit history + LC closeout", "\n".join(sat_audit_history))

    if sat_closed:
        fn = f"{d}_06_closed_queues.md"
        satellites[fn] = render_satellite(d, "closed §6 queues", "\n".join(sat_closed))
    if sat_registers:
        fn = f"{d}_master_registers.md"
        satellites[fn] = render_satellite(d, "master registers", "\n".join(sat_registers))
    if sat_embedded:
        fn = f"{d}_embedded_detail.md"
        satellites[fn] = render_satellite(d, "embedded detail", "\n".join(sat_embedded))
    if sat_phases:
        fn = f"{d}_phase_closeout.md"
        satellites[fn] = render_satellite(d, "phase closeout", "\n".join(sat_phases))
    if sat_appendices:
        fn = f"{d}_appendices.md"
        satellites[fn] = render_satellite(d, "appendices", "\n".join(sat_appendices))

    for fname, content in satellites.items():
        (SAT_DIR / fname).write_text(content, encoding="utf-8")

    hub_text = "\n".join(hub)
    hub_text = hub_text.replace(
        "read **only** the architecture doc and this plan doc for the domain.",
        "read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).",
    )
    if satellites:
        hub_text = insert_satellite_index(hub_text, satellite_index_rows(d, satellites))

    hub_text = dedupe_satellite_index(hub_text)

    # Shared appendix links → platform satellites when duplicated
    hub_text = re.sub(
        r"\[Appendix ([B-M])\]\(plan/" + re.escape(d) + r"\.md\)",
        r"[Appendix \1](plan/satellites/PLATFORM_FOUNDATION_appendices.md)",
        hub_text,
    )

    source.write_text(hub_text.rstrip() + "\n", encoding="utf-8")

    stats = {
        "hub_tokens": tokens(hub_text),
        "hub_lines": len(hub_text.splitlines()),
        "orig_tokens": tokens(raw),
        "sat_tokens": sum(tokens(c) for c in satellites.values()),
    }
    print(f"{cfg.domain}: hub {stats['hub_lines']} lines ~{stats['hub_tokens']:,} tok "
          f"(was ~{stats['orig_tokens']:,}, satellites ~{stats['sat_tokens']:,})")
    for fname in sorted(satellites):
        print(f"  wrote plan/satellites/{fname}")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split domain plan into hub + satellites")
    parser.add_argument(
        "domains",
        nargs="*",
        default=list(CONFIGS.keys()),
        help=f"Domain basenames (default: all configured: {', '.join(CONFIGS)})",
    )
    args = parser.parse_args(argv)
    for name in args.domains:
        if name not in CONFIGS:
            print(f"unknown domain config: {name}", file=sys.stderr)
            return 1
        split_plan(CONFIGS[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
