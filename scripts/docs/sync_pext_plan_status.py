#!/usr/bin/env python3
"""One-off sync: mark P-Ext narrative rows Done in intergrax_runtime_architecture.md."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs" / "project" / "architecture" / "intergrax_runtime_architecture.md


def main() -> int:
    text = PLAN.read_text(encoding="utf-8")
    lines: list[str] = []
    for line in text.splitlines():
        if "P-Ext." in line and "| **Planned** |" in line:
            line = line.replace("| **Planned** |", "| **Done** |")
        if "P-Ext." in line and "| **Partial** |" in line:
            line = line.replace("| **Partial** |", "| **Done** |")
        lines.append(line)
    text = "\n".join(lines) + "\n"
    replacements = [
        (
            "**Paydown open:** [P-Ext.6](#p-ext6--production-closure-paydown) + Appendix I",
            "**Done** (2026-06-02) — production closure complete; see Appendix I",
        ),
        (
            "| **Plugin catalogs (Phase P-Ext)** | **MVP Done** (paydown open) |",
            "| **Plugin catalogs (Phase P-Ext)** | **Done** (2026-06-02) |",
        ),
        (
            "**production closure:** [P-Ext.6](#p-ext6--production-closure-paydown) (**Planned**).",
            "**production closure:** **Done** (2026-06-02).",
        ),
        (
            "| **2c — Plugin catalogs (P-Ext)** | Entry points + `ToolPlugin` + `SkillPlugin` + `bootstrap_catalogs()` | **MVP Done** (paydown [P-Ext.6](#p-ext6--production-closure-paydown)) |",
            "| **2c — Plugin catalogs (P-Ext)** | Entry points + `ToolPlugin` + `SkillPlugin` + `bootstrap_catalogs()` | **Done** (2026-06-02) |",
        ),
        (
            "PARALLEL (harness-only): M.6 provider slugs on demand; R-Skill catalog expansion (platform packs); **Phase P-Ext.6 paydown** (plugin production closure)",
            "PARALLEL (harness-only): M.6 provider slugs on demand; R-Skill catalog expansion (platform packs)",
        ),
        (
            "**Status:** **Planned** · **not** default before gate green. Execute **one P-Ext.\\*** ID per PR when extending plugin catalogs.",
            "**Status:** **Done** (2026-06-02) · closure complete; extend catalogs via Appendix I + author guide.",
        ),
        (
            "**Status:** **MVP Done** (2026-06-02) · **paydown open** ·",
            "**Status:** **Done** (2026-06-02) ·",
        ),
        (
            "| **`resolve_typed.py`** | Exists: `resolve_relational_store`, `resolve_key_value_cache`, `resolve_document_parser` only | **Partial** — not used in lab wiring yet |",
            "| **`resolve_typed.py`** | Six typed helpers incl. vector_store, notification_channel, object_storage | **Done** |",
        ),
        (
            "| **Tier-3 bootstrap** | `tool_wiring` / `skill_wiring` → `bootstrap_catalogs()`; **lab/poc** `integration_wiring.py` still calls `register_default_integrations()` directly | **Gap** (P-Ext.1.10) |",
            "| **Tier-3 bootstrap** | `integration_wiring` / `tool_wiring` / `skill_wiring` → `bootstrap_catalogs()` + lazy bundle ids | **Done** |",
        ),
        (
            "| **Entry points** | Wired in `catalog_bootstrap`; `discover_entry_points=False` default; no fixture pip test | Paydown P-Ext.0.5 / 1.6 |",
            "| **Entry points** | Fixture pip package + EP tests; `INTERGRAX_DISCOVER_PLUGINS` for lab | **Done** |",
        ),
        (
            "| **Health API** | `integrations/registry/health.py` — **missing** | Optional (P-Ext.1.4) |",
            "| **Health API** | `integrations/registry/health.py` — `ping_integration` / `integration_registered` | **Done** |",
        ),
        (
            "| **Lazy catalog** | `bootstrap_catalogs(tool_bundle_ids=…)` supported; **`tool_wiring` does not pass profile bundles** (registers all 13, filters at `build_registry_from_profile`) | **Partial** (P-Ext.2.12) |",
            "| **Lazy catalog** | `tool_wiring` passes `tool_bundle_ids` from `ToolProfile` | **Done** |",
        ),
        (
            "| **Lazy catalog** | `bootstrap_catalogs(skill_bundle_ids=…)` supported; **`skill_wiring` does not pass profile bundles** (registers all 3, filters at `build_registry_from_profile`) | **Partial** (P-Ext.3.9) |",
            "| **Lazy catalog** | `skill_wiring` passes `skill_bundle_ids` from `SkillProfile` | **Done** |",
        ),
        (
            "**Status:** Architecture + catalog documentation **Done** (2026-06-02); implementation **Planned**.",
            "**Status:** **Done** (2026-06-02) — docs + implementation waves W-ML.0–W-ML.8.",
        ),
        (
            "**Default next:** §6.1 maintenance. **Optional parallel:** [§6.1p P-Ext paydown](#61p-phase-p-ext-paydown-band-2c--optional-parallel-with-61) (start P-Ext.0.5).",
            "**Default next:** §6.1 maintenance + Band 2 hardening (V-*, R-Skill expansion). P-Ext **Done**.",
        ),
        (
            "| Registry-driven extensibility (agent/tool/skill/policy/prompt/eval) | canon §7.1.5.1–§7.1.8, §15, §53.2 | ideal §19 | Phase R/U + V-CG/V-PE/V-EVAL + **P-Ext** | **MVP Done** — [P-Ext.6 paydown](#p-ext6--production-closure-paydown) for EP fixture + external tool/skill tests; marketplace UI out of scope |",
            "| Registry-driven extensibility (agent/tool/skill/policy/prompt/eval) | canon §7.1.5.1–§7.1.8, §15, §53.2 | ideal §19 | Phase R/U + V-CG/V-PE/V-EVAL + **P-Ext** | **Done** — plugin catalogs production-ready; marketplace UI out of scope |",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    PLAN.write_text(text, encoding="utf-8")
    print(f"Synced {PLAN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
