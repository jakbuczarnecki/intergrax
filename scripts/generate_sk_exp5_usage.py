# © Artur Czarnecki. All rights reserved.
"""Generate SK-EXP5 per-skill USAGE.md files."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "intergrax" / "skills" / "providers"

SK_EXP5_IDS = tuple(
    skill_id
    for bundle in (
        "rag",
        "legal",
        "research",
        "workspace",
        "memory",
        "ops",
        "dev",
        "platform",
        "collaboration",
        "data",
        "hitl",
        "graph",
        "sandbox",
        "storage",
        "message_bus",
        "cache",
        "eval",
        "modality",
        "notify",
        "cost",
        "identity",
        "health",
        "filesystem",
        "harness",
        "agent",
    )
    for skill_id in (
        __import__(
            f"intergrax.skills.providers.{bundle}.manifests",
            fromlist=["_"],
        ).__dict__
    )
    if False
)

# Build list from manifests files directly
SK_EXP5_IDS_LIST: list[str] = []
for bundle_dir in sorted(ROOT.iterdir()):
    if not bundle_dir.is_dir():
        continue
    manifests = bundle_dir / "manifests.py"
    if not manifests.is_file():
        continue
    for m in re.finditer(r'skill_id="([^"]+)"', manifests.read_text(encoding="utf-8")):
        sid = m.group(1)
        if sid not in SK_EXP5_IDS_LIST:
            pass
    # only SK-EXP5: skills with USAGE not yet or we track by known prefix from scaffold
EXP5_SUFFIXES = {
    "semantic_qa", "ingest_pipeline", "metadata_search", "redline_draft", "regulatory_scan",
    "obligation_tracker", "deep_dive", "source_validator", "report_compiler", "draft_reviewer",
    "artifact_exporter", "cross_turn_notes", "fact_extractor", "oncall_runbook", "postmortem_writer",
    "change_approver", "capacity_planner", "pr_reviewer", "release_notes", "sprint_planner",
    "runbook_hub", "flag_rollout", "deploy_inspector", "meeting_brief", "stakeholder_ping",
    "pipeline_probe", "schema_documenter", "escalation_router", "decision_auditor", "path_finder",
    "knowledge_linker", "test_runner", "refactor_loop", "backup_sync", "presigned_share",
    "retry_handler", "dead_letter", "warm_prefetch", "baseline_runner", "regression_guard",
    "audio_transcript", "image_analyst", "escalation_ladder", "chargeback_report",
    "session_bootstrap", "identity_probe", "stat_auditor", "cost_analyst", "integration_sweep",
    "capability_mapper",
}
for bundle_dir in sorted(ROOT.iterdir()):
    manifests = bundle_dir / "manifests.py"
    if not manifests.is_file():
        continue
    for m in re.finditer(r'skill_id="([^"]+)"', manifests.read_text(encoding="utf-8")):
        sid = m.group(1)
        if sid.split(".", 1)[-1] in EXP5_SUFFIXES:
            SK_EXP5_IDS_LIST.append(sid)

TEMPLATE = """# `{skill_id}`

**Bundle:** `{bundle}` · **Version:** 1.0.0 · **Risk:** `{risk}`

## Purpose

{purpose}

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `{bundle}` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: {tool_summary}.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
{tool_rows}

## Related skills

- Other `{bundle}` bundle skills — see bundle [USAGE.md](../USAGE.md)
"""


def _load_manifest(skill_id: str):
    bundle, _ = skill_id.split(".", 1)
    mod = importlib.import_module(f"intergrax.skills.providers.{bundle}.manifests")
    const = skill_id.replace(".", "_").upper()
    return getattr(mod, const)


def main() -> None:
    for skill_id in sorted(SK_EXP5_IDS_LIST):
        m = _load_manifest(skill_id)
        bundle = skill_id.split(".", 1)[0]
        risk = m.risk_tier.value if hasattr(m.risk_tier, "value") else str(m.risk_tier)
        path = ROOT / bundle / skill_id / "USAGE.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        tool_rows = "\n".join(f"| `{tid}` | Catalog tool |" for tid in m.tool_ids)
        path.write_text(
            TEMPLATE.format(
                skill_id=skill_id,
                bundle=bundle,
                risk=risk,
                purpose=m.description,
                tool_summary=", ".join(f"`{t}`" for t in m.tool_ids),
                tool_rows=tool_rows,
            ),
            encoding="utf-8",
        )

    for bundle in {s.split(".", 1)[0] for s in SK_EXP5_IDS_LIST}:
        index_path = ROOT / bundle / "USAGE.md"
        existing = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        all_sids: set[str] = set()
        manifests = ROOT / bundle / "manifests.py"
        for m in re.finditer(r'skill_id="([^"]+)"', manifests.read_text(encoding="utf-8")):
            all_sids.add(m.group(1))
        lines = [f"# {bundle.title()} skill bundle", ""]
        if "SK-EXP5" not in existing:
            cls = "".join(p.capitalize() for p in bundle.split("_")) + "SkillPlugin"
            lines.append(f"**Bundle id:** `{bundle}` · **Plugin:** `{cls}` · SK-EXP5 extended")
            lines.append("")
        lines.extend(["| skill_id | Guide |", "|----------|-------|"])
        for sid in sorted(all_sids):
            if (ROOT / bundle / sid / "USAGE.md").is_file():
                lines.append(f"| `{sid}` | [{sid}/USAGE.md]({sid}/USAGE.md) |")
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(SK_EXP5_IDS_LIST)} SK-EXP5 skill USAGE files")


if __name__ == "__main__":
    main()
