# © Artur Czarnecki. All rights reserved.
"""One-shot generator for SK-EXP4 per-skill USAGE.md files."""

from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "intergrax" / "skills" / "providers"

SK_EXP4_IDS = (
    "catalog.tool_introspect",
    "cloud_platform.resolver",
    "code.runner",
    "filesystem.local_io",
    "http.api_client",
    "interaction.session_handler",
    "interaction.input_capture",
    "jira.task_navigator",
    "gitlab.issue_creator",
    "ml.explain_predict",
    "openai.vector_admin",
    "browser.interactive_run",
    "cache.key_admin",
    "knowledge.confluence_navigator",
    "data.sql_mutator",
    "data.records_admin",
    "eval.observation_browser",
    "harness.run_comparator",
    "harness.run_exporter",
    "health.full_stack_probe",
    "ops.log_tail",
    "ops.incident_ack",
    "memory.semantic_search",
    "notify.batch_dispatch",
    "platform.secret_admin",
    "platform.workflow_cancel",
    "storage.object_lifecycle",
    "vector_store.purge",
    "modality.vision_segment",
    "research.web_cache_admin",
)

PRESETS: dict[str, str] = {
    "catalog": "catalog_skill_profile()",
    "cloud_platform": "cloud_platform_skill_profile()",
    "code": "code_skill_profile()",
    "filesystem": "filesystem_skill_profile()",
    "http": "http_skill_profile()",
    "interaction": "interaction_skill_profile()",
    "jira": "jira_skill_profile()",
    "gitlab": "gitlab_skill_profile()",
    "ml": "ml_skill_profile()",
    "openai": "openai_skill_profile()",
}

TEMPLATE = """# `{skill_id}`

**Bundle:** `{bundle}` · **Version:** 1.0.0 · **Risk:** `{risk}`

## Purpose

{purpose}

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

{preset_hint}Enable bundle `{bundle}` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: {tool_summary}.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
{tool_rows}

## Related skills

{related}
"""


def _load_manifest(skill_id: str):
    bundle, _ = skill_id.split(".", 1)
    mod = importlib.import_module(f"intergrax.skills.providers.{bundle}.manifests")
    const = skill_id.replace(".", "_").upper()
    return getattr(mod, const)


def main() -> None:
    bundle_skills: dict[str, list[str]] = {}
    for skill_id in SK_EXP4_IDS:
        m = _load_manifest(skill_id)
        bundle = skill_id.split(".", 1)[0]
        risk = m.risk_tier.value if hasattr(m.risk_tier, "value") else str(m.risk_tier)
        preset = PRESETS.get(bundle, f"SkillProfile(enabled_bundles=['{bundle}'])")
        tool_rows = "\n".join(f"| `{tid}` | Catalog tool |" for tid in m.tool_ids)
        related = "\n".join(f"- `{bundle}.*` peers in same bundle")
        path = ROOT / bundle / skill_id / "USAGE.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            TEMPLATE.format(
                skill_id=skill_id,
                bundle=bundle,
                risk=risk,
                purpose=m.description,
                preset_hint=f"Use `{preset}`; " if bundle in PRESETS else "",
                tool_summary=", ".join(f"`{t}`" for t in m.tool_ids),
                tool_rows=tool_rows,
                related=related,
            ),
            encoding="utf-8",
        )
        bundle_skills.setdefault(bundle, []).append(skill_id)

    for bundle, sids in bundle_skills.items():
        index_path = ROOT / bundle / "USAGE.md"
        existing = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        all_sids: set[str] = set(sids)
        if index_path.exists():
            for line in existing.splitlines():
                if line.startswith("| `") and "/USAGE.md)" in line:
                    all_sids.add(line.split("`")[1])
        lines = [f"# {bundle.title()} skill bundle", ""]
        if bundle in PRESETS and "SK-EXP4" not in existing:
            cls = "".join(p.capitalize() for p in bundle.split("_")) + "SkillPlugin"
            lines.append(f"**Bundle id:** `{bundle}` · **Plugin:** `{cls}` · SK-EXP4")
            lines.append("")
        lines.extend(["| skill_id | Guide |", "|----------|-------|"])
        for sid in sorted(all_sids):
            if (ROOT / bundle / sid / "USAGE.md").is_file():
                lines.append(f"| `{sid}` | [{sid}/USAGE.md]({sid}/USAGE.md) |")
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(SK_EXP4_IDS)} SK-EXP4 skill USAGE files")


if __name__ == "__main__":
    main()
