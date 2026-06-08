# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def register_parser(sub: argparse._SubParsersAction) -> None:
    for command in ("new-skill", "new-skill-bundle"):
        parser = sub.add_parser(
            command,
            help="Create intergrax/skills/providers/<domain>/ SkillPlugin scaffold",
        )
        parser.add_argument("skill_id", help="Skill id (e.g. legal.contract_review)")
        parser.add_argument("--domain", help="Provider folder (default: first segment of skill_id)")
        parser.add_argument("--root", type=Path, default=Path.cwd())
        parser.add_argument("--force", action="store_true")


def run_new_skill(args: argparse.Namespace) -> int:
    skill_id = args.skill_id.strip()
    if not skill_id or " " in skill_id:
        print("error: skill_id must be non-empty without spaces", flush=True)
        return 1
    domain = (args.domain or skill_id.split(".", 1)[0]).strip().lower()
    root = args.root.resolve()
    provider_dir = root / "intergrax" / "skills" / "providers" / domain
    if provider_dir.exists() and not args.force:
        print(f"error: {provider_dir} already exists (use --force)", flush=True)
        return 1
    provider_dir.mkdir(parents=True, exist_ok=True)
    const = skill_id.upper().replace(".", "_")
    class_name = "".join(part.title() for part in domain.split("_")) + "SkillPlugin"
    manifest_path = provider_dir / "manifests.py"
    plugin_path = provider_dir / "plugin.py"
    bundle_path = provider_dir / "bundle.py"
    usage_path = provider_dir / "USAGE.md"
    skill_usage_dir = provider_dir / skill_id
    skill_usage_path = skill_usage_dir / "USAGE.md"
    if not manifest_path.exists() or args.force:
        manifest_path.write_text(
            textwrap.dedent(
                f'''\
                # © Artur Czarnecki. All rights reserved.

                from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

                {const} = SkillManifest(
                    skill_id="{skill_id}",
                    version="1.0.0",
                    description="TODO: describe this skill pack",
                    tool_ids=(),
                    prompt_instruction_ids=(),
                    policy_fragment_id=None,
                    risk_tier=SkillRiskTier.MEDIUM,
                    tags=("{domain}",),
                )
                '''
            ),
            encoding="utf-8",
        )
    if not plugin_path.exists() or args.force:
        plugin_path.write_text(
            textwrap.dedent(
                f'''\
                # © Artur Czarnecki. All rights reserved.

                from __future__ import annotations

                from intergrax.skills.core.manifest import SkillBundleManifest
                from intergrax.skills.providers.{domain}.manifests import {const}
                from intergrax.skills.registry.catalog import SkillBundleStatus
                from intergrax.skills.registry.runtime import SkillRegistry


                class {class_name}:
                    @classmethod
                    def skill_bundle_manifest(cls) -> SkillBundleManifest:
                        return SkillBundleManifest(
                            bundle_id="{domain}",
                            skill_ids=("{skill_id}",),
                            status=SkillBundleStatus.BETA,
                            description="{domain} skill packs",
                        )

                    @classmethod
                    def skill_manifests(cls) -> tuple:
                        return ({const},)

                    @classmethod
                    def register_skills(cls, registry: SkillRegistry) -> None:
                        registry.register({const})
                '''
            ),
            encoding="utf-8",
        )
    if not bundle_path.exists() or args.force:
        bundle_path.write_text(
            textwrap.dedent(
                f'''\
                # © Artur Czarnecki. All rights reserved.

                from intergrax.skills.providers.{domain}.plugin import {class_name}
                from intergrax.skills.registry.plugin_register import register_skill_plugin


                def register_{domain}_skill_bundle(*, override: bool = False) -> None:
                    register_skill_plugin({class_name}, override=override)
                '''
            ),
            encoding="utf-8",
        )
    if not usage_path.exists() or args.force:
        usage_path.write_text(
            f"# {domain.title()} skill bundle\n\n"
            f"| skill_id | Guide |\n"
            f"|----------|-------|\n"
            f"| `{skill_id}` | [{skill_id}/USAGE.md]({skill_id}/USAGE.md) |\n",
            encoding="utf-8",
        )
    if not skill_usage_path.exists() or args.force:
        skill_usage_dir.mkdir(parents=True, exist_ok=True)
        skill_usage_path.write_text(
            textwrap.dedent(
                f'''\
                # `{skill_id}`

                **Bundle:** `{domain}` · **Version:** 1.0.0 · **Risk:** `medium`

                ## Purpose

                TODO: Why this skill exists and which agents/hosts should use it.

                ## How it works

                TODO: Registration → SkillResolver → allowed_tools; list resolved `tool_ids`.

                ## How to use

                ```python
                from intergrax.skills.providers.{domain}.manifests import {const}
                from intergrax.skills.registry.profile import SkillProfile

                AgentContract(id="my_agent", skills=[{const}], ...)
                ```

                ## What you get

                TODO: Reusable allow-list, conformance, traceability benefits.

                ## Tools unlocked

                | `tool_id` | Role |
                |-----------|------|
                | TODO | TODO |

                ## Related skills

                - TODO
                '''
            ),
            encoding="utf-8",
        )
    print(f"Created skill scaffold under {provider_dir}")
    return 0
