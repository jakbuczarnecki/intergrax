# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("new-skill", help="Create intergrax/skills/providers/<domain>/ manifest scaffold")
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
    manifest_path = provider_dir / "manifests.py"
    bundle_path = provider_dir / "bundle.py"
    usage_path = provider_dir / "USAGE.md"
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
    if not bundle_path.exists() or args.force:
        bundle_path.write_text(
            textwrap.dedent(
                f'''\
                # © Artur Czarnecki. All rights reserved.

                from intergrax.skills.providers.{domain}.manifests import {const}
                from intergrax.skills.registry.catalog import SkillBundleEntry, register_skill_bundle
                from intergrax.skills.registry.runtime import SkillRegistry


                def _register(registry: SkillRegistry) -> None:
                    registry.register({const})


                def register_{domain}_skill_bundle(*, override: bool = False) -> None:
                    register_skill_bundle(
                        SkillBundleEntry(
                            bundle_id="{domain}",
                            skill_ids=("{skill_id}",),
                            register=_register,
                            description="{domain} skill packs",
                        ),
                        override=override,
                    )
                '''
            ),
            encoding="utf-8",
        )
    if not usage_path.exists() or args.force:
        usage_path.write_text(
            f"# Skill provider `{domain}`\n\nRegister bundle in `intergrax/skills/registry/bootstrap.py`.\n",
            encoding="utf-8",
        )
    print(f"Created skill scaffold under {provider_dir}")
    return 0
