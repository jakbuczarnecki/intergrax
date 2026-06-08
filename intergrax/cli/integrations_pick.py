# © Artur Czarnecki. All rights reserved.

"""``intergrax integrations-pick`` — emit IntegrationProfile fragments (Phase DX-4.2)."""

from __future__ import annotations

import argparse
import sys

from intergrax.integrations.registry import presets


_PRESET_MAP = {
    "lab": presets.lab_stack,
    "legal": presets.legal_stack,
    "research": presets.research_stack,
    "data": presets.data_stack,
    "observability": presets.observability_stack,
    "harness_production": presets.harness_production_stack,
    "harness_metrics": presets.harness_metrics_stack,
    "harness_eval": presets.harness_eval_stack,
    "harness_async": presets.harness_async_stack,
    "harness_ci": presets.harness_ci_stack,
    "harness_security": presets.harness_security_stack,
    "harness_sandbox": presets.harness_sandbox_stack,
    "harness_identity": presets.harness_identity_stack,
    "harness_gitops": presets.harness_gitops_stack,
    "research_web": presets.research_web_stack,
    "document_ingest": presets.document_ingest_stack,
    "chat_bot": presets.chat_bot_stack,
}


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "integrations-pick",
        help="Print a named integration stack preset for environment_profile.py",
    )
    parser.add_argument(
        "preset",
        choices=sorted(_PRESET_MAP.keys()),
        help="Named stack preset",
    )


def run_pick(args: argparse.Namespace) -> int:
    factory = _PRESET_MAP[args.preset]
    profile = factory()
    print("# Paste into environment_profile.py")
    print(f"integration_profile = IntegrationProfile.model_validate({profile.model_dump(mode='json')!r})")
    print(f"# Or: IntegrationProfile.{args.preset}_stack()")
    return 0
