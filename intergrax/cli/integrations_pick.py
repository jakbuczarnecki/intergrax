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
